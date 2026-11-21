from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy import signal

from .audition import RATES, read_wav, wav_bytes
from .contracts import flag, identifier, integer, mapping, option, sequence, shipment_stem, split_rows, text
from .errors import BenchError, Conflict, Limit
from .store import CUE_KINDS, CUE_STATES
from .timing import format_side_time


MAX_PLAYLIST = 12
MAX_CONTINUOUS_DURATION = 90


@dataclass(frozen=True)
class PlaylistItem:
    take_id: str
    cue_id: str | None
    label: str | None
    gap_ms: int

    @classmethod
    def parse(cls, raw):
        value = mapping(raw, "监听串联项")
        take = identifier(value.get("take"), "监听编号")
        cue = value.get("cue")
        if cue is not None and cue != "":
            cue = identifier(cue, "标记编号")
        else:
            cue = None
        label = text(value.get("label", ""), "试听标签", maximum=85, required=False) or None
        gap = integer(value.get("gap_ms", 150), "片段前留白（毫秒）", 0, 2000)
        return cls(take, cue, label, gap)


@dataclass(frozen=True)
class PacketRequest:
    label: str
    items: tuple[PlaylistItem, ...]
    proofs: tuple[str, ...]
    sources: bool
    continuous: bool
    rate: int
    include_cues: bool

    @classmethod
    def parse(cls, raw):
        values = mapping(raw, "交付设置")
        title = text(values.get("title", ""), "交付标题", maximum=100)
        items = sequence(values.get("items", []), "监听顺序", maximum=MAX_PLAYLIST)
        proofs = sequence(values.get("proofs", []), "图像证据", maximum=20)
        proofs = tuple(dict.fromkeys(identifier(value, "证据编号") for value in proofs))
        source = flag(values.get("include_sources", False), "纳入原始扫描")
        continuous = flag(values.get("continuous", True), "制作串听文件")
        cue = flag(values.get("include_cues", True), "纳入标记表")
        rate = integer(values.get("rate", 24000), "串听采样率", min(RATES), max(RATES))
        if rate not in RATES:
            raise BenchError("串听采样率须为标准 WAV 采样率。")
        if not items and not proofs and not cue:
            raise BenchError("请选择监听、图像证据或审听标记之后再交付。")
        return cls(title, tuple(PlaylistItem.parse(item) for item in items), proofs, source, continuous, rate, cue)


def parse_cue(values, height):
    values = mapping(values, "审听标记")
    title = text(values.get("title", ""), "标记名", maximum=90)
    start, stop = split_rows(values.get("rows"), height)
    kind = option(values.get("kind", "excerpt"), CUE_KINDS, "标记类别")
    status = option(values.get("status", "open"), CUE_STATES, "标记状态")
    note = text(values.get("note", ""), "说明", maximum=600, required=False, multiline=True)
    picked_take = values.get("take") or None
    anchor = {"take": identifier(picked_take, "监听编号")} if picked_take is not None else None
    previous = values.get("previous_updated")
    if previous is not None:
        previous = text(previous, "原更新时刻", maximum=45)
    return {"title": title, "y0": start, "y1": stop, "kind": kind, "status": status, "note": note,
            "anchor": anchor, "previous_updated": previous}


def validate_cue_anchor(cue, take, scan_id):
    if take["scan_id"] != scan_id:
        raise Conflict("标记与监听快照必须来自同一张扫描。")
    a, b = take["payload"]["audio"]["source_rows"]
    if not a <= cue["y0"] < cue["y1"] <= b:
        raise Conflict("标记不在所选监听的原生扫描行范围内。")
    details = take["payload"]["clock"]
    scale = details["rows_per_second"]
    if details["reverse"]:
        start, stop = (b - cue["y1"]) / scale, (b - cue["y0"]) / scale
    else:
        start, stop = (cue["y0"] - a) / scale, (cue["y1"] - a) / scale
    return {"take": take["id"], "recipe": take["recipe_id"], "clock": details["source"],
            "take_offsets": [round(start, 8), round(stop, 8)]}


def cue_coverage(cues, rows=None):
    by_state = {state: 0 for state in CUE_STATES}
    by_kind = {kind: 0 for kind in CUE_KINDS}
    overlaps = []
    previous = []
    active_scan = None
    for cue in cues:
        if rows and not (max(rows[0], cue["y0"]) < min(rows[1], cue["y1"])):
            continue
        by_state[cue["status"]] += 1
        by_kind[cue["kind"]] += 1
        if cue["scan_id"] != active_scan:
            active_scan = cue["scan_id"]
            previous.clear()
        if cue["status"] == "omit":
            continue
        previous = [other for other in previous if other["y1"] > cue["y0"]]
        for other in previous:
            common = (max(other["y0"], cue["y0"]), min(other["y1"], cue["y1"]))
            if common[0] < common[1] and len(overlaps) < 60:
                overlaps.append({"a": other["id"], "b": cue["id"], "rows": list(common),
                                 "a_label": other["title"], "b_label": cue["title"]})
        previous.append(cue)
    return {"states": by_state, "kinds": by_kind, "visible_count": sum(by_state.values()), "overlaps": overlaps}


def original_span(take):
    return take["payload"]["audio"]["source_rows"]


def clip_cut(take, cues, requested, wave, rate):
    if requested is None:
        return wave, tuple(original_span(take)), (0, len(wave)), None
    cue = cues.get(requested)
    if cue is None:
        raise Conflict("监听串联引用了该卷中不存在的审听标记。")
    if cue["status"] == "omit":
        raise Conflict(f"标记“{cue['title']}”已设为略过。")
    position = validate_cue_anchor(cue, take, take["scan_id"])
    at, stop = [max(0, min(len(wave), round(point * rate))) for point in position["take_offsets"]]
    if at >= stop:
        raise BenchError("标记小于当前监听输出中的一个样本。")
    return wave[at:stop], (cue["y0"], cue["y1"]), (at, stop), cue


def conform_clip(values, original_rate, appended_rate):
    if values.ndim != 1:
        raise BenchError("串听须用声轨生成的单声道监听。")
    if original_rate == appended_rate:
        return values
    ratio = Fraction(appended_rate, original_rate)
    count = max(1, round(len(values) * appended_rate / original_rate))
    if len(values) < 3:
        return np.full(count, values.mean())
    windowed = signal.resample_poly(values, ratio.numerator, ratio.denominator, padtype="line")
    return windowed[:count]


def expand_items(items, takes, cues, output_rate, continuous):
    clip_manifest = []
    assets = []
    compiled = []
    cursor = 0
    for n, item in enumerate(items, 1):
        take = takes.get(item.take_id)
        if take is None:
            raise Conflict("一个监听快照已不可用。")
        if take["status"] != "reviewed":
            raise Conflict(f"监听“{take['title']}”须标记为已复核。")
        waves, rate = read_wav(take["wav"])
        excerpt, rows, samples, cue = clip_cut(take, cues, item.cue_id, waves, rate)
        title = item.label or (cue["title"] if cue else take["title"])
        length = len(excerpt)
        if continuous:
            gap = round(output_rate * item.gap_ms / 1000) if n != 1 else 0
            converted = conform_clip(excerpt, rate, output_rate)
            start = cursor + gap
            stop = start + len(converted)
            if stop > output_rate * MAX_CONTINUOUS_DURATION:
                raise Limit(f"串听文件总长最多 {MAX_CONTINUOUS_DURATION} 秒。")
            if gap:
                compiled.append(np.zeros(gap, dtype=np.float64))
            compiled.append(converted)
            timestamp = {"samples": [start, stop], "rate": output_rate, "timecodes": [format_side_time(start / output_rate),
                                                                                  format_side_time(stop / output_rate)]}
            cursor = stop
        else:
            timestamp = None
        filename = f"audio/{n:02d}-{shipment_stem(title)}.wav"
        assets.append((filename, wav_bytes(excerpt, rate) if cue else take["wav"], "audio/wav", "listen_clip"))
        clip_manifest.append({"index": n, "title": title, "file": filename, "take": take["id"],
                              "take_sha256": take["wav_sha256"], "recipe_id": take["recipe_id"], "scan": take["scan_id"],
                              "cue_id": cue["id"] if cue else None, "source_rows": list(rows), "take_samples": list(samples),
                              "clip_sample_rate": rate, "clip_frames": length, "continuous": timestamp})
    if compiled:
        assets.append(("audio/continuous-review.wav", wav_bytes(np.concatenate(compiled), output_rate), "audio/wav", "continuous"))
    return clip_manifest, assets
