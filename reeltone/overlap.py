"""Longitudinal overlap joining for two passes of the same soundtrack strip.

The two scans share width and walking direction; only their longitudinal
position differs.  The module estimates the offset between native rows of the
two passes, proposes several plausible alignments, scores every aligned row of
the overlap for local clarity (dust, scratches and signal quality), and plans
a seam path that guarantees every physical film row enters the resulting band
exactly once.  The assembled band is a normal raster that the existing reader
can open, while every output row keeps a provenance pointer back to the
original scan and native row it came from.

Offset convention: tail native row ``t`` depicts the same physical position as
head native row ``t + offset`` (equivalently head row ``h`` == tail row
``h - offset``).
"""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage, signal

from .contracts import integer, mapping, option
from .errors import BenchError, Conflict, Limit
from .raster import encode_image
from .visual import ACCENT, BLUE, MUTED, PAGE, PANEL, RED, TEAL, print_text
from PIL import Image, ImageDraw


MAX_CANDIDATES = 6
MIN_OVERLAP_ROWS = 24
DEFAULT_CROSSFADE = 6


@dataclass(frozen=True)
class Alignment:
    offset: int
    score: float
    secondary_score: float
    overlap_rows: int
    accepted_rows: int

    def as_dict(self):
        return {"offset": self.offset, "score": round(self.score, 6),
                "weighted_score": round(self.weighted, 6),
                "secondary_score": round(self.secondary_score, 6),
                "overlap_rows": self.overlap_rows, "accepted_rows": self.accepted_rows}

    @property
    def weighted(self):
        return self.score * _support_weight(self.overlap_rows)


@dataclass(frozen=True)
class Segment:
    start: int
    stop: int
    source: str
    source_scan_id: str
    source_rows: tuple[int, int]
    reason: str

    def as_dict(self):
        return {"output_start": self.start, "output_stop": self.stop, "source": self.source,
                "source_scan_id": self.source_scan_id, "source_first_row": self.source_rows[0],
                "source_after_last_row": self.source_rows[1], "rows": self.stop - self.start,
                "reason": self.reason}


def _row_feature(image, columns):
    """Robust per-row descriptor inside a column guide.

    A trimmed mean across columns keeps the row-to-row track modulation while
    discarding the darkest and brightest pixels (dust specks and scratches);
    a light median pass then suppresses single-row impulse noise.
    """
    band = image[:, columns[0]:columns[1]].astype(np.float64)
    trimmed = np.mean(np.sort(band, axis=1)[:, max(1, band.shape[1] // 8):-max(1, band.shape[1] // 8)], axis=1)
    return ndimage.median_filter(trimmed, size=3, mode="nearest")


def _standardize(values):
    centered = values - np.mean(values)
    scale = np.sqrt(np.mean(centered ** 2))
    return centered / scale if scale > 1e-9 else centered


def _offset_bounds(head_h, tail_h, head_mark, tail_mark):
    if head_mark is not None and tail_mark is not None:
        h0, h1 = head_mark
        t0, t1 = tail_mark
        nominal = h0 - t0
        slack = max(48, (h1 - h0) + (t1 - t0))
        low = max(-(tail_h - MIN_OVERLAP_ROWS), nominal - slack)
        high = min(head_h - MIN_OVERLAP_ROWS, nominal + slack)
    else:
        low, high = -(tail_h - MIN_OVERLAP_ROWS), head_h - MIN_OVERLAP_ROWS
    if high < low:
        raise BenchError("标出的重叠段无法产生至少 %d 行的共同区间，请重新在两张图上框选重叠区域。" % MIN_OVERLAP_ROWS)
    return low, high


def _ncc_at(feature_a, feature_b, offset):
    if offset >= 0:
        a, b = feature_a[offset:], feature_b
    else:
        a, b = feature_a, feature_b[-offset:]
    length = min(len(a), len(b))
    if length < MIN_OVERLAP_ROWS:
        return None, length
    a, b = _standardize(a[:length]), _standardize(b[:length])
    return float(np.dot(a, b) / length), length


def _support_weight(overlap_rows):
    """Down-rank alignments whose common region is too short to be trustworthy."""
    return float(min(1.0, np.log1p(overlap_rows) / np.log1p(480)))


def search_alignments(head, tail, head_mark=None, tail_mark=None, limit=MAX_CANDIDATES):
    if head.width != tail.width:
        raise BenchError("两张扫描的列数不同。两次扫描未调整宽度时才能做纵向接续。")
    if head.recipe.orientation.as_dict() != tail.recipe.orientation.as_dict():
        raise Conflict("两张扫描的走片方向设置不同。请先在各自配方里统一镜像与反向设置。")
    if tail.recipe.aperture.as_list() != head.recipe.aperture.as_list():
        raise Conflict("两张扫描的声轨读窗列范围不同。请先统一读片配方的读窗。")
    track_a = _row_feature(head.pixels, head.recipe.aperture.as_list())
    track_b = _row_feature(tail.pixels, tail.recipe.aperture.as_list())
    perf_a = _row_feature(head.pixels, head.recipe.perforations.as_list())
    perf_b = _row_feature(tail.pixels, tail.recipe.perforations.as_list())
    low, high = _offset_bounds(head.height, tail.height, head_mark, tail_mark)
    scored = []
    for offset in range(low, high + 1):
        score, overlap = _ncc_at(track_a, track_b, offset)
        if score is None:
            continue
        second, _ = _ncc_at(perf_a, perf_b, offset)
        scored.append((offset, score, second or 0.0, overlap))
    if not scored:
        raise BenchError("标出的范围内没有足够长的共同行，无法估计重叠。")
    scores = np.array([row[1] for row in scored])
    peaks, _ = signal.find_peaks(np.r_[scores.min(), scores, scores.min()], distance=24)
    peak_offsets = {scored[index - 1][0] for index in peaks - 1} if len(peaks) else set()
    # Rank primarily by overlap-supported NCC; raw peak status and perforation
    # agreement only break near-ties so short coincidental matches cannot win.
    ranked = sorted(scored, key=lambda row: (round(row[1] * _support_weight(row[3]), 5),
                                             row[0] in peak_offsets, row[2]), reverse=True)
    chosen = []
    for offset, score, second, overlap in ranked:
        if any(abs(offset - item.offset) < 24 for item in chosen):
            continue
        chosen.append(Alignment(offset, score, second, overlap, head.height + tail.height - overlap))
        if len(chosen) >= limit:
            break
    if not chosen:
        raise BenchError("未能在标出的重叠段中找到稳定的对应关系。")
    curve = [{"offset": int(offset), "score": round(float(value), 6),
              "weighted_score": round(float(value) * _support_weight(
                  min(head.height - max(0, offset), tail.height - max(0, -offset))), 6)}
             for offset, value in zip(np.arange(low, high + 1), scores)]
    return chosen, curve


def overlap_extent(head_h, tail_h, offset):
    """Head rows, tail rows, and output-row span of the shared region."""
    head_lo, tail_lo = max(0, offset), max(0, -offset)
    overlap = min(head_h - head_lo, tail_h - tail_lo)
    if overlap < MIN_OVERLAP_ROWS:
        raise BenchError("该对齐的共同行不足，无法接续。")
    return (head_lo, head_lo + overlap), (tail_lo, tail_lo + overlap), (head_lo, head_lo + overlap)


def tonal_match(head_pixels, tail_pixels, head_rows, tail_rows):
    """Linear map bringing the tail's tonal scale onto the head's."""
    a = head_pixels[head_rows[0]:head_rows[1]]
    b = tail_pixels[tail_rows[0]:tail_rows[1]]
    qa = np.quantile(a.ravel(), np.linspace(0.02, 0.98, 49))
    qb = np.quantile(b.ravel(), np.linspace(0.02, 0.98, 49))
    center_a, center_b = qa.mean(), qb.mean()
    var_b = np.mean((qb - center_b) ** 2)
    gain = float(np.mean((qa - center_a) * (qb - center_b)) / var_b) if var_b > 1e-12 else 1.0
    gain = float(np.clip(gain, 0.5, 2.0))
    return gain, float(center_a - gain * center_b)


def clarity_rows(image, rows, columns):
    """Absolute defect score; lower means a cleaner, more trustworthy row.

    Structural track modulation runs across all columns of a row, so a vertical
    median preserves it while localised defects (dust covering only some
    columns, or a narrow scratch) remain in the residual.  Rows that lose tonal
    information at either code limit are additionally penalised.
    """
    band = image[rows[0]:rows[1], columns[0]:columns[1]].astype(np.float64)
    row_level = np.median(band, axis=1, keepdims=True)
    horizontal = ndimage.median_filter(band, size=(1, 7), mode="reflect")
    anomaly = np.abs(band - horizontal)
    # A scratch affects a sustained column range: compare against the row median
    # as well so column-wide darkening is not mistaken for clean structure.
    residual = np.abs(band - row_level)
    defect = np.percentile(anomaly, 92, axis=1) + 0.5 * np.percentile(residual, 95, axis=1)
    clipped = np.mean((band <= 0.002) | (band >= 0.998), axis=1)
    return (defect + 1.2 * clipped).astype(np.float64)


def paired_clarity(head_pixels, tail_pixels, head_rows, tail_rows, columns):
    """Return (head_clear, tail_clear) in (0, 1], calibrated against each other.

    Because the two rows depict the same physical position, their ordinary
    signal is identical and the relative defect level decides the cleaner side.
    A shared baseline derived from both scans keeps the scale comparable even
    when one pass has a different exposure or noise floor.
    """
    head_defect = clarity_rows(head_pixels, head_rows, columns)
    tail_defect = clarity_rows(tail_pixels, tail_rows, columns)
    baseline = max(float(np.median(np.minimum(head_defect, tail_defect))), 1e-6)
    scale = max(float(np.percentile(np.concatenate((head_defect, tail_defect)), 90)), baseline * 1.2)
    def to_score(defect):
        normalised = np.clip((defect - baseline) / max(scale - baseline, 1e-9), 0.0, 1.0)
        return np.exp(-2.4 * normalised)
    return to_score(head_defect), to_score(tail_defect)


def _runs(values):
    text_values = values.astype(str)
    changes = np.flatnonzero(text_values[1:] != text_values[:-1]) + 1
    edges = np.r_[0, changes, len(text_values)]
    return [(int(edges[i]), int(edges[i + 1])) for i in range(len(edges) - 1)]


def _dwell_filter(preference, min_segment):
    """Enforce a minimum dwell length without erasing a long-lived preference.

    A block-majority vote over windows of ``min_segment`` rows first removes
    row-scale flicker caused by measurement noise, then any remaining
    sub-minimum runs are absorbed into their longer neighbour.
    """
    min_segment = max(4, min_segment)
    length = len(preference)
    voted = preference.copy()
    for start in range(0, length, min_segment):
        block = preference[start:min(length, start + min_segment)]
        voted[start:start + len(block)] = "head" if np.sum(block == "head") >= np.sum(block == "tail") else "tail"
    result = voted
    for _ in range(length + 1):
        runs = _runs(result)
        if all(stop - start >= min_segment for start, stop in runs):
            break
        start, stop = min((run for run in runs if run[1] - run[0] < min_segment),
                          key=lambda run: run[1] - run[0])
        index = next(i for i, run in enumerate(runs) if run[0] == start and run[1] == stop)
        left = runs[index - 1] if index > 0 else None
        right = runs[index + 1] if index + 1 < len(runs) else None
        if left is None and right is None:
            break
        if left is None:
            fill = result[right[0]]
        elif right is None:
            fill = result[left[0]]
        elif (right[1] - right[0]) >= (left[1] - left[0]):
            fill = result[right[0]]
        else:
            fill = result[left[0]]
        result[start:stop] = fill
    return result


def plan_seams(head, tail, alignment, *, mode="quality", min_segment=48, crossfade=DEFAULT_CROSSFADE):
    offset = alignment.offset
    head_rows, tail_rows, (out_lo, out_hi) = overlap_extent(head.height, tail.height, offset)
    columns = head.recipe.aperture.as_list()
    head_clear, tail_clear = paired_clarity(head.pixels, tail.pixels, head_rows, tail_rows, columns)
    overlap_len = head_rows[1] - head_rows[0]
    if mode == "single":
        winners = np.where(np.arange(overlap_len) < overlap_len // 2, "head", "tail")
        reasons = np.full(overlap_len, "single_seam", dtype=object)
        preference = winners
    else:
        # Only switch when one side is meaningfully clearer; near-ties keep the
        # current side so ordinary read noise cannot create extra splices.
        margin = np.abs(head_clear - tail_clear)
        confident = margin > max(0.04, float(np.median(margin)) * 1.5)
        preference = np.where(head_clear >= tail_clear, "head", "tail")
        winners = _dwell_filter(preference, min_segment)
        for index in np.flatnonzero(~confident):
            winners[index] = winners[index - 1] if index > 0 else "head"
        reasons = np.where(winners == preference, "clearer_side", "segment_stability")
    crossfade = integer(crossfade, "过渡行数", 0, 40)
    segments, blends = _build_segments(head, tail, offset, out_lo, out_hi, winners, reasons, crossfade)
    stats = {"head_clarity_median": round(float(np.median(head_clear)), 5),
             "tail_clarity_median": round(float(np.median(tail_clear)), 5),
             "head_rows_preferred": int(np.sum(preference == "head")),
             "tail_rows_preferred": int(np.sum(preference == "tail")),
             "switch_count": int(len(blends)), "crossfade_rows": crossfade}
    return segments, blends, stats, (head_clear, tail_clear), (head_rows, tail_rows)


def _native_row(source, output_row, offset):
    return output_row if source == "head" else output_row - offset


def _build_segments(head, tail, offset, out_lo, out_hi, winners, reasons, crossfade):
    """Partition output rows into contiguous single-source segments.

    Blend windows are reported separately for assembly, but every output row
    belongs to exactly one segment, so a physical film row never enters twice.
    """
    total_h = head.height + tail.height - (out_hi - out_lo)
    segments, blends = [], []

    def add(start, stop, source, reason):
        if stop <= start:
            return
        native = _native_row(source, start, offset)
        scan_id = head.record["id"] if source == "head" else tail.record["id"]
        segments.append(Segment(start, stop, source, scan_id, (native, native + (stop - start)), str(reason)))

    add(0, out_lo, "head", "head_only")
    cursor = out_lo
    for start, stop in _runs(winners):
        source = str(winners[start])
        seg_lo, seg_hi = out_lo + start, out_lo + stop
        if crossfade and start > 0:
            previous = str(winners[start - 1])
            half = crossfade // 2
            cut = seg_lo + half
            add(cursor, cut, previous, "crossfade")
            blends.append({"output_start": cut - half, "output_stop": cut + (crossfade - half),
                           "from": previous, "to": source, "rows": crossfade})
            cursor = cut
        if crossfade and stop < len(winners):
            cut = seg_hi - crossfade // 2
            add(cursor, cut, source, reasons[start])
            cursor = cut
        else:
            add(cursor, seg_hi, source, reasons[start])
            cursor = seg_hi
    if cursor < out_hi:
        add(cursor, out_hi, str(winners[-1]), reasons[-1])
    add(out_hi, total_h, "tail", "tail_only")
    return _merge_adjacent(segments), blends


def _merge_adjacent(segments):
    merged = []
    for segment in segments:
        if merged and merged[-1].source == segment.source and merged[-1].stop == segment.start:
            previous = merged.pop()
            segment = Segment(previous.start, segment.stop, segment.source, segment.source_scan_id,
                              (previous.source_rows[0], previous.source_rows[1] + segment.stop - segment.start),
                              previous.reason)
        merged.append(segment)
    return merged


def _fetch(material, source, output_start, count, offset, gain, bias):
    native = _native_row(source, output_start, offset)
    pixels = material.pixels[native:native + count].astype(np.float64)
    if source == "tail":
        pixels = np.clip(pixels * gain + bias, 0.0, 1.0)
    return pixels


def assemble_band(head, tail, alignment, segments, blends):
    height = head.height + tail.height - alignment.overlap_rows
    if height * head.width > 14_000_000:
        raise Limit("接续后的连续边带超过 1,400 万像素上限。")
    head_rows, tail_rows = _overlap_rows(head, tail, alignment.offset)
    gain, bias = tonal_match(head.pixels, tail.pixels, head_rows, tail_rows)
    band = np.empty((height, head.width), dtype=np.float32)
    used = np.zeros(height, dtype=bool)
    for segment in segments:
        a, b = segment.start, segment.stop
        material = head if segment.source == "head" else tail
        band[a:b] = _fetch(material, segment.source, a, b - a, alignment.offset, gain, bias).astype(np.float32)
        if used[a:b].any():
            raise Conflict("接续方案存在重复行，已中止。")
        used[a:b] = True
    for blend in blends:
        a, b = max(0, blend["output_start"]), min(height, blend["output_stop"])
        if b <= a:
            continue
        length = b - a
        weights = np.linspace(0.0, 1.0, length, dtype=np.float64)[:, None]
        before = _fetch(head if blend["from"] == "head" else tail, blend["from"], a, length, alignment.offset, gain, bias)
        after = _fetch(head if blend["to"] == "head" else tail, blend["to"], a, length, alignment.offset, gain, bias)
        band[a:b] = (before * (1 - weights) + after * weights).astype(np.float32)
    if not used.all():
        raise Conflict("接续方案遗漏了输出行，已中止。")
    return band, {"tonal_gain": round(gain, 6), "tonal_bias": round(bias, 6)}


def _overlap_rows(head, tail, offset):
    head_span, tail_span, _ = overlap_extent(head.height, tail.height, offset)
    return head_span, tail_span


def join_plate(head, tail, alignment, segments, clarity):
    head_clear, tail_clear = clarity
    width = 1080
    canvas = Image.new("RGB", (width, 470), PAGE)
    draw = ImageDraw.Draw(canvas)
    print_text(draw, (22, 18), "REELTONE  /  OVERLAP JOIN PLAN", 17, TEAL)
    print_text(draw, (22, 46), f"offset {alignment.offset:+d} rows  ·  overlap {alignment.overlap_rows} rows  ·  "
                               f"NCC {alignment.score:.4f}", 13, MUTED)
    track = (36, 96, width - 36, 250)
    draw.rectangle(track, fill=PANEL)
    height = head.height + tail.height - alignment.overlap_rows
    colors = {"head": TEAL, "tail": ACCENT}
    for segment in segments:
        x0 = track[0] + round(segment.start / height * (track[2] - track[0]))
        x1 = track[0] + round(segment.stop / height * (track[2] - track[0]))
        draw.rectangle((x0, track[1], max(x0 + 1, x1), track[3]), fill=colors[segment.source])
    print_text(draw, (track[0], track[3] + 10), f"head only · {head.record['title'][:40]}", 11, TEAL)
    print_text(draw, (track[2], track[3] + 10), f"tail only · {tail.record['title'][:40]}", 11, ACCENT, anchor="ra")
    plot = (36, 292, width - 36, 446)
    draw.rectangle(plot, fill=PANEL)
    print_text(draw, (plot[0] + 8, plot[1] + 6), "ROW CLARITY INSIDE OVERLAP", 12, MUTED)
    for values, color in ((head_clear, TEAL), (tail_clear, ACCENT)):
        if len(values) < 2:
            continue
        points = [(plot[0] + round(i / (len(values) - 1) * (plot[2] - plot[0])),
                   plot[3] - 4 - round(float(v) * (plot[3] - plot[1] - 26))) for i, v in enumerate(values)]
        draw.line(points, fill=color, width=2)
    return encode_image(canvas, "PNG")


def score_curve_png(curve, chosen):
    width, height = 980, 300
    canvas = Image.new("RGB", (width, height), PAGE)
    draw = ImageDraw.Draw(canvas)
    print_text(draw, (18, 12), "LONGITUDINAL OFFSET  /  NORMALIZED CROSS CORRELATION", 15, TEAL)
    left, top, right, bottom = 60, 46, width - 24, height - 42
    offsets = np.array([row["offset"] for row in curve])
    scores = np.array([row["score"] for row in curve])
    lo, hi = float(offsets.min()), float(offsets.max())
    v_lo, v_hi = float(min(scores.min(), 0.0)), float(max(scores.max(), 1.0))
    px = lambda value: round(left + (value - lo) / max(hi - lo, 1) * (right - left))
    py = lambda value: round(bottom - (value - v_lo) / max(hi - v_lo, 1e-9) * (bottom - top))
    draw.line((left, py(0), right, py(0)), fill="#364856")
    draw.line(list(zip([px(v) for v in offsets], [py(v) for v in scores])), fill=BLUE, width=2)
    for index, alignment in enumerate(chosen):
        x, y = px(alignment.offset), py(alignment.score)
        color = RED if index == 0 else ACCENT
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
        print_text(draw, (x + 7, y - 7), f"#{index + 1} {alignment.offset:+d}", 11, color)
    return encode_image(canvas, "PNG")


def parse_join_request(values, head_height=44000, tail_height=44000):
    values = mapping(values, "接续请求")

    def mark(key, name, height):
        raw = values.get(key)
        if raw is None:
            return None
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise BenchError(f"{name}须为两个行号。")
        first = integer(raw[0], f"{name}起始行", 0, height - 1)
        last = integer(raw[1], f"{name}停止行", first + 1, height)
        if last - first < MIN_OVERLAP_ROWS:
            raise BenchError(f"{name}至少要标出 {MIN_OVERLAP_ROWS} 行共同区域。")
        return first, last

    head_mark = mark("head_rows", "片头重叠段", head_height)
    tail_mark = mark("tail_rows", "片尾重叠段", tail_height)
    mode = option(values.get("mode", "quality"), ("quality", "single"), "接缝方式")
    crossfade = integer(values.get("crossfade", DEFAULT_CROSSFADE), "过渡行数", 0, 40)
    min_segment = integer(values.get("min_segment", 48), "最短来源段", 4, 4000)
    return head_mark, tail_mark, mode, crossfade, min_segment
