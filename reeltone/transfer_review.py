from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
import html

from .errors import Conflict
from .packet import csv_bytes
from .timing import format_side_time
from .visual import ACCENT, BLUE, GRID, MUTED, PAGE, PANEL, RED, TEAL, TEXT, print_text, wrapped_ascii
from .raster import encode_image
from PIL import Image, ImageDraw


STATUS_TEXT = {"draft": "Draft", "hold": "Hold", "reviewed": "Reviewed"}
TONE_COLORS = {"reviewed": TEAL, "hold": RED, "draft": BLUE}
KIND_COLORS = {"excerpt": TEAL, "splice": ACCENT, "damage": RED, "silence": BLUE, "sync": "#aeaeef", "leader": "#ffc694"}


@dataclass(frozen=True)
class Interval:
    start: int
    stop: int

    def as_pair(self):
        return [self.start, self.stop]

    @property
    def duration(self):
        return self.stop - self.start

    def intersect(self, other):
        start, stop = max(self.start, other.start), min(self.stop, other.stop)
        return Interval(start, stop) if start < stop else None

    def covers(self, other):
        return self.start <= other.start and self.stop >= other.stop


def partition(spans, total):
    events = defaultdict(lambda: [set(), set()])
    events[0]
    events[total]
    for identity, span in spans:
        events[span.start][0].add(identity)
        events[span.stop][1].add(identity)
    points = sorted(events)
    active = set()
    sections = []
    for index, start in enumerate(points[:-1]):
        begin, end = events[start]
        active -= end
        active |= begin
        stop = points[index + 1]
        if stop > start:
            sections.append((Interval(start, stop), frozenset(active)))
    return sections


def summary_spans(scanned, duration):
    result = {"unheard": 0, "single": 0, "multiple": 0, "unheard_spans": [], "alternate_spans": []}
    for span, listening in scanned:
        count = len(listening)
        if count == 0:
            result["unheard"] += span.duration
            result["unheard_spans"].append(span.as_pair())
        elif count == 1:
            result["single"] += span.duration
        else:
            result["multiple"] += span.duration
            result["alternate_spans"].append({"rows": span.as_pair(), "takes": sorted(listening)})
    result["reviewed_fraction"] = round((duration - result["unheard"]) / duration, 6)
    result["unheard_spans"] = result["unheard_spans"][:80]
    result["alternate_spans"] = result["alternate_spans"][:80]
    return result


def take_row(take, max_rows):
    footprint = take["payload"]["audio"]["source_rows"]
    if not 0 <= footprint[0] < footprint[1] <= max_rows:
        raise Conflict("监听快照的原片行范围超出了扫描高度。")
    meta = take["payload"]["clock"]
    notices = take["payload"]["audio"].get("warnings", [])
    estimated = meta.get("estimated_spans_in_range", []) if meta.get("mapping") == "local_perforation" else []
    if meta.get("mapping") == "local_perforation":
        notices = [*notices]
        if estimated:
            notices.append(f"该监听有 {len(estimated)} 段齿孔节奏无法确认，相应时间为估算。")
    return {"id": take["id"], "title": take["title"], "status": take["status"], "rows": list(footprint),
            "rate": meta["rows_per_second"], "clock": meta["source"], "clock_mapping": meta.get("mapping", "uniform"),
            "reverse": bool(meta["reverse"]), "estimated_spans": estimated,
            "duration": take["payload"]["audio"]["duration_seconds"], "warnings": list(notices),
            "recipe_id": take["recipe_id"], "wav_sha256": take["wav_sha256"], "recorded_at": take["created"]}


def by_cue(cue, available):
    span = Interval(cue["y0"], cue["y1"])
    statuses = defaultdict(list)
    whole = []
    overlap = []
    for item in available:
        take_span = Interval(*item["rows"])
        if take_span.covers(span):
            statuses[item["status"]].append(item["id"])
            whole.append(item["id"])
        elif hit := span.intersect(take_span):
            overlap.append({"take": item["id"], "status": item["status"], "rows": hit.as_pair(),
                            "portion": round(hit.duration / span.duration, 6)})
    anchored = cue.get("anchor")
    flags = []
    if cue["status"] == "omit":
        state = "omitted"
    elif statuses["reviewed"]:
        state = "can_deliver"
    elif whole:
        state = "unreviewed"
    elif overlap:
        state = "partial"
    else:
        state = "unread"
    if anchored and anchored["take"] not in whole:
        flags.append("标记引用的监听未覆盖完整区间。")
    elif anchored and anchored["take"] not in statuses["reviewed"]:
        flags.append("标记引用的监听还未成为可交付版本。")
    if len(statuses["reviewed"]) > 1:
        flags.append("有多个已复核版本覆盖区间，交付前须明确选择。")
    return {"id": cue["id"], "title": cue["title"], "status": cue["status"], "kind": cue["kind"], "rows": span.as_pair(),
            "outcome": state, "flags": flags, "listed_take": anchored["take"] if anchored else None,
            "reviewed": statuses["reviewed"], "draft": statuses["draft"], "held": statuses["hold"], "partial": overlap[:20]}


def scan_review(scan, takes, cues, proofs):
    total = scan["metadata"]["height"]
    native_takes = [take_row(take, total) for take in takes]
    reviewed = [(row["id"], Interval(*row["rows"])) for row in native_takes if row["status"] == "reviewed"]
    coverage = summary_spans(partition(reviewed, total), total)
    cue_summary = [by_cue(cue, native_takes) for cue in cues]
    verified_ranges = [entry["rows"] for entry in cue_summary if entry["outcome"] == "can_deliver"]
    open_ranges = [entry["rows"] for entry in cue_summary if entry["outcome"] in {"unreviewed", "partial", "unread"}]
    proof_types = defaultdict(int)
    for proof in proofs:
        proof_types[proof["category"]] += 1
    alternate_clock = len({(row["clock_mapping"], row["rate"], row["reverse"])
                           for row in native_takes if row["status"] == "reviewed"}) > 1
    return {"scan": {"id": scan["id"], "title": scan["title"], "sha256": scan["sha256"], "height": total},
            "coverage": coverage, "takes": native_takes, "cues": cue_summary, "proof_types": dict(proof_types),
            "cue_ready_count": len(verified_ranges), "cue_open_count": len(open_ranges),
            "held_takes": sum(row["status"] == "hold" for row in native_takes), "multiple_read_clocks": alternate_clock,
            "estimated_takes": sum(bool(row["estimated_spans"]) for row in native_takes)}


def rollup(reel, scan_rows):
    heard = sum(entry["coverage"]["single"] + entry["coverage"]["multiple"] for entry in scan_rows)
    total_rows = sum(item["scan"]["height"] for item in scan_rows)
    cues_total = sum(len(item["cues"]) for item in scan_rows)
    return {"reel": reel, "number_of_scans": len(scan_rows), "retained_takes": sum(len(item["takes"]) for item in scan_rows),
            "reviewed_rows": heard, "all_native_rows": total_rows, "row_readiness": round(heard / total_rows, 6) if total_rows else 0,
            "ready_cues": sum(item["cue_ready_count"] for item in scan_rows), "pending_cues": sum(item["cue_open_count"] for item in scan_rows),
            "all_cues": cues_total, "runs": scan_rows}


def from_catalog(catalog, reel_id):
    reel = catalog.get_reel(reel_id)
    scans = catalog.list_scans(reel_id)
    by_scan = defaultdict(list)
    for cue in catalog.list_cues(reel_id):
        by_scan[cue["scan_id"]].append(cue)
    report = []
    for scan in scans:
        report.append(scan_review(scan, catalog.list_takes(scan["id"]), by_scan[scan["id"]], catalog.list_proofs(scan["id"])))
    return rollup(reel, report)


def cue_review_csv(report):
    result = [("scan_file", "scan_sha256", "cue_label", "check_status", "start_native_row", "stop_native_row_exclusive",
               "reviewed_candidates", "previously_selected_take", "warning")]
    for run in report["runs"]:
        for cue in run["cues"]:
            result.append((run["scan"]["title"], run["scan"]["sha256"], cue["title"], cue["outcome"], *cue["rows"],
                           " ".join(cue["reviewed"]), cue["listed_take"] or "", " / ".join(cue["flags"])))
    return csv_bytes(result)


def transfer_html(report):
    e = lambda raw: html.escape(str(raw), quote=True)
    heading = e(report["reel"]["title"])
    document = [f'<!doctype html><html lang="zh-CN"><meta charset="UTF-8"><title>{heading} · ReelTone 审听覆盖</title>',
                '<meta name="viewport" content="width=device-width">',
                '<style>body{font:15px system-ui;color:#18222b;max-width:1024px;margin:2rem auto;padding:0 1rem}',
                'h2{padding-top:1rem} table{width:100%;border-collapse:collapse} td,th{text-align:left;padding:.5rem;border-bottom:1px solid #d3dfdf}',
                'th{background:#eef4f5} small{color:#41565b} .warn{color:#7d362a}</style>',
                f'<h1>{heading}&nbsp;审听覆盖</h1><p>{report["number_of_scans"]} 张扫描，{report["retained_takes"]} 份监听。',
                f'已复核行占比 {report["row_readiness"]:.1%}。待审听标记 {report["pending_cues"]} 项。</p>']
    for run in report["runs"]:
        scan = run["scan"]
        coverage = run["coverage"]
        document.append(f'<article><h2>{e(scan["title"])}</h2><small>SHA256 {e(scan["sha256"])}</small>')
        document.append(f'<p>{scan["height"]} 行。未覆盖 {coverage["unheard"]} 行；多份复核覆盖 {coverage["multiple"]} 行。</p>')
        if run["multiple_read_clocks"]:
            document.append('<p class="warn">已复核的监听使用多个时间基准；以上仅比较原片行覆盖。</p>')
        if run["takes"]:
            document.append('<table><thead><tr><th>监听</th><th>状态</th><th>原片行</th><th>秒</th><th>定时</th></tr></thead><tbody>')
            for take in run["takes"]:
                a, b = take["rows"]
                note = "估算" if take.get("estimated_spans") else (
                    "局部齿孔" if take.get("clock_mapping") == "local_perforation" else "统一")
                document.append(f'<tr><td>{e(take["title"])}</td><td>{e(take["status"])}</td><td>{a}–{b}</td>'
                                f'<td>{e(format_side_time(take["duration"]))}</td><td>{e(note)}</td></tr>')
            document.append('</tbody></table>')
        if run["cues"]:
            document.append('<table><thead><tr><th>标记</th><th>原片行</th><th>可用性</th><th>提示</th></tr></thead><tbody>')
            for cue in run["cues"]:
                a, b = cue["rows"]
                document.append(f'<tr><td>{e(cue["title"])}</td><td>{a}–{b}</td><td>{e(cue["outcome"])}</td>'
                                f'<td>{e(" / ".join(cue["flags"]))}</td></tr>')
            document.append('</tbody></table>')
        document.append('</article>')
    document.append('</html>')
    return "\n".join(document).encode("utf-8")


def clip_bars(draw, origin, base, width, scan_height, items, tone_field, palette):
    lo, hi = origin
    slots = []
    for row in sorted(items, key=lambda cue: cue["rows"][0]):
        a, b = row["rows"]
        x0 = round(lo + width * a / scan_height)
        x1 = max(x0 + 3, round(lo + width * b / scan_height))
        index = next((n for n, end in enumerate(slots) if end <= a), len(slots))
        if index >= 5:
            index = 4
        if index == len(slots):
            slots.append(b)
        else:
            slots[index] = max(slots[index], b)
        y = base + index * 10
        color = palette.get(row[tone_field], MUTED)
        draw.rounded_rectangle((x0, y, min(hi, x1), y + 7), radius=3, fill=color)
    return len(slots)


def coverage_image(report):
    runs = report["runs"]
    pages = min(24, len(runs))
    width, row_height = 1050, 142
    canvas = Image.new("RGB", (width, max(175, 76 + pages * row_height)), PAGE)
    draw = ImageDraw.Draw(canvas)
    print_text(draw, (22, 18), "REELTONE  /  TRANSFER COVERAGE", 19, TEAL)
    print_text(draw, (22, 49), f"{report['reviewed_rows']:,} reviewed scan rows / {report['all_native_rows']:,}", 13, MUTED)
    for i, row in enumerate(runs[:pages]):
        start_y = 79 + i * row_height
        title = " ".join(wrapped_ascii(row["scan"]["title"], 51))[:51]
        print_text(draw, (22, start_y + 8), f"{i+1:02}  {title}", 14, TEXT)
        print_text(draw, (22, start_y + 31), f"{row['coverage']['reviewed_fraction']:.1%} reviewed", 11, MUTED)
        print_text(draw, (22, start_y + 59), f"{row['cue_ready_count']} ready cues", 11, MUTED)
        left, right = 335, width - 28
        draw.rounded_rectangle((left, start_y, right, start_y + 116), radius=7, fill=PANEL)
        total = row["scan"]["height"]
        for mark in range(0, 6):
            x = left + (right - left) * mark / 5
            draw.line((x, start_y + 2, x, start_y + 116), fill=GRID)
            print_text(draw, (x, start_y + 120), f"{round(total * mark / 5):,}", 9, MUTED, anchor="mt")
        clip_bars(draw, (left, right), start_y + 5, right - left, total, row["takes"], "status", TONE_COLORS)
        clip_bars(draw, (left, right), start_y + 62, right - left, total, row["cues"], "kind", KIND_COLORS)
    return encode_image(canvas, "PNG")


def export_review(report, format):
    if format == "csv":
        data, media, filename = cue_review_csv(report), "text/csv", "reel-review-coverage.csv"
    elif format == "html":
        data, media, filename = transfer_html(report), "text/html", "reel-review-coverage.html"
    else:
        data, media, filename = coverage_image(report), "image/png", "reel-review-coverage.png"
    return data, media, filename, sha256(data).hexdigest()
