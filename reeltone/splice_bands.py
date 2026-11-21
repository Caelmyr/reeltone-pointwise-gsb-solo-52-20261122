import warnings
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .contracts import integer, real, sequence
from .errors import BenchError, Limit
from .geometry import Box
from .raster import encode_image
from .splices import step_evidence
from .visual import ACCENT, MUTED, PAGE, PANEL, RED, TEAL, PlotSeries, line_plot, place_film, print_text
from PIL import Image, ImageDraw


MIN_BANDS = 2
MAX_BANDS = 4
MAX_BAND_PIXELS = 700_000
MAX_WORK_PIXELS = 1_500_000
MAX_ROWS = 2000
# The slope grid finer than the merge tolerance is what lets one line collect support
# from bands at different columns without splitting into several near-duplicate lines.
SLOPE_STEPS = 65
MAX_FINDINGS = 16
OBSERVE_TOLERANCE = 2.5
COLUMN_COVERAGE = 0.15
SUPPORT_COVERAGE = 0.35
MERGE_ROWS = 3.0
CLASS_ORDER = {"corroborated": 0, "single_band": 1, "picture_only": 2}


@dataclass(frozen=True)
class Band:
    index: int
    box: Box
    role: str
    center: float
    work: Box

    def as_dict(self):
        return {"index": self.index, "box": self.box.as_list(), "role": self.role,
                "center_column": round(self.center, 2), "observation_box": self.work.as_list()}


@dataclass(frozen=True)
class BandEvidence:
    pivots: np.ndarray
    evidence: np.ndarray
    coverage: np.ndarray
    consistency: np.ndarray
    signed_step: np.ndarray


def band_role(box, aperture, perforations):
    def overlap(guide):
        return max(0, min(box.x1, guide.x1) - max(box.x0, guide.x0))
    sound, holes = overlap(aperture), overlap(perforations)
    if sound and holes:
        return "mixed"
    if sound:
        return "soundtrack"
    if holes:
        return "perforation"
    return "picture"


def parse_bands(material, raw, window, max_slant):
    items = sequence(raw, "观察带", minimum=MIN_BANDS, maximum=MAX_BANDS)
    boxes, total = [], 0
    for number, item in enumerate(items, 1):
        box = Box.parse(item, material.width, material.height, minimum=(6, 2 * window + 4),
                        name=f"第 {number} 条观察带")
        if any(box.x0 < old.x1 and old.x0 < box.x1 for old in boxes):
            raise BenchError("观察带要落在胶片不同的横向位置，列范围不能重叠。")
        if box.height > MAX_ROWS:
            raise Limit(f"单条观察带最多 {MAX_ROWS:,} 行。缩短行范围后再试。")
        total += box.area
        if total > MAX_BAND_PIXELS:
            raise Limit(f"观察带合计最多读取 {MAX_BAND_PIXELS:,} 原生像素。缩窄观察带或行范围后再试。")
        boxes.append(box)
    centers = [0.5 * (box.x0 + box.x1) for box in boxes]
    if max(centers) - min(centers) < 8:
        raise BenchError("观察带须分布在胶片不同的横向位置，列范围中心至少相隔 8 列。")
    reference = float(np.average(centers, weights=[box.width for box in boxes]))
    bands = []
    for number, box in enumerate(boxes):
        reach = abs(centers[number] - reference) + box.width / 2
        pad = int(np.ceil(max_slant * reach)) + window + 3
        work = Box(box.x0, max(0, box.y0 - pad), box.x1, min(material.height, box.y1 + pad))
        bands.append(Band(number, box, band_role(box, material.recipe.aperture, material.recipe.perforations),
                          centers[number], work))
    expanded = sum(band.work.area for band in bands)
    if expanded > MAX_WORK_PIXELS:
        raise Limit(f"按最大倾斜展开后观察带合计 {expanded:,} 像素，超过 {MAX_WORK_PIXELS:,}。缩短行范围后再试。")
    shared = all(max(a.box.y0, b.box.y0) < min(a.box.y1, b.box.y1) for a in bands for b in bands)
    if not shared:
        raise BenchError("各观察带的行范围须有共同重叠，才能比较同一处接片。")
    return bands, reference, shared


def sample_band(box, values, slope, reference):
    band = box.subimage(values)
    if slope == 0:
        return band
    rows, columns = box.height, box.width
    across = np.arange(box.x0, box.x1, dtype=np.float64)[None, :]
    shifted = np.arange(rows, dtype=np.float64)[:, None] + slope * (across - reference)
    lower = np.floor(shifted)
    fraction = shifted - lower
    lower = np.clip(lower.astype(np.int64), 0, rows - 1)
    upper = np.clip(lower + 1, 0, rows - 1)
    indexes = np.broadcast_to(np.arange(columns)[None, :], (rows, columns))
    return (1.0 - fraction) * band[lower, indexes] + fraction * band[upper, indexes]


def band_evidence(values, band, slope, reference, window, contrast):
    sheared = sample_band(band.work, values, slope, reference)
    pivots, _, _, _, step = step_evidence(sheared, window)
    magnitude = np.abs(step)
    strong = magnitude >= contrast * 0.5
    coverage = strong.mean(axis=1)
    counts = np.maximum(strong.sum(axis=1), 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        evidence = np.nanmedian(np.where(strong, magnitude, np.nan), axis=1)
        signed = np.nanmedian(np.where(strong, step, np.nan), axis=1)
    rising = np.count_nonzero(strong & (step > 0), axis=1) / counts
    thin = ~np.isfinite(evidence)
    evidence = np.where(thin, np.median(magnitude, axis=1), evidence)
    signed = np.where(thin, np.median(step, axis=1), signed)
    consistency = np.maximum(rising, 1.0 - rising)
    consistency[coverage < COLUMN_COVERAGE] = 0.0
    evidence[coverage < COLUMN_COVERAGE] = 0.0
    # A splice is a step shared across most columns, unlike the band's own recurring
    # structure (waveform edges, perforation rows) which repeats row after row.
    # Subtract each band's own typical response so only distinctly strong rows remain.
    baseline = float(np.median(evidence))
    excess = np.maximum(evidence - baseline, 0.0)
    signed = np.where(excess > 0, signed, 0.0)
    return BandEvidence(band.work.y0 + pivots, excess, coverage, consistency, signed)


def slope_grid(bands, maximum):
    centers = [band.center for band in bands]
    if max(centers) - min(centers) < 8:
        return np.array([0.0]), False
    return np.linspace(-maximum, maximum, SLOPE_STEPS), True


def accumulate(values, bands, slopes, reference, window, contrast):
    bottom = min(band.work.y0 for band in bands)
    top = max(band.work.y1 for band in bands)
    rows = np.arange(bottom, top)
    total = np.zeros((len(slopes), len(rows)), dtype=np.float64)
    responses = {}
    for position, slope in enumerate(slopes):
        for band in bands:
            response = band_evidence(values, band, slope, reference, window, contrast)
            responses[(position, band.index)] = response
            indexes = response.pivots - bottom
            keep = (indexes >= 0) & (indexes < len(rows))
            if np.any(keep):
                np.add.at(total[position], indexes[keep], response.evidence[keep])
    return rows, total, responses


def observe(band, response, slope, reference, row, contrast):
    expected = row + slope * (band.center - reference)
    result = {"index": band.index, "role": band.role, "box": band.box.as_list(), "seen": False,
              "evidence_state": "absent", "column_coverage": 0.0, "sign_consistency": 0.0,
              "evidence": 0.0, "signed_step": 0.0, "row_at_center": round(float(expected), 2),
              "row_offset": None,
              "line_rows": [round(float(row + slope * (band.box.x0 - reference)), 2),
                            round(float(row + slope * (band.box.x1 - reference)), 2)]}
    if not (band.work.y0 <= row < band.work.y1) or not (band.box.y0 <= expected < band.box.y1):
        result["evidence_state"] = "outside"
        return result
    nearest = int(np.argmin(np.abs(response.pivots - row)))
    if abs(float(response.pivots[nearest]) - row) > OBSERVE_TOLERANCE:
        return result
    coverage = float(response.coverage[nearest])
    evidence = float(response.evidence[nearest])
    consistency = float(response.consistency[nearest])
    result.update(evidence=round(evidence, 5), column_coverage=round(coverage, 4),
                  sign_consistency=round(consistency, 4),
                  signed_step=round(float(response.signed_step[nearest]), 5),
                  row_offset=round(float(response.pivots[nearest]) - row, 2))
    supported = coverage >= SUPPORT_COVERAGE and consistency >= 0.6 and evidence >= contrast * 0.8
    result["seen"] = bool(supported)
    result["evidence_state"] = "supported" if supported else ("partial" if coverage >= COLUMN_COVERAGE else "absent")
    return result


def classify(observations):
    seen = [item for item in observations if item["seen"]]
    roles = {item["role"] for item in seen}
    if len(seen) >= 2 and any(item["role"] != "picture" for item in seen):
        return "corroborated", roles
    if len(seen) >= 2:
        return "picture_only", roles
    if len(seen) == 1 and next(iter(seen))["role"] != "picture":
        return "single_band", roles
    return "picture_only", roles


def step_agreement(observations):
    """Sign of the brightness step must agree where two or more bands really saw the line."""
    steps = [item["signed_step"] for item in observations if item["seen"]]
    if len(steps) < 2:
        return True
    signs = {value > 0 for value in steps}
    return len(signs) == 1


def same_finding(support, accepted, tolerance):
    for other in accepted:
        shared = set(support) & set(other)
        if shared and all(abs(support[index]["row_at_center"] - other[index]["row_at_center"]) <= tolerance
                          for index in shared):
            return True
    return False


def describe(reference, row, slope, classification, roles, bands, observations, score):
    seen = [item for item in observations if item["seen"]]
    focus = next((item for item in seen if item["role"] != "picture"), seen[0])
    focus_band = next(band for band in bands if band.index == focus["index"])
    columns = sorted({value for item in seen for value in (item["box"][0], item["box"][2])})
    return {
        "reference_row": round(row, 2), "reference_column": round(reference, 2),
        "slope_rows_per_column": round(slope, 5), "classification": classification,
        "support_count": len(seen), "band_count": len(bands), "roles": sorted(roles),
        "line_columns": [round(value, 1) for value in (min(columns), max(columns))],
        "line_rows": [round(float(row + slope * (value - reference)), 1)
                      for value in (min(columns), max(columns))],
        "focus_row": int(round(row + slope * (focus_band.center - reference))),
        "focus_column": int(round(focus_band.center)),
        "score": round(score, 5), "observations": observations}


def same_line(hypothesis, other, reference, tolerance, bands):
    return all(abs((hypothesis["row"] + hypothesis["slope"] * (band.center - reference))
                   - (other["row"] + other["slope"] * (band.center - reference))) <= tolerance
               for band in bands)


def line_findings(rows, total, responses, slopes, bands, reference, window, contrast):
    """Rank aggregated peaks and keep the first line of each physically distinct splice.

    Peaks are ranked by summed band evidence, but aggregation alone cannot separate a real
    splice from the recurring structure of a busy band, so the ranking floor is low and the
    classification of every peak is what actually decides retention. A peak that no band
    accepts as a step is skipped; one that duplicates an accepted line is merged; anything
    else - including a frame edge only present in the picture band - is kept and labelled.
    """
    smoothed = ndimage.gaussian_filter1d(total, 0.6, axis=1, mode="nearest")
    tolerance = MERGE_ROWS * window
    peaked = smoothed == ndimage.maximum_filter(smoothed, size=(3, 3), mode="nearest")
    candidates = np.argwhere(peaked & (smoothed >= contrast * 0.45))
    findings, accepted = [], []
    if len(candidates):
        ranked = candidates[np.argsort(smoothed[candidates[:, 0], candidates[:, 1]])[::-1]]
        for position, at in ranked[:MAX_FINDINGS * 12]:
            position, at = int(position), int(at)
            slope, row = float(slopes[position]), float(rows[at])
            if any(same_line({"row": row, "slope": slope}, other, reference, tolerance, bands)
                   for other in accepted):
                continue
            observations = [observe(band, responses[(position, band.index)], slope, reference, row, contrast)
                            for band in bands]
            support = {item["index"]: item for item in observations if item["seen"]}
            if not support or same_finding(support, [other["support"] for other in accepted], tolerance):
                continue
            if not step_agreement(observations):
                # Bands disagree on bright-to-dark versus dark-to-bright: these are
                # different structures (waveform edge against a perforation, say), not one line.
                continue
            classification, roles = classify(observations)
            accepted.append({"row": row, "slope": slope, "support": support})
            findings.append(describe(reference, row, slope, classification, roles, bands, observations,
                                     float(smoothed[position, at])))
            if len(findings) >= MAX_FINDINGS:
                break
    picture_only = sum(item["classification"] == "picture_only" for item in findings)
    findings.sort(key=lambda item: (CLASS_ORDER[item["classification"]], -item["support_count"], -item["score"]))
    return findings, picture_only, len(findings) >= MAX_FINDINGS


def inspect_splice_bands(material, values):
    window = integer(values.get("window", 8), "边界一侧样本行数", 3, 30)
    contrast = real(values.get("contrast", 0.08), "接缝候选最小明暗差", 0.015, 0.9)
    max_slant = real(values.get("max_slant", 0.35), "最大倾斜（行／列）", 0.02, 1.5)
    bands, reference, shared = parse_bands(material, values.get("bands"), window, max_slant)
    slopes, resolved = slope_grid(bands, max_slant)
    rows, total, responses = accumulate(material.pixels, bands, slopes, reference, window, contrast)
    findings, picture_only, truncated = line_findings(rows, total, responses, slopes, bands, reference,
                                                      window, contrast)
    summary = {
        "parent": material.identity(),
        "box": [min(band.box.x0 for band in bands), min(band.box.y0 for band in bands),
                max(band.box.x1 for band in bands), max(band.box.y1 for band in bands)],
        "bands": [band.as_dict() for band in bands], "band_count": len(bands),
        "reference_column": round(reference, 2), "window_each_side": window, "contrast": contrast,
        "max_slant": max_slant, "slant_resolved": resolved, "bands_share_rows": shared,
        "candidate_count": len(findings),
        "corroborated_count": sum(item["classification"] == "corroborated" for item in findings),
        "limited_count": sum(item["classification"] != "corroborated" for item in findings),
        "picture_only_count": picture_only, "truncated": truncated, "findings": findings,
        "caveat": "各观察带沿同一条斜线对齐后比较横向共识的明暗跳变。至少两条带支持且其中一条不是画格区的接片列为证据较完整；只由画格边或单条带支持的条目仍保留，但列为有限证据，须回原片核对。"}
    return summary, splice_bands_plate(material, findings, bands, summary), splice_bands_chart(findings, bands)


def splice_bands_chart(findings, bands):
    marks = []
    for number, item in enumerate(findings):
        marks.append({"index": number + 1, "seen": item["support_count"], "classification": item["classification"]})
    total = [{"index": number + 1, "seen": len(bands)} for number in range(len(findings))]
    series = [line for line in (
        PlotSeries(MUTED, total, lambda item: item["seen"], lambda item: item["index"], "Observation bands used"),
        PlotSeries(ACCENT, [row for row in marks if row["classification"] == "corroborated"],
                   lambda item: item["seen"], lambda item: item["index"], "Corroborated line"),
        PlotSeries(RED, [row for row in marks if row["classification"] != "corroborated"],
                   lambda item: item["seen"], lambda item: item["index"], "Limited evidence")) if line.records]
    span = max(4, len(bands))
    return line_plot("BANDS THAT SAW EACH SPLICE LINE", series,
                     x_range=(0.5, max(2.0, len(findings) + 0.5)), y_range=(0, span),
                     formatter=lambda x: f"line {x:.0f}", y_formatter=lambda y: f"{y:.0f}")


def splice_bands_plate(material, findings, bands, summary):
    width = 990
    map_area = Box(18, 60, 520, 386)
    side_area = Box(534, 60, 972, 386)
    card_height = 160 if findings else 0
    canvas = Image.new("RGB", (width, 400 + card_height), PAGE)
    draw = ImageDraw.Draw(canvas)
    print_text(draw, (20, 16), "REELTONE  /  MULTI-BAND SPLICE LINES", 16, TEAL)
    print_text(draw, (20, 38), f"{len(bands)} observation bands · {len(findings)} retained lines · "
                               f"reference column {summary['reference_column']:.0f}", 11, MUTED)
    region = Box(*summary["box"])
    if findings:
        window = place_film(canvas, material.pixels, region, material.recipe.orientation, map_area, stretch=True)
        for band in bands:
            left, top, right, bottom = window.rect(band.box)
            draw.rectangle((left, top, right - 1, bottom - 1), outline=MUTED, width=1)
            print_text(draw, (left + 2, top + 1), str(band.index + 1), 10, MUTED)
        for rank, finding in enumerate(findings[:5]):
            color = ACCENT if finding["classification"] == "corroborated" else RED
            x_a, y_a = window.point(finding["line_columns"][0], finding["line_rows"][0])
            x_b, y_b = window.point(finding["line_columns"][1], finding["line_rows"][1])
            draw.line((x_a, y_a, x_b, y_b), fill=color, width=2)
            print_text(draw, (min(x_a, x_b) + 3, min(y_a, y_b) - 12), f"L{rank + 1}", 10, color)
        print_text(draw, (map_area.x0, map_area.y1 + 4),
                   "Bands outlined; lines drawn across the columns that actually saw them.", 10, MUTED)
    else:
        draw.rectangle(tuple(map_area.as_list()), fill=PANEL)
        print_text(draw, (map_area.x0 + 16, map_area.y0 + 18), "No line reached the retained support rules.",
                   12, MUTED)
    draw.rectangle(tuple(side_area.as_list()), fill=PANEL)
    print_text(draw, (side_area.x0 + 14, side_area.y0 + 12), "LINE SUMMARY", 12, TEAL)
    labels = {"corroborated": "Corroborated", "single_band": "Single band", "picture_only": "Picture edge only"}
    y = side_area.y0 + 36
    if not findings:
        print_text(draw, (side_area.x0 + 14, y), "Try wider rows around the splice.", 11, MUTED)
    for rank, finding in enumerate(findings[:6]):
        if y > side_area.y1 - 32:
            break
        color = ACCENT if finding["classification"] == "corroborated" else RED
        print_text(draw, (side_area.x0 + 14, y), f"L{rank + 1}  {labels[finding['classification']]}", 11, color)
        print_text(draw, (side_area.x0 + 14, y + 13),
                   f"{finding['support_count']}/{finding['band_count']} bands · "
                   f"rows {finding['line_rows'][0]:.0f} to {finding['line_rows'][1]:.0f} · "
                   f"slope {finding['slope_rows_per_column']:+.3f}", 10, MUTED)
        y += 32
    if findings:
        top = findings[0]
        card_width = (width - 26 - (len(bands) - 1) * 8) // len(bands)
        for position, band in enumerate(bands):
            observation = next(item for item in top["observations"] if item["index"] == band.index)
            left = 16 + position * (card_width + 8)
            area = Box(left, 400, left + card_width, 400 + card_height - 8)
            draw.rounded_rectangle(tuple(area.as_list()), radius=7, fill=PANEL)
            color = ACCENT if observation["seen"] else RED
            print_text(draw, (left + 10, 406), f"Band {band.index + 1} · {band.role}", 11, color)
            print_text(draw, (left + 10, 420),
                       f"{observation['evidence_state']} · {observation['column_coverage']:.0%} columns",
                       10, MUTED)
            rows = observation["line_rows"]
            expected = 0.5 * (rows[0] + rows[1])
            half = max(15, abs(top["slope_rows_per_column"]) * band.box.width * 0.5 + 13)
            detail = Box(band.box.x0, max(0, int(round(expected - half))), band.box.x1,
                         min(material.height, int(round(expected + half)) + 1))
            inner = Box(left + 8, 436, left + card_width - 8, area.y1 - 7)
            window = place_film(canvas, material.pixels, detail, material.recipe.orientation, inner, stretch=True)
            x_a, y_a = window.point(detail.x0, rows[0])
            x_b, y_b = window.point(detail.x1 - 1, rows[1])
            draw.line((x_a, y_a, x_b, y_b), fill=color, width=2)
    return encode_image(canvas, "PNG")
