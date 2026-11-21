from dataclasses import dataclass

import numpy as np
from scipy import ndimage, stats

from .exposure import measure_exposure
from .geometry import Box
from .raster import aggregated_rows


MEASURED, ESTIMATED, WITHHELD = 0, 1, 2
STATUS_NAMES = {MEASURED: "measured", ESTIMATED: "estimated", WITHHELD: "withheld"}
REFERENCE_MIN_PIXELS = 4
REFERENCE_MIN_CONTRAST = 0.03
REFERENCE_MAX_SPREAD = 0.14
REFERENCE_MAX_CLIPPED = 0.22
REFERENCE_MAX_SHIFT = 0.055
REFERENCE_MAX_LOW_SHIFT = 0.09
REFERENCE_MAX_GAP = 0.24
REFERENCE_EDGE_LOW = 0.006
REFERENCE_EDGE_HIGH = 0.994


@dataclass(frozen=True)
class Pair:
    left: Box
    right: Box
    left_source: str
    right_source: str
    left_recipe: str | None = None
    right_recipe: str | None = None

    def as_dict(self):
        return {"left": self.left.as_list(), "right": self.right.as_list(), "left_source": self.left_source,
                "right_source": self.right_source, "left_recipe": self.left_recipe,
                "right_recipe": self.right_recipe}


@dataclass(frozen=True)
class Comparison:
    selection: Pair
    left: np.ndarray
    right: np.ndarray
    delta: np.ndarray
    left_hist: dict
    right_hist: dict
    agreement: dict
    illumination: dict | None = None
    corrected_left: np.ndarray | None = None
    corrected_right: np.ndarray | None = None
    corrected_delta: np.ndarray | None = None
    band_status: np.ndarray | None = None

    def summary(self):
        output = {"selection": self.selection.as_dict(), "left_statistics": self.left_hist,
                  "right_statistics": self.right_hist, "agreement": self.agreement,
                  "display_grid": [self.delta.shape[1], self.delta.shape[0]],
                  "interpretation": "两张图由操作者分别框选作为对照，均按各自框的相对位置读数。差值不作为像素级配准依据。"}
        if self.illumination is not None:
            output["illumination"] = self.illumination
        return output


def _frame_grid(values, size):
    height, width = size
    rows, _ = aggregated_rows(values, size=height)
    if len(rows) != height or values.shape[1] != width:
        standardized = ndimage.zoom(rows, (height / len(rows), width / rows.shape[1]), order=1, grid_mode=True,
                                    mode="grid-mirror", prefilter=False)
        return standardized[:height, :width]
    return rows


def agreement_measures(a, b):
    delta = b - a
    v0, v1 = a.ravel(), b.ravel()
    pearson = None
    spearman = None
    if np.std(v0) > 1e-7 and np.std(v1) > 1e-7:
        sample = max(1, len(v0) // 16000)
        pearson = float(np.corrcoef(v0, v1)[0, 1])
        spearman = float(stats.spearmanr(v0[::sample], v1[::sample]).statistic)
    return {"mean_delta": round(float(delta.mean()), 6), "mean_absolute_delta": round(float(np.abs(delta).mean()), 6),
            "q95_absolute_delta": round(float(np.percentile(np.abs(delta), 95)), 6),
            "right_brighter_cells": int(np.sum(delta > 0.025)), "right_darker_cells": int(np.sum(delta < -0.025)),
            "linear_r": round(pearson, 5) if pearson is not None else None,
            "rank_r": round(spearman, 5) if spearman is not None else None}


def compare_exposure(left_image, right_image, selection: Pair):
    a, b = selection.left.subimage(left_image), selection.right.subimage(right_image)
    grid_size = (min(a.shape[0], b.shape[0], 420), min(a.shape[1], b.shape[1], 240))
    left, right = _frame_grid(a, grid_size), _frame_grid(b, grid_size)
    return Comparison(selection, left, right, right - left, measure_exposure(a).as_dict(),
                      measure_exposure(b).as_dict(), agreement_measures(left, right))


def _band_edges(start, stop, count):
    edges = np.rint(np.linspace(start, stop, count + 1)).astype(int)
    return np.maximum.accumulate(edges)


def _strip_bands(values, reference, selection, count):
    """Measure one reference patch inside each grid row band of the comparison selection."""
    edges = _band_edges(selection.y0, selection.y1, count)
    median = np.full(count, np.nan)
    low = np.full(count, np.nan)
    spread = np.full(count, np.nan)
    pixels = np.zeros(count, dtype=int)
    clipped = np.full(count, np.nan)
    for index in range(count):
        top = max(edges[index], reference.y0)
        bottom = min(max(edges[index + 1], edges[index] + 1), reference.y1)
        if bottom <= top:
            continue
        patch = values[top:bottom, reference.x0:reference.x1]
        if patch.size == 0:
            continue
        q10, q50, q90 = np.percentile(patch, [10, 50, 90])
        median[index] = q50
        low[index] = q10
        spread[index] = q90 - q10
        pixels[index] = patch.size
        clipped[index] = float(np.mean((patch <= 0.008) | (patch >= 0.992)))
    return {"median": median, "low": low, "spread": spread, "pixels": pixels, "clipped": clipped}


def _strip_quality(profile):
    median, low = profile["median"], profile["low"]
    count = len(median)
    reasons = np.zeros(count, dtype=np.int8)
    reasons[profile["pixels"] < REFERENCE_MIN_PIXELS] = 1
    usable = reasons == 0
    saturated = usable & ((median <= REFERENCE_EDGE_LOW) | (median >= REFERENCE_EDGE_HIGH) |
                          (profile["clipped"] > REFERENCE_MAX_CLIPPED))
    reasons[saturated] = 2
    uneven = (reasons == 0) & (profile["spread"] > REFERENCE_MAX_SPREAD)
    reasons[uneven] = 3
    valid = reasons == 0
    half = max(2, min(12, count // 24))
    for index in np.flatnonzero(valid):
        near = np.arange(max(0, index - half), min(count, index + half + 1))
        near = near[valid[near]]
        if len(near) < max(2, half // 2):
            continue
        reference_median = np.median(median[near])
        reference_low = np.median(low[near])
        if (abs(median[index] - reference_median) > REFERENCE_MAX_SHIFT or
                abs(low[index] - reference_low) > REFERENCE_MAX_LOW_SHIFT):
            reasons[index] = 4
    profile["reasons"] = reasons
    profile["valid"] = reasons == 0
    return profile


def _fill_gaps(profile, count):
    valid = profile["valid"]
    if not np.any(valid):
        return np.full(count, np.nan), np.zeros(count, dtype=bool), np.zeros(count, dtype=bool)
    anchors = np.flatnonzero(valid)
    positions = np.arange(count)
    values = np.interp(positions, anchors, profile["median"][anchors])
    distance = np.min(np.abs(positions[:, None] - anchors[None, :]), axis=1)
    gap = max(2, int(round(count * REFERENCE_MAX_GAP)))
    return values, valid, distance <= gap


def _safe(value, digits=6):
    return round(float(value), digits) if value is not None and np.isfinite(value) else None


def _finite_mean(values):
    values = values[np.isfinite(values)]
    return _safe(values.mean()) if values.size else None


def _reference_summary(values, dark_box, light_box):
    dark = dark_box.subimage(values)
    light = light_box.subimage(values)
    dark_median = float(np.median(dark))
    light_median = float(np.median(light))
    return {"dark": dark_box.as_list(), "light": light_box.as_list(),
            "dark_median": _safe(dark_median), "light_median": _safe(light_median),
            "contrast": _safe(light_median - dark_median),
            "dark_clipped_fraction": _safe(np.mean((dark <= 0.008) | (dark >= 0.992)), 7),
            "light_clipped_fraction": _safe(np.mean((light <= 0.008) | (light >= 0.992)), 7)}


def _quality_counts(profile):
    reasons = profile["reasons"]
    return {"usable": int(np.count_nonzero(reasons == 0)), "missing": int(np.count_nonzero(reasons == 1)),
            "saturated": int(np.count_nonzero(reasons == 2)), "uneven": int(np.count_nonzero(reasons == 3)),
            "scratch_or_obstruction": int(np.count_nonzero(reasons == 4))}


def _masked_agreement(left, right, rows):
    if not np.any(rows):
        return None
    a, b = left[rows], right[rows]
    measures = agreement_measures(a, b)
    measures["bands"] = int(np.count_nonzero(rows))
    measures["cells"] = int(a.size)
    return measures


def _profile_records(profile, status, selection, count, limit=96):
    edges = _band_edges(selection.left.y0, selection.left.y1, count)
    second_edges = _band_edges(selection.right.y0, selection.right.y1, count)
    marks = np.linspace(0, count, min(count, limit) + 1, dtype=int)
    records = []
    for a, b in zip(marks[:-1], marks[1:]):
        state = int(np.max(status[a:b]))
        record = {"fraction": [_safe(a / count, 5), _safe(b / count, 5)],
                  "first_rows": [int(edges[a]), int(edges[b])], "second_rows": [int(second_edges[a]), int(second_edges[b])],
                  "status": STATUS_NAMES[state]}
        for key in ("first_dark", "first_light", "second_dark", "second_light"):
            record[key] = _finite_mean(profile[key][a:b])
        record["first_contrast"] = (_safe(record["first_light"] - record["first_dark"])
                                    if record["first_light"] is not None and record["first_dark"] is not None else None)
        record["second_contrast"] = (_safe(record["second_light"] - record["second_dark"])
                                     if record["second_light"] is not None and record["second_dark"] is not None else None)
        records.append(record)
    return records


def _segments(profile, status, selection, count):
    edges = _band_edges(selection.left.y0, selection.left.y1, count)
    second_edges = _band_edges(selection.right.y0, selection.right.y1, count)
    changes = np.flatnonzero(np.diff(status)) + 1
    boundaries = [0, *changes.tolist(), count]
    output = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        if b <= a:
            continue
        record = {"status": STATUS_NAMES[int(status[a])], "grid_rows": [int(a), int(b)],
                  "first_rows": [int(edges[a]), int(edges[b])],
                  "second_rows": [int(second_edges[a]), int(second_edges[b])]}
        for key in ("first_dark", "first_light", "second_dark", "second_light"):
            record[key] = _finite_mean(profile[key][a:b])
        output.append(record)
    return output


def _warnings(status, references):
    warnings = []
    total = len(status)
    withheld = int(np.count_nonzero(status == WITHHELD))
    estimated = int(np.count_nonzero(status == ESTIMATED))
    if withheld == total:
        warnings.append("所选行的片边参照都不足以校正；校正差值留空，未把整段当作可靠读数。")
    else:
        if withheld:
            warnings.append(f"约 {withheld / total:.0%} 的显示行带没有足够参照，校正差值留空并排除在校正统计外。")
        if estimated:
            warnings.append(f"约 {estimated / total:.0%} 的显示行带由相邻可用参照估算，相关校正读数为估计值。")
    labels = (("first", "dark", "第一张暗参照"), ("first", "light", "第一张亮参照"),
              ("second", "dark", "第二张暗参照"), ("second", "light", "第二张亮参照"))
    for reference, side, label in labels:
        counts = references[reference]["quality"][side]
        for reason, caption in (("saturated", "接近码值边界或饱和"), ("uneven", "差异过大，可能跨到片边杂物"),
                                ("scratch_or_obstruction", "有局部跳变，疑似划痕、遮挡或污点")):
            if counts[reason]:
                warnings.append(f"{label}在 {counts[reason]} 个行带{caption}；这些行带未直接用于校正。")
        if counts["missing"]:
            warnings.append(f"{label}在 {counts['missing']} 个行带没有覆盖到的像素；这些位置只能由相邻行带估算或留空。")
    for reference, label in (("first", "第一张"), ("second", "第二张")):
        contrast = references[reference]["contrast"]
        if contrast is not None and contrast < REFERENCE_MIN_CONTRAST:
            warnings.append(f"{label}的亮参照没有明显亮于暗参照（差 {contrast:.3f}），无法用来校平照明；请检查是否圈反或照到饱和区。")
    return warnings


def analyze_illumination(left, right, selection, illumination, left_image, right_image):
    first_dark, first_light = illumination["first"]
    second_dark, second_light = illumination["second"]
    count = left.shape[0]
    profiles = {
        "first_dark": _strip_quality(_strip_bands(left_image, first_dark, selection.left, count)),
        "first_light": _strip_quality(_strip_bands(left_image, first_light, selection.left, count)),
        "second_dark": _strip_quality(_strip_bands(right_image, second_dark, selection.right, count)),
        "second_light": _strip_quality(_strip_bands(right_image, second_light, selection.right, count)),
    }
    model = {}
    for key, profile in profiles.items():
        values, direct, reachable = _fill_gaps(profile, count)
        model[key] = values
        model[f"{key}_direct"] = direct
        model[f"{key}_reachable"] = reachable
    reachable = (model["first_dark_reachable"] & model["first_light_reachable"] &
                 model["second_dark_reachable"] & model["second_light_reachable"])
    first_contrast = model["first_light"] - model["first_dark"]
    second_contrast = model["second_light"] - model["second_dark"]
    contrasted = reachable & (first_contrast >= REFERENCE_MIN_CONTRAST) & (second_contrast >= REFERENCE_MIN_CONTRAST)
    direct = (model["first_dark_direct"] & model["first_light_direct"] &
              model["second_dark_direct"] & model["second_light_direct"] & contrasted)
    status = np.where(direct, MEASURED, np.where(contrasted, ESTIMATED, WITHHELD)).astype(np.int8)
    admitted = status != WITHHELD
    corrected_left = np.full(left.shape, np.nan)
    corrected_right = np.full(right.shape, np.nan)
    with np.errstate(invalid="ignore"):
        corrected_left[admitted] = ((left[admitted] - model["first_dark"][admitted, None]) /
                                    first_contrast[admitted, None])
        corrected_right[admitted] = ((right[admitted] - model["second_dark"][admitted, None]) /
                                     second_contrast[admitted, None])
    corrected_delta = corrected_right - corrected_left
    profile = {key: model[key] for key in ("first_dark", "first_light", "second_dark", "second_light")}
    corrected = {"admitted": _masked_agreement(corrected_left, corrected_right, admitted),
                 "measured_only": _masked_agreement(corrected_left, corrected_right, status == MEASURED),
                 "withheld_bands": int(np.count_nonzero(status == WITHHELD)),
                 "estimated_bands": int(np.count_nonzero(status == ESTIMATED)),
                 "measured_bands": int(np.count_nonzero(status == MEASURED))}
    references = {}
    for label, (dark_box, light_box), values in (
            ("first", (first_dark, first_light), left_image), ("second", (second_dark, second_light), right_image)):
        summary = _reference_summary(values, dark_box, light_box)
        summary["scan_id"] = selection.left_source if label == "first" else selection.right_source
        summary["recipe_id"] = selection.left_recipe if label == "first" else selection.right_recipe
        summary["quality"] = {
            "dark": _quality_counts(profiles[f"{label}_dark"]),
            "light": _quality_counts(profiles[f"{label}_light"]),
        }
        references[label] = summary
    segment_list = _segments(model, status, selection, count)
    report = {
        "mode": "edge_references",
        "method": "每张扫描各自按显示行带读取片边暗、亮参照，用 (码值 − 暗) / (亮 − 暗) 校平可解释的照明变化后，再比较两张图。",
        "references": references,
        "coverage": {"bands": int(count), "measured": int(np.count_nonzero(status == MEASURED)),
                     "estimated": int(np.count_nonzero(status == ESTIMATED)),
                     "withheld": int(np.count_nonzero(status == WITHHELD)),
                     "measured_fraction": _safe(np.mean(status == MEASURED), 5),
                     "estimated_fraction": _safe(np.mean(status == ESTIMATED), 5),
                     "withheld_fraction": _safe(np.mean(status == WITHHELD), 5),
                     "thresholds": {"minimum_contrast": REFERENCE_MIN_CONTRAST,
                                    "maximum_gap_fraction": REFERENCE_MAX_GAP,
                                    "maximum_patch_spread": REFERENCE_MAX_SPREAD}},
        "corrected": corrected,
        "segments": segment_list,
        "band_states": [int(state) for state in status],
        "profile": _profile_records(model, status, selection, count),
        "warnings": _warnings(status, references),
        "interpretation": "原始码值和未校正差值仍保留在左、右统计与逐行剖面中。校正只去除片边参照能解释的亮度与反差变化；片内横向照明不均、参照条自身损伤、两张框未精确对准的内容，仍会留在校正差值里。"
    }
    return {"report": report, "left": corrected_left, "right": corrected_right, "delta": corrected_delta,
            "status": status}


def paired_line_profile(comparison):
    length = comparison.delta.shape[0]
    marks = np.linspace(0, length, min(length, 180) + 1, dtype=int)
    selection = comparison.selection
    result = []
    for a, b in zip(marks[:-1], marks[1:]):
        frac0 = float(a / length)
        frac1 = float(b / length)
        record = {
            "fraction": [round(frac0, 5), round(frac1, 5)],
            "left_rows": [round(selection.left.y0 + f * selection.left.height, 2) for f in (frac0, frac1)],
            "right_rows": [round(selection.right.y0 + f * selection.right.height, 2) for f in (frac0, frac1)],
            "left": round(float(comparison.left[a:b].mean()), 6), "right": round(float(comparison.right[a:b].mean()), 6),
            "delta": round(float(comparison.delta[a:b].mean()), 6),
        }
        if comparison.band_status is not None:
            state = int(np.max(comparison.band_status[a:b]))
            record["status"] = STATUS_NAMES[state]
            if state != WITHHELD:
                record["corrected_left"] = _finite_mean(comparison.corrected_left[a:b])
                record["corrected_right"] = _finite_mean(comparison.corrected_right[a:b])
                record["corrected_delta"] = _finite_mean(comparison.corrected_delta[a:b])
            else:
                record["corrected_left"] = record["corrected_right"] = record["corrected_delta"] = None
        result.append(record)
    return result
