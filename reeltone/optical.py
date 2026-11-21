from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .geometry import Box, Guide, runs
from .recipe import Tone


@dataclass(frozen=True)
class OpticalRead:
    source_rows: tuple
    width: int
    exposure_midpoint: float
    openness: np.ndarray
    mean_light: np.ndarray
    edge_low: np.ndarray
    edge_high: np.ndarray
    islands: np.ndarray
    clipped_low: np.ndarray
    clipped_high: np.ndarray
    touch_edge: np.ndarray

    def warnings(self):
        notices = []
        if np.mean(self.touch_edge) > 0.05:
            notices.append("部分图案触到读窗左右边界，宽度可能被截断。")
        if np.mean(self.islands > 1) > 0.08:
            notices.append("部分扫描行有多个独立透光区，需排查双轨或灰尘。")
        if np.percentile(self.openness, 96) - np.percentile(self.openness, 4) < 0.012:
            notices.append("光学宽度变化很小，请确认轨道、极性与阈值。")
        return notices

    def summary(self):
        openness = np.percentile(self.openness, [5, 50, 95])
        row_count = len(self.openness)
        return {"source_rows": list(self.source_rows), "aperture_width": self.width,
                "threshold_absolute": round(self.exposure_midpoint, 6),
                "opening_q05_q50_q95": [round(float(value), 5) for value in openness],
                "mean_light": round(float(np.mean(self.mean_light)), 5), "multiple_island_rows": int(np.sum(self.islands > 1)),
                "touch_window_rows": int(np.sum(self.touch_edge)), "fully_dark_rows": int(np.sum(self.openness <= 0.001)),
                "zero_code_pixels": int(self.clipped_low.sum()), "full_code_pixels": int(self.clipped_high.sum()),
                "row_count": row_count, "warnings": self.warnings()}


def tonal_plane(values, tone):
    relative = np.clip((values - tone.black) / (tone.white - tone.black), 0.0, 1.0)
    return 1.0 - relative if tone.polarity == "dark" else relative


def occupied_fractions(values, above, level):
    left = values[:, :-1]
    right = values[:, 1:]
    rising = ~above[:, :-1] & above[:, 1:]
    falling = above[:, :-1] & ~above[:, 1:]
    crossing = rising | falling
    fraction = np.zeros(left.shape, dtype=np.float32)
    np.divide(level - left, right - left, where=crossing, out=fraction)
    span = (above[:, :-1] & above[:, 1:]).astype(np.float32)
    span[rising] = 1 - fraction[rising]
    span[falling] = fraction[falling]
    edge_halves = 0.5 * (above[:, 0].astype(np.float32) + above[:, -1])
    return (span.sum(axis=1, dtype=np.float64) + edge_halves) / values.shape[1]


def outside_edges(values, above, x0, threshold):
    detected = np.any(above, axis=1)
    first = above.argmax(axis=1)
    last = above.shape[1] - 1 - np.flip(above, axis=1).argmax(axis=1)
    lo = np.full(above.shape[0], np.nan)
    hi = np.full(above.shape[0], np.nan)
    rows = np.flatnonzero(detected & (first > 0))
    positions = first[rows]
    left = values[rows, positions - 1]
    right = values[rows, positions]
    lo[rows] = x0 + positions - 0.5 + (threshold - left) / (right - left)
    rows = np.flatnonzero(detected & (last < above.shape[1] - 1))
    positions = last[rows]
    left = values[rows, positions]
    right = values[rows, positions + 1]
    hi[rows] = x0 + positions + 0.5 + (threshold - left) / (right - left)
    lo[detected & (first == 0)] = x0
    hi[detected & (last == above.shape[1] - 1)] = x0 + above.shape[1]
    return lo, hi


def read_area(image, aperture: Guide, tone: Tone, source_rows):
    a, b = source_rows
    values = image[a:b, aperture.x0:aperture.x1]
    plane = tonal_plane(values, tone)
    above = plane >= tone.threshold
    openness = occupied_fractions(plane, above, tone.threshold)
    left, right = outside_edges(plane, above, aperture.x0, tone.threshold)
    islands = np.sum(~above[:, :-1] & above[:, 1:], axis=1) + above[:, 0]
    midpoint = tone.black + (tone.white - tone.black) * (tone.threshold if tone.polarity == "clear" else 1 - tone.threshold)
    return OpticalRead(source_rows, aperture.width, midpoint, openness, values.mean(axis=1), left, right,
                       islands, (values <= 0).sum(axis=1), (values >= 1).sum(axis=1), above[:, 0] | above[:, -1])


def _finite_round(value, digits=3):
    return round(float(value), digits) if np.isfinite(value) else None


def scan_trace(read, size=400):
    top, bottom = read.source_rows
    boundaries = np.linspace(0, bottom - top, min(size, bottom - top) + 1, dtype=int)
    bins = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        left = read.edge_low[a:b]
        right = read.edge_high[a:b]
        decoded = read.openness[a:b]
        left = left[np.isfinite(left)]
        right = right[np.isfinite(right)]
        bins.append({"rows": [int(top + a), int(top + b)], "min": round(float(decoded.min()), 5),
                     "max": round(float(decoded.max()), 5), "mean": round(float(decoded.mean()), 5),
                     "left_median": _finite_round(np.median(left)) if left.size else None,
                     "right_median": _finite_round(np.median(right)) if right.size else None,
                     "border_rows": int(read.touch_edge[a:b].sum())})
    return bins


def track_column_search(image, rows, exclusion: Guide, current: Guide, preferred_width):
    a, b = rows
    step = max(1, (b - a) // 2400)
    values = image[a:b:step]
    quantile_low, quantile_high = np.percentile(values, [12, 88], axis=0)
    movement = quantile_high - quantile_low
    blur = ndimage.gaussian_filter1d(movement, sigma=1.4)
    blur[exclusion.x0:exclusion.x1] = 0
    midpoint = ndimage.uniform_filter1d(blur, size=max(3, preferred_width // 5))
    threshold = max(0.03, float(np.percentile(midpoint, 93)) * 0.32)
    bands = runs(midpoint >= threshold, min_count=4)
    results = []
    for left, right in bands:
        band = Box(left, a, right, b)
        evidence = float(np.mean(blur[left:right]))
        if band.width > round(preferred_width * 2.8):
            continue
        padding = max(2, min(9, preferred_width // 9))
        resolved = Guide(max(0, left - padding), min(image.shape[1], right + padding))
        if resolved.x0 < exclusion.x1 and resolved.x1 > exclusion.x0:
            if right <= exclusion.x0:
                resolved = Guide(resolved.x0, exclusion.x0)
            elif left >= exclusion.x1:
                resolved = Guide(exclusion.x1, resolved.x1)
            else:
                continue
        overlap = max(0, min(current.x1, right) - max(current.x0, left))
        results.append({"guide": resolved.as_list(), "evidence": round(evidence, 5),
                        "motion_width": band.width, "overlap": overlap,
                        "score": evidence * np.sqrt(band.width) * (1 + overlap / max(1, band.width) * 0.35)})
    results.sort(key=lambda result: result["score"], reverse=True)
    for result in results:
        result["score"] = round(float(result["score"]), 5)
    return {"source_rows": [a, b], "samples": len(values), "candidates": results[:8],
            "column_modulation": [round(float(v), 5) for v in movement], "threshold": round(threshold, 5)}
