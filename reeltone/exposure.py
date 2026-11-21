from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .geometry import Box, runs


@dataclass(frozen=True)
class Histogram:
    count: int
    sampled: int
    lo: float
    hi: float
    mean: float
    deviation: float
    percentiles: tuple
    histogram: np.ndarray
    low_count: int
    high_count: int

    def as_dict(self, include_hist=True):
        record = {"pixel_count": self.count, "sample_count": self.sampled, "minimum": round(self.lo, 6),
                  "maximum": round(self.hi, 6), "mean": round(self.mean, 6), "deviation": round(self.deviation, 6),
                  "percentiles": [round(float(value), 6) for value in self.percentiles],
                  "near_black_pixels": self.low_count, "near_white_pixels": self.high_count,
                  "near_black_fraction": round(self.low_count / self.count, 7),
                  "near_white_fraction": round(self.high_count / self.count, 7)}
        if include_hist:
            record["histogram"] = [int(value) for value in self.histogram]
        return record


def measure_exposure(values, bins=48, sample_limit=300_000):
    count = values.size
    stride = max(1, int(np.ceil(count / sample_limit)))
    flat = values.ravel()
    sample = flat[::stride]
    hist, _ = np.histogram(sample, bins=bins, range=(0.0, 1.0000001))
    quantiles = tuple(np.percentile(sample, [1, 10, 25, 50, 75, 90, 99]))
    return Histogram(count, len(sample), float(values.min()), float(values.max()), float(values.mean()),
                     float(values.std()), quantiles, hist, int(np.count_nonzero(values <= 0.008)),
                     int(np.count_nonzero(values >= 0.992)))


def exposure_map(image, box: Box, *, row_cells=20, col_cells=12):
    column_edges = np.linspace(box.x0, box.x1, min(box.width, col_cells) + 1, dtype=int)
    row_edges = np.linspace(box.y0, box.y1, min(box.height, row_cells) + 1, dtype=int)
    output = []
    for y, bottom in zip(row_edges[:-1], row_edges[1:]):
        row = []
        for x, right in zip(column_edges[:-1], column_edges[1:]):
            stats = measure_exposure(image[y:bottom, x:right], sample_limit=5000)
            row.append({"bounds": [int(x), int(y), int(right), int(bottom)], "mean": round(stats.mean, 5),
                        "range_10_90": round(float(stats.percentiles[5] - stats.percentiles[1]), 5),
                        "low_fraction": round(stats.low_count / stats.count, 5),
                        "high_fraction": round(stats.high_count / stats.count, 5)})
        output.append(row)
    return output


def row_exposure(values, native_start):
    averages = np.mean(values, axis=1)
    zeros = np.mean(values <= 0.008, axis=1)
    maxima = np.mean(values >= 0.992, axis=1)
    bands = np.linspace(0, len(values), min(320, len(values)) + 1, dtype=int)
    density = []
    for a, b in zip(bands[:-1], bands[1:]):
        density.append({"rows": [int(a + native_start), int(b + native_start)],
                        "mean": round(float(averages[a:b].mean()), 5), "min_row_mean": round(float(averages[a:b].min()), 5),
                        "max_row_mean": round(float(averages[a:b].max()), 5),
                        "dark_fraction": round(float(zeros[a:b].mean()), 5),
                        "bright_fraction": round(float(maxima[a:b].mean()), 5)})
    def grouped(values):
        return [[a + native_start, b + native_start] for a, b in runs(values > 0.35, min_count=3)][:60]
    return {"profile": density, "blackout": grouped(zeros), "whiteout": grouped(maxima)}


def exposure_report(image, box):
    values = box.subimage(image)
    stats = measure_exposure(values)
    cells = exposure_map(image, box)
    return {"box": box.as_list(), "statistics": stats.as_dict(), "percentile_levels": [1, 10, 25, 50, 75, 90, 99],
            "tiles": cells, "by_row": row_exposure(values, box.y0)}


def reference_suggestion(dark, light):
    dark_median = float(np.median(dark))
    light_median = float(np.median(light))
    dark_spread = float(np.percentile(dark, 90) - np.percentile(dark, 10))
    light_spread = float(np.percentile(light, 90) - np.percentile(light, 10))
    warnings = []
    if light_median - dark_median < 0.035:
        warnings.append("亮参考需要明显亮于暗参考，当前两块不足以校准。")
    if dark_spread > 0.10 or light_spread > 0.10:
        warnings.append("参考区域不均匀，请避开齿孔、轨道边缘及纹理。")
    if light_median > 0.995 or dark_median < 0.005:
        warnings.append("参考区靠近扫描的码值边界，可能已经饱和。")
    return {"black": round(dark_median, 5), "white": round(light_median, 5),
            "contrast": round(light_median - dark_median, 5), "dark_spread": round(dark_spread, 5),
            "light_spread": round(light_spread, 5), "usable": light_median - dark_median >= 0.035, "warnings": warnings}


def density_rolloff(values):
    profile = np.mean(values, axis=0)
    softened = ndimage.gaussian_filter1d(profile, max(0.8, len(profile) / 100))
    center_columns = softened[len(profile) // 3: max(len(profile) // 3 + 1, len(profile) * 2 // 3)]
    center = float(np.median(center_columns))
    edge_count = max(1, len(profile) // 8)
    return {"center_mean": round(center, 6), "left_delta": round(float(np.mean(softened[:edge_count])) - center, 6),
            "right_delta": round(float(np.mean(softened[-edge_count:])) - center, 6),
            "column_means": [round(float(value), 5) for value in softened]}
