from dataclasses import dataclass

import numpy as np

from .contracts import integer, real
from .geometry import runs
from .optical import read_area
from .visual import ACCENT, BLUE, TEAL, PlotSeries, line_plot


@dataclass(frozen=True)
class SlitSeries:
    left: np.ndarray
    right: np.ndarray
    admitted: np.ndarray
    origin: int
    pixel_size: float | None
    apparent_light: np.ndarray

    @property
    def midpoints(self):
        return (self.left + self.right) * 0.5

    @property
    def widths(self):
        return self.right - self.left

    @property
    def samples(self):
        return len(self.left)

    def calibrated(self, run):
        numeric = run[self.admitted]
        if numeric.size == 0:
            return {"available": False, "q10_q50_q90": [], "span_q05_q95": None, "physical_mm": None}
        lower, mid, upper = [float(value) for value in np.percentile(numeric, [10, 50, 90])]
        extreme0, extreme1 = [float(value) for value in np.percentile(numeric, [5, 95])]
        physical = None
        if self.pixel_size:
            scale = self.pixel_size
            physical = {"q10_q50_q90": [round(value * scale, 5) for value in (lower, mid, upper)],
                        "span_q05_q95": round((extreme1 - extreme0) * scale, 5)}
        return {"available": True, "q10_q50_q90": [round(lower, 5), round(mid, 5), round(upper, 5)],
                "span_q05_q95": round(extreme1 - extreme0, 5), "physical_mm": physical}

    def data(self):
        return {"rows": [self.origin, self.origin + self.samples], "used": int(self.admitted.sum()),
                "withheld": int(np.count_nonzero(~self.admitted)), "millimeters_per_pixel": self.pixel_size,
                "widths": self.calibrated(self.widths), "center": self.calibrated(self.midpoints),
                "unusable_runs": [[start + self.origin, stop + self.origin] for start, stop in runs(~self.admitted, 3)[:80]]}


def slit_series(material, request):
    from .geometry import Guide

    rows = material.rows(request, min_span=10)
    numeric = request.get("pixels_per_mm")
    pitch = 1 / real(numeric, "每毫米的扫描像素数", 1, 25_000) if numeric not in (None, "") else None
    guide = material.recipe.aperture
    if request.get("columns") is not None:
        guide = Guide.parse(request.get("columns"), material.width, "测量列范围", 8)
    optical = read_area(material.pixels, guide, material.recipe.tone, rows)
    single = optical.islands == 1
    admitted = single & ~optical.touch_edge & np.isfinite(optical.edge_low) & np.isfinite(optical.edge_high)
    admitted &= optical.edge_high - optical.edge_low >= 0.8
    return SlitSeries(optical.edge_low, optical.edge_high, admitted, rows[0], pitch, optical.openness), guide


def slits_by_block(series, target):
    boundaries = np.linspace(0, series.samples, min(target, series.samples) + 1, dtype=int)
    report = []
    midpoints = series.midpoints
    widths = series.widths
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        supported = series.admitted[a:b]
        center = None
        width = None
        if supported.any():
            avg = lambda values: round(float(np.median(values[a:b][supported])), 5)
            center = avg(midpoints)
            width = avg(widths)
        report.append({"rows": [int(a + series.origin), int(b + series.origin)], "center": center, "width": width,
                       "support": round(float(supported.mean()), 5), "opening": round(float(np.mean(series.apparent_light[a:b])), 5)})
    return report


def movement_outliers(series, limit):
    centers = series.midpoints
    if series.samples < 3 or not series.admitted.any():
        return []
    reference = float(np.median(centers[series.admitted]))
    trip = series.admitted & (np.abs(centers - reference) > limit)
    indexes = []
    for a, b in runs(trip, 4):
        nearest = np.asarray(centers[a:b]) - reference
        w = series.widths[a:b]
        indexes.append({"rows": [a + series.origin, b + series.origin], "max_center_deviation": round(float(np.max(np.abs(nearest))), 5),
                        "median_width": round(float(np.median(w)), 5), "drift": "left" if np.median(nearest) < 0 else "right"})
    indexes.sort(key=lambda item: -item["max_center_deviation"])
    return indexes[:60]


def chart(series, reports, guide):
    lines = [PlotSeries(TEAL, reports, lambda item: item["center"], lambda item: item["rows"][0], "Slit center x"),
             PlotSeries(ACCENT, reports, lambda item: guide.x0 + item["width"] if item["width"] is not None else None,
                        lambda item: item["rows"][0], "Width from left guide"),
             PlotSeries(BLUE, reports, lambda item: guide.x0 + item["support"] * guide.width,
                        lambda item: item["rows"][0], "Support · normalized to guide")]
    return line_plot("SLIT CROSS SECTION  ·  GEOMETRY IN NATIVE COLUMNS", lines, x_range=(series.origin, series.origin + series.samples),
                     y_range=(guide.x0, guide.x1), formatter=lambda a: f"{a:.0f}", y_formatter=lambda a: f"{a:.1f}")


def examine_slit(material, request):
    threshold = real(request.get("center_deviation", 3), "偏离中线的告警距离（像素）", 0.2, 180)
    series, guide = slit_series(material, request)
    bin_count = integer(request.get("bins", 240), "沿走片方向图表分箱数", 20, 480)
    reports = slits_by_block(series, bin_count)
    data = {"parent": material.identity(), "guide": guide.as_list(), "summary": series.data(), "readings": reports,
            "deviation_limit_px": threshold, "displaced_runs": movement_outliers(series, threshold),
            "scope": "仅在截面两边界均可见、且读窗内为单一连续信号时计算宽度和中心。"}
    return data, chart(series, reports, guide)
