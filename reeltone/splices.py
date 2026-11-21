from dataclasses import dataclass

import numpy as np
from scipy import ndimage, signal

from .contracts import integer, real
from .errors import BenchError
from .geometry import Box
from .raster import encode_image
from .visual import ACCENT, MUTED, PAGE, PANEL, RED, TEAL, PlotSeries, line_plot, place_film, print_text
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class Transition:
    edge: int
    recognition_band: tuple[int, int]
    lateral_support: float
    score: float
    signed_delta: float
    seam_width: int
    strength: str

    def as_dict(self):
        return {"edge": self.edge, "recognition_columns": list(self.recognition_band), "supported_fraction": round(self.lateral_support, 5),
                "score": round(self.score, 5), "signed_step": round(self.signed_delta, 5), "transition_width": self.seam_width,
                "strength": self.strength}


def scan_baselines(values, span):
    profile = np.median(values, axis=1)
    smooth_length = max(3, min(33, span // 3 * 2 + 1))
    trend = ndimage.median_filter(profile, size=smooth_length, mode="nearest")
    contrast = values - trend[:, None]
    return profile, trend, contrast


def step_evidence(values, radius):
    size = len(values)
    effective = len(values) - radius * 2 + 1
    if effective < 1:
        raise BenchError("区间长度不足以在接缝两侧独立取样。")
    padding = np.vstack((np.zeros((1, values.shape[1]), dtype=np.float64), np.cumsum(values, axis=0, dtype=np.float64)))
    pivots = np.arange(radius, size - radius + 1)
    before = (padding[pivots] - padding[pivots - radius]) / radius
    after = (padding[pivots + radius] - padding[pivots]) / radius
    step = after - before
    center = np.median(step, axis=1)
    absolute = np.abs(step)
    evidence = np.median(absolute, axis=1)
    agree = (step * np.sign(center)[:, None] > 0).mean(axis=1)
    return pivots, center, evidence, agree, step


def collect_transitions(values, grid, min_contrast, geometry, box, window, min_separation):
    pivots, baseline_delta, strength, sides, delta = grid
    score = strength * sides * sides
    high = score >= min_contrast * 0.8
    detect = np.where(high, score, 0.0)
    peaks, _ = signal.find_peaks(np.r_[0, detect, 0], height=min_contrast * 0.8, distance=min_separation)
    quantiles = np.percentile(values, [4, 95])
    extent = max(0.025, float(quantiles[1] - quantiles[0]))
    candidates = []
    for peak in peaks - 1:
        relevant = delta[peak]
        magnitude = baseline_delta[peak]
        side_count = float((np.abs(relevant) >= min_contrast * 0.7).mean())
        if side_count < 0.24 or abs(magnitude) < min_contrast * 0.4:
            continue
        first = max(0, peak - window)
        last = min(len(detect), peak + window + 1)
        spread = max(1, int(np.count_nonzero(detect[first:last] >= detect[peak] * 0.72)))
        signal_certainty = "strong" if side_count > 0.66 and strength[peak] >= extent * 0.34 else "candidate"
        candidates.append(Transition(int(pivots[peak]) + box.y0, (geometry.x0, geometry.x1), side_count,
                                     float(score[peak]), float(magnitude), spread, signal_certainty))
    return sorted(candidates, key=lambda entry: (-entry.score, entry.edge))


def trace_bands(grid, rows):
    pivots, centers, strength, agreement, _ = grid
    count = min(len(pivots), 540)
    blocks = np.linspace(0, len(pivots), count + 1, dtype=int)
    result = []
    for a, b in zip(blocks[:-1], blocks[1:]):
        signal_score = strength[a:b] * agreement[a:b] ** 2
        at = a + int(np.argmax(signal_score))
        result.append({"edge": int(pivots[at]) + rows[0], "mean_step": round(float(centers[at]), 5),
                       "evidence": round(float(signal_score.max()), 5), "support": round(float(agreement[at]), 4)})
    return result


def splice_plate(material, hits, region):
    caption = "REELTONE  /  CROSS-STRIP TRANSITIONS"
    if not hits:
        return line_plot(caption, [], x_range=(region.y0, region.y1), y_range=(0, 1), formatter=lambda x: f"row {x:.0f}")
    hits = hits[:6]
    column = min(len(hits), 3)
    width = 990
    row_count = (len(hits) + column - 1) // column
    height = 79 + 214 * row_count
    canvas = Image.new("RGB", (width, height), PAGE)
    draw = ImageDraw.Draw(canvas)
    print_text(draw, (20, 16), caption, 16, TEAL)
    w = (width - 29) // column
    for n, hit in enumerate(hits):
        a = 15 + (n % column) * w
        y = 55 + (n // column) * 214
        area = Box(a, y, a + w - 11, y + 198)
        draw.rounded_rectangle(tuple(area.as_list()), radius=8, fill=PANEL)
        geometry = Box(max(0, region.x0 - 6), max(0, hit.edge - 35), min(material.width, region.x1 + 6), min(material.height, hit.edge + 35))
        inner = Box(a + 19, y + 58, area.x1 - 18, area.y1 - 20)
        window = place_film(canvas, material.pixels, geometry, material.recipe.orientation, inner, stretch=True)
        x0, ry = window.point(geometry.x0, hit.edge)
        x1, _ = window.point(geometry.x1, hit.edge)
        color = ACCENT if hit.strength == "strong" else RED
        draw.line((min(x0, x1), ry, max(x0, x1), ry), fill=color, width=2)
        print_text(draw, (a + 14, y + 12), f"Row {hit.edge}  •  {hit.strength}", 14)
        print_text(draw, (a + 14, y + 34), f"Support {hit.lateral_support:.0%} · step {hit.signed_delta:+.3f}", 11, MUTED)
    return encode_image(canvas, "PNG")


def inspect_splices(material, values):
    window = integer(values.get("window", 8), "边界两侧取样行数", 3, 30)
    min_contrast = real(values.get("contrast", 0.08), "接缝候选最小明暗差", 0.015, 0.9)
    spacing = integer(values.get("spacing", 40), "最小候选间隔（行）", 6, 1200)
    fallback = Box(material.recipe.aperture.x1, 0, material.width, min(material.height, 4400))
    region = material.region(values, fallback=fallback, minimum=(10, 2 * window + 4))
    if region.width < 10 or region.height < 2 * window + 4:
        raise BenchError("接缝检查至少需要十列和足以包含边界两侧的行数。")
    if region.area > 2_350_000:
        raise BenchError("接缝检查最多可读 235 万像素，请缩小选区。")
    raster = region.subimage(material.pixels)
    grid = step_evidence(raster, window)
    candidates = collect_transitions(raster, grid, min_contrast, region, region, window, spacing)
    trace = trace_bands(grid, (region.y0, region.y1))
    _, trend, without_trend = scan_baselines(raster, window)
    summary = {"parent": material.identity(), "box": region.as_list(), "window_each_side": window, "contrast": min_contrast,
               "candidate_count": len(candidates), "findings": [item.as_dict() for item in candidates[:50]],
               "ordinary_transmission": round(float(np.median(trend)), 5), "background_variation": round(float(np.median(np.abs(without_trend))), 5),
               "edge_profile": trace, "caveat": "检测该带内横向共识的亮度跳变。画格边界、胶带和片面遮蔽可能产生类似信号。"}
    chart = line_plot("SOURCE-ROW TRANSITION SUPPORT", [PlotSeries(TEAL, trace, lambda item: item["evidence"], lambda item: item["edge"], "Across-column evidence"),
                                                       PlotSeries(ACCENT, trace, lambda item: abs(item["mean_step"]), lambda item: item["edge"], "Median step")],
                      x_range=(region.y0, region.y1), y_range=(0, max(0.25, max(row["evidence"] for row in trace))),
                      formatter=lambda x: f"row {x:.0f}", y_formatter=lambda y: f"{y:.2f}")
    return summary, splice_plate(material, candidates, region), chart
