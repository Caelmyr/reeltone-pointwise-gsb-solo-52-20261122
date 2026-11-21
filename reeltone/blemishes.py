from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .errors import Limit
from .geometry import Box


MAX_QC_PIXEL_COUNT = 2_500_000


@dataclass(frozen=True)
class Mark:
    box: Box
    pixels: int
    same_sign: str
    center: tuple[float, float]
    peak: float
    median: float
    touches_border: bool

    def as_dict(self):
        return {"box": self.box.as_list(), "pixels": self.pixels, "sign": self.same_sign,
                "center": [round(self.center[0], 1), round(self.center[1], 1)],
                "peak_contrast": round(self.peak, 5), "median_contrast": round(self.median, 5),
                "touches_border": self.touches_border, "elongation": round(self.box.height / self.box.width, 3)}


@dataclass(frozen=True)
class BlemishMap:
    region: Box
    marks: tuple[Mark, ...]
    bright: np.ndarray
    dark: np.ndarray
    total: int
    threshold: float
    flattened_range: tuple[float, float]

    def summary(self, limit=60):
        largest = sorted(self.marks, key=lambda entry: (entry.peak, entry.pixels), reverse=True)
        bright = int(self.bright.sum())
        dark = int(self.dark.sum())
        return {"region": self.region.as_list(), "bright_pixels": bright, "dark_pixels": dark,
                "coverage": round((bright + dark) / self.region.area, 7), "threshold": self.threshold,
                "candidate_count": self.total, "shown_regions": min(len(largest), limit),
                "background_percentiles": [round(x, 5) for x in self.flattened_range],
                "marks": [mark.as_dict() for mark in largest[:limit]],
                "scope": "使用周围局部纹理作为背景。跨越声轨轮廓的信号变化也可能成为候选。"}


def without_tiny(mask, min_pixels):
    labels, total = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    counts = np.bincount(labels.ravel(), minlength=total + 1)
    chosen = counts >= min_pixels
    chosen[0] = False
    return labels, chosen, counts


def summaries(labels, chosen, counts, residual, region, sign, limit):
    index = np.flatnonzero(chosen)
    order = sorted(index, key=lambda idx: counts[idx], reverse=True)
    slices = ndimage.find_objects(labels)
    selected = []
    for label in order[:limit]:
        dy, dx = slices[label - 1]
        component = labels[dy, dx] == label
        local = np.abs(residual[dy, dx][component])
        ys, xs = np.nonzero(component)
        box = Box(region.x0 + dx.start, region.y0 + dy.start, region.x0 + dx.stop, region.y0 + dy.stop)
        center = (float(box.x0 + xs.mean() + 0.5), float(box.y0 + ys.mean() + 0.5))
        touches = box.x0 == region.x0 or box.x1 == region.x1 or box.y0 == region.y0 or box.y1 == region.y1
        selected.append(Mark(box, int(counts[label]), sign, center, float(local.max()), float(np.median(local)), touches))
    return tuple(selected)


def scan_blemishes(image, region, *, threshold=0.17, support=7, min_pixels=5):
    if region.area > MAX_QC_PIXEL_COUNT:
        raise Limit("瑕疵巡检每次最多 250 万原生像素，请缩小框选。")
    raw = region.subimage(image)
    vy = min(support, (region.height - 1) // 2 * 2 + 1)
    vx = min(support, (region.width - 1) // 2 * 2 + 1)
    background = ndimage.median_filter(raw, size=(vy, vx), mode="reflect")
    residual = raw - background
    above = without_tiny(residual > threshold, min_pixels)
    below = without_tiny(residual < -threshold, min_pixels)
    bright = above[1][above[0]]
    dark = below[1][below[0]]
    total = int(above[1].sum() + below[1].sum())
    marks = summaries(*above, residual, region, "bright", 90) + summaries(*below, residual, region, "dark", 90)
    quantiles = tuple(float(x) for x in np.percentile(background, [5, 95]))
    return BlemishMap(region, marks, bright, dark, total, threshold, quantiles)


def mark_overlaps(source_boxes, region):
    warnings = []
    for known in source_boxes:
        intersection = known.intersection(region)
        if intersection is not None:
            warnings.append({"box": known.as_list(), "overlap_pixels": intersection.area,
                             "its_fraction": round(intersection.area / known.area, 5)})
    return warnings
