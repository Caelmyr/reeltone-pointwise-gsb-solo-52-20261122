from dataclasses import dataclass

import numpy as np
from scipy import ndimage, stats

from .exposure import measure_exposure
from .geometry import Box
from .raster import aggregated_rows


@dataclass(frozen=True)
class Pair:
    left: Box
    right: Box
    left_source: str
    right_source: str

    def as_dict(self):
        return {"left": self.left.as_list(), "right": self.right.as_list(), "left_source": self.left_source,
                "right_source": self.right_source}


@dataclass(frozen=True)
class Comparison:
    selection: Pair
    left: np.ndarray
    right: np.ndarray
    delta: np.ndarray
    left_hist: dict
    right_hist: dict
    agreement: dict

    def summary(self):
        return {"selection": self.selection.as_dict(), "left_statistics": self.left_hist,
                "right_statistics": self.right_hist, "agreement": self.agreement,
                "display_grid": [self.delta.shape[1], self.delta.shape[0]],
                "interpretation": "两张图由操作者分别框选作为对照，均按各自框的相对位置读数。差值不作为像素级配准依据。"}


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


def paired_line_profile(comparison):
    length = comparison.delta.shape[0]
    marks = np.linspace(0, length, min(length, 180) + 1, dtype=int)
    selection = comparison.selection
    result = []
    for a, b in zip(marks[:-1], marks[1:]):
        frac0 = float(a / length)
        frac1 = float(b / length)
        result.append({
            "fraction": [round(frac0, 5), round(frac1, 5)],
            "left_rows": [round(selection.left.y0 + f * selection.left.height, 2) for f in (frac0, frac1)],
            "right_rows": [round(selection.right.y0 + f * selection.right.height, 2) for f in (frac0, frac1)],
            "left": round(float(comparison.left[a:b].mean()), 6), "right": round(float(comparison.right[a:b].mean()), 6),
            "delta": round(float(comparison.delta[a:b].mean()), 6),
        })
    return result
