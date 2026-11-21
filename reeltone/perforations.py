from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .errors import BenchError
from .geometry import Guide, runs


@dataclass(frozen=True)
class Hole:
    start: int
    stop: int
    center: float
    width: int
    contrast: float
    clipped: bool
    trusted: bool

    def as_dict(self):
        return {"rows": [self.start, self.stop], "center": round(self.center, 3), "width": self.width,
                "contrast": round(self.contrast, 4), "clipped": self.clipped, "trusted": self.trusted}


@dataclass(frozen=True)
class Spacing:
    first: float
    second: float
    steps: int
    pitch: float
    residual: float
    trusted: bool

    def as_dict(self):
        return {"first": round(self.first, 3), "second": round(self.second, 3), "steps": self.steps,
                "pitch": round(self.pitch, 4), "residual": round(self.residual, 4), "trusted": self.trusted}


@dataclass(frozen=True)
class PerforationReport:
    source_rows: tuple
    threshold: float
    low: float
    high: float
    estimate: float | None
    deviation: float | None
    holes: tuple[Hole, ...]
    spacing: tuple[Spacing, ...]
    missing_estimate: int
    warning: str | None

    @property
    def trusted_count(self):
        return sum(hole.trusted for hole in self.holes)

    def as_dict(self, maximum=220, *, models=False):
        payload = {
            "source_rows": list(self.source_rows),
            "threshold": round(self.threshold, 4),
            "low": round(self.low, 4),
            "high": round(self.high, 4),
            "pitch": round(self.estimate, 4) if self.estimate is not None else None,
            "deviation": round(self.deviation, 4) if self.deviation is not None else None,
            "candidate_count": len(self.holes),
            "trusted_count": self.trusted_count,
            "trusted_intervals": sum(interval.trusted for interval in self.spacing),
            "estimated_missing_holes": self.missing_estimate,
            "warning": self.warning,
            "holes_list_capped": len(self.holes) > maximum,
        }
        if models:
            payload["holes"] = [hole.as_dict() for hole in self.holes[:maximum]]
            payload["spacing"] = [interval.as_dict() for interval in self.spacing[:maximum]]
        return payload


def projected_holes(values, pitch):
    row_probe = np.quantile(values, 0.62, axis=1)
    lower, upper = (float(value) for value in np.percentile(row_probe, [20, 92]))
    cutoff = lower + (upper - lower) * 0.55
    if upper - lower < 0.055:
        return row_probe, [], cutoff, lower, upper
    on = row_probe >= cutoff
    hole_close = max(1, min(7, round(pitch * 0.035)))
    if hole_close > 1:
        on = ndimage.binary_closing(on, structure=np.ones(hole_close), border_value=0)
        on[:hole_close // 2] |= row_probe[:hole_close // 2] >= cutoff
        on[-hole_close // 2:] |= row_probe[-hole_close // 2:] >= cutoff
    found = runs(on, min_count=max(2, round(pitch * 0.04)))
    return row_probe, found, cutoff, lower, upper


def classify_holes(values, probe, candidates, cutoff, black, pitch, source_top):
    conservative_max = max(7, int(pitch * 0.83))
    conservative_min = max(3, int(pitch * 0.09))
    results = []
    for start, stop in candidates:
        local = probe[start:stop]
        weights = np.maximum(1e-5, local - cutoff)
        center = float(np.average(np.arange(start, stop) + source_top + 0.5, weights=weights))
        column_support = np.mean(values[start:stop] >= cutoff, axis=0)
        width = int(np.count_nonzero(column_support >= 0.40))
        contrast = float(np.median(local) - black)
        clipped = start <= 1 or stop >= len(probe) - 1
        trusted = (not clipped and conservative_min <= stop - start <= conservative_max
                   and width >= max(3, int(values.shape[1] * 0.15)) and contrast >= 0.060)
        results.append(Hole(start + source_top, stop + source_top, center, width, contrast, clipped, trusted))
    return tuple(results)


def validate_spacings(holes, initial_pitch):
    centers = np.asarray([hole.center for hole in holes if hole.trusted])
    distances = np.diff(centers)
    likely_unit = distances[(distances >= initial_pitch * 0.72) & (distances <= initial_pitch * 1.30)]
    if likely_unit.size < 2:
        return None, None, (), 0
    pitch = float(np.median(likely_unit))
    intervals = []
    missed = 0
    one_step = []
    for a, b in zip(centers[:-1], centers[1:]):
        delta = float(b - a)
        steps = max(1, round(delta / pitch))
        per_hole = delta / steps
        residual = abs(per_hole - pitch) / pitch
        trusted = steps <= 5 and residual <= 0.18
        intervals.append(Spacing(float(a), float(b), steps, per_hole, residual, trusted))
        if trusted and steps > 1:
            missed += steps - 1
        if trusted and steps == 1:
            one_step.append(per_hole)
    if len(one_step) < 2:
        return None, None, tuple(intervals), missed
    confirmed = float(np.median(one_step))
    dispersion = float(np.median(np.abs(np.array(one_step) - confirmed)))
    return confirmed, dispersion, tuple(intervals), missed


def inspect_perfs(image, guide: Guide, pitch, rows=None):
    source_top, source_bottom = rows if rows is not None else (0, image.shape[0])
    if source_bottom - source_top < 24:
        raise BenchError("齿孔检测区间至少要有 24 行。")
    values = image[source_top:source_bottom, guide.x0:guide.x1]
    probe, candidates, cutoff, low, high = projected_holes(values, pitch)
    holes = classify_holes(values, probe, candidates, cutoff, low, pitch, source_top)
    estimate, deviation, spacing, missed = validate_spacings(holes, pitch)
    warning = None
    if high - low < 0.055:
        warning = "齿孔带明暗差太小，请检查所选列。"
    elif estimate is None:
        warning = "可确认的相邻孔不足，无法从该区间给出定时。"
    elif deviation and deviation > estimate * 0.09:
        warning = "齿孔间距波动较大，此处只能采用区间的代表速度。"
    return PerforationReport((source_top, source_bottom), cutoff, low, high, estimate, deviation, holes,
                             spacing, missed, warning)
