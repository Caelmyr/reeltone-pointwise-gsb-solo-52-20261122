from dataclasses import dataclass

import numpy as np
from scipy import optimize

from .contracts import mapping, real, sequence, text
from .errors import BenchError, Limit
from .geometry import Box
from .visual import ACCENT, RED, TEAL, PlotSeries, line_plot


@dataclass(frozen=True)
class Step:
    label: str
    box: Box
    target: float
    median: float
    low: float
    high: float
    clipped: float
    weight: float

    @property
    def spread(self):
        return self.high - self.low

    def as_dict(self):
        return {"label": self.label, "box": self.box.as_list(), "target_transmission": round(self.target, 6),
                "median": round(self.median, 6), "q10": round(self.low, 6), "q90": round(self.high, 6),
                "spread": round(self.spread, 6), "clipped_fraction": round(self.clipped, 6), "weight": round(self.weight, 5)}


@dataclass(frozen=True)
class Fit:
    black: float
    white: float
    gamma: float
    objective: float
    issued: bool

    def predicted(self, drive):
        return self.black + (self.white - self.black) * np.asarray(drive).clip(0, 1) ** self.gamma

    def as_dict(self):
        return {"predicted_black": round(self.black, 6), "predicted_white": round(self.white, 6),
                "gamma": round(self.gamma, 6), "robust_error": round(self.objective, 7), "converged": self.issued}


def extract_steps(image, raw):
    configured = sequence(raw, "阶梯参照", minimum=3, maximum=12)
    height, width = image.shape
    steps = []
    total_pixels = 0
    for n, candidate in enumerate(configured, 1):
        candidate = mapping(candidate, "参照记录")
        label = text(candidate.get("label", f"阶 {n}"), "参照标签", maximum=55)
        transmission = real(candidate.get("target"), "已知透射目标", 0, 1)
        box = Box.parse(candidate.get("box"), width, height, minimum=(3, 5), name=f"第 {n} 个参照框")
        if any(box.intersection(old.box) for old in steps):
            raise BenchError("阶梯取样框彼此不能重叠。")
        if any(abs(transmission - old.target) < 0.00075 for old in steps):
            raise BenchError("每个参照阶须拥有不同的目标透射值。")
        total_pixels += box.area
        if total_pixels > 720_000:
            raise Limit("所有参照阶累计最多读取 72 万原生像素。")
        data = box.subimage(image)
        lo, mid, hi = np.percentile(data, [10, 50, 90])
        clipped = float(np.mean((data <= 0.005) | (data >= 0.995)))
        relative_uncertainty = float(hi - lo) + 0.025
        weight = 1 / relative_uncertainty ** 1.1
        steps.append(Step(label, box, transmission, float(mid), float(lo), float(hi), clipped, weight))
    return sorted(steps, key=lambda step: step.target)


def robust_fit(steps, baseline):
    admissible = [step for step in steps if step.clipped < 0.75]
    if len(admissible) < 3:
        return None, ["多数参照阶处于扫描码值边界，剩余点不足。"]
    drive = np.array([step.target for step in admissible])
    if float(np.ptp(drive)) < 0.24:
        return None, ["参照目标透射跨度太小，无法分别评估底限、上限和曲线弯曲。"]
    values = np.array([step.median for step in admissible])
    weight = np.asarray([step.weight for step in admissible])
    weight = np.sqrt(weight / weight.max())
    if baseline == "fixed":
        def residual(parameters):
            gain, gamma = parameters
            return weight * (gain * drive ** gamma - values)
        result = optimize.least_squares(residual, (0.95, 1.0), bounds=([0.1, 0.20], [1.0, 4.0]),
                                        loss="soft_l1", f_scale=0.025, max_nfev=120)
        black = 0.0
        gain, gamma = result.x
    else:
        def residual(parameters):
            black, admitted, gamma = parameters
            gain = admitted * (1.0 - black)
            return weight * (black + gain * drive ** gamma - values)
        result = optimize.least_squares(residual, (0.08, 0.82, 1.0), bounds=([0.0, 0.05, 0.20], [0.65, 1.0, 4.0]),
                                        loss="soft_l1", f_scale=0.025, max_nfev=120)
        black, admitted, gamma = result.x
        gain = admitted * (1.0 - black)
    error = float(np.sqrt(np.mean(result.fun ** 2)))
    warnings = []
    if np.any(np.diff(values) < -0.022):
        warnings.append("有相邻的标定阶出现透射反序，请检查取样框或阶梯读数。")
    if error > 0.050:
        warnings.append("拟合残差较大，避免套用单一曲线作为该扫描的校准。")
    if len(admissible) < len(steps):
        warnings.append("重度截断的参照阶仅列为证据，未用于曲线拟合。")
    return Fit(float(black), float(black + gain), float(gamma), error, bool(result.success)), warnings


def witness_range(steps):
    values = []
    for a, b in zip(steps[:-1], steps[1:]):
        advance = b.target - a.target
        amp = b.median - a.median
        if a.high >= b.low:
            status = "overlap" if amp > -0.015 else "reversed"
        else:
            status = "separated"
        values.append({"a": a.label, "b": b.label, "target_interval": [a.target, b.target], "median_delta": round(amp, 6),
                       "response_gain": round(amp / advance, 6), "status": status})
    return values


def predicted_curve(fit):
    positions = np.linspace(0, 1, 151)
    values = fit.predicted(positions)
    flat = [{"drive": float(x), "value": float(y)} for x, y in zip(positions, values)]
    return flat


def calibration_preview(steps, fit):
    step_points = [{"drive": item.target, "median": item.median, "low": item.low, "high": item.high} for item in steps]
    series = [PlotSeries(ACCENT, step_points, lambda row: row["median"], lambda row: row["drive"], "Measured median"),
              PlotSeries(RED, step_points, lambda row: row["high"], lambda row: row["drive"], "Within-patch upper quantile"),
              PlotSeries(RED, step_points, lambda row: row["low"], lambda row: row["drive"], "Within-patch lower quantile")]
    if fit is not None:
        series.append(PlotSeries(TEAL, predicted_curve(fit), lambda row: row["value"], lambda row: row["drive"], "Fitted response"))
    return line_plot("EXPOSURE WEDGE · TRANSMISSION RESPONSE", series, x_range=(0, 1), y_range=(0, 1),
                     formatter=lambda x: f"{x:.0%}", y_formatter=lambda x: f"{x:.0%}")


def assess_steps(image, raw, baseline):
    steps = extract_steps(image, raw)
    fit, warnings = robust_fit(steps, baseline)
    if any(item.spread > 0.13 for item in steps):
        warnings.append("有参照框内的码值差异较大，可能跨过两个阶或遇到不均匀照明。")
    if any(item.clipped > 0.15 for item in steps):
        warnings.append("部分参照像素接近黑白边界，请结合截断量判断。")
    response = []
    for row in steps:
        entry = row.as_dict()
        if fit:
            guessed = float(fit.predicted(row.target))
            entry.update({"modeled": round(guessed, 6), "residual": round(row.median - guessed, 6)})
        response.append(entry)
    return {"steps": response, "neighbors": witness_range(steps), "response": fit.as_dict() if fit else None,
            "baseline_mode": baseline, "warnings": warnings}, calibration_preview(steps, fit)
