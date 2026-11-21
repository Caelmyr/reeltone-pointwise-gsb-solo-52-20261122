from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from .audition import read_wav, waveform_statistics
from .bench import load_material, render_audition
from .contracts import flag, option, real
from .errors import Conflict
from .visual import ACCENT, BLUE, TEAL, PlotSeries, line_plot


@dataclass(frozen=True)
class ReplayInput:
    use_recipe: str
    ignore_fades: bool
    tolerance: float

    @classmethod
    def parse(cls, data):
        use = option(data.get("recipe", "retained"), ("retained", "current"), "复算读片版本")
        faded = flag(data.get("ignore_fades", False), "略过渐入渐出区域")
        tolerance = real(data.get("tolerance", 0.002), "允许的绝对误差", 0.00001, 0.2)
        return cls(use, faded, tolerance)


def validate_parent(take, rebuilt):
    initial = take["payload"]["parent"]
    actual = rebuilt["parent"]
    if initial["scan_id"] != actual["scan_id"] or initial["source_sha256"] != actual["source_sha256"]:
        raise Conflict("试听保存的原片标识和复算原片不一致。")
    return {
        "source": initial["source_sha256"],
        "baseline_recipe": initial["recipe_id"],
        "evaluated_recipe": actual["recipe_id"],
        "baseline_fingerprint": initial["recipe_fingerprint"],
        "evaluated_fingerprint": actual["recipe_fingerprint"],
        "same_recipe": initial["recipe_fingerprint"] == actual["recipe_fingerprint"],
    }


@dataclass(frozen=True)
class SampleDiff:
    data: np.ndarray
    first: np.ndarray
    second: np.ndarray
    began: int
    ended: int
    match_rate: int
    same_clock: bool
    padding_excluded: int
    tolerance: float

    def describe(self):
        if not self.data.size:
            return {"interpretable": False, "compared_samples": 0, "match_rate": self.match_rate, "same_clock": self.same_clock}
        differences = np.abs(self.data)
        passed = differences <= self.tolerance
        return {"interpretable": True, "compared_samples": len(self.data), "sample_span": [self.began, self.ended],
                "match_rate": self.match_rate, "same_clock": self.same_clock,
                "intentionally_excluded_each_edge": self.padding_excluded, "absolute_error_limit": self.tolerance,
                "match_count": int(passed.sum()), "match_fraction": round(float(passed.mean()), 7),
                "max_absolute_difference": round(float(differences.max()), 7),
                "q99_absolute_difference": round(float(np.percentile(differences, 99)), 7),
                "rms_difference": round(float(np.sqrt(np.mean(self.data ** 2))), 7),
                "first": waveform_statistics(self.first), "second": waveform_statistics(self.second)}


def clock_key(payload):
    fields = payload["clock"]
    segments = tuple((entry.get("native_rows") and tuple(entry["native_rows"]), entry.get("rate"), entry.get("trusted"))
                     for entry in fields.get("local_segments") or ())
    return (fields["native_height"], fields["rows_per_second"], fields["reverse"], tuple(fields["source_rows"]),
            fields.get("mapping", "uniform"), segments)


def compare_waveforms(first_wave, new_wave, old_read, new_read, request):
    a, rate_a = read_wav(first_wave)
    b, rate_b = read_wav(new_wave)
    clock_locked = clock_key(old_read) == clock_key(new_read)
    if rate_a != rate_b or not clock_locked:
        nothing = np.zeros(0, dtype=float)
        return SampleDiff(nothing, nothing, nothing, 0, 0, rate_b, clock_locked, 0, request.tolerance)
    count = min(len(a), len(b))
    margin = 0
    if request.ignore_fades:
        largest = max(old_read["monitor"]["ramp_ms"], new_read["monitor"]["ramp_ms"])
        margin = min(count // 3, round(rate_a * largest / 1000.0))
    end = count - margin
    return SampleDiff(b[margin:end] - a[margin:end], a[margin:end], b[margin:end], margin, end,
                      rate_a, clock_locked, margin, request.tolerance)


def compact_trace(difference, max_entries=320):
    count = len(difference.data)
    if not count:
        return []
    boundaries = np.linspace(0, count, min(count, max_entries) + 1, dtype=int)
    data = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        diff = difference.data[a:b]
        first = difference.first[a:b]
        second = difference.second[a:b]
        data.append({"time": (a + difference.began) / difference.match_rate,
                     "error": float(np.sqrt(np.mean(diff ** 2))), "first": float(np.sqrt(np.mean(first ** 2))),
                     "second": float(np.sqrt(np.mean(second ** 2)))})
    return data


def comparison_figure(diff):
    trace = compact_trace(diff)
    if trace:
        upper = max(max(row["first"], row["second"], row["error"]) for row in trace)
        series = [PlotSeries(color, trace, lambda entry, key=key: entry[key], lambda entry: entry["time"], label)
                  for key, color, label in (("first", TEAL, "Retained"), ("second", ACCENT, "Replayed"), ("error", BLUE, "Delta RMS"))]
        extent = (trace[0]["time"], diff.ended / diff.match_rate)
        return line_plot("MONITOR REPLAY · OUTPUT-SAMPLE COMPARISON", series, x_range=extent, y_range=(0, max(upper, 0.03)),
                         formatter=lambda a: f"{a:.3f}s", y_formatter=lambda a: f"{a:.3f}")
    return None


def retained_replay(catalog, take_id, raw):
    request = ReplayInput.parse(raw)
    take = catalog.get_take(take_id, raw=True)
    actual_stored = sha256(take["wav"]).hexdigest()
    if actual_stored != take["wav_sha256"]:
        raise Conflict("原监听内容和已保存的摘要不匹配。")
    recipe = take["recipe_id"] if request.use_recipe == "retained" else None
    material = load_material(catalog, take["scan_id"], recipe)
    original = take["payload"]
    request_body = {"rows": original["audio"]["source_rows"], "monitor": take["settings"]}
    report, _, _, wave = render_audition(material, request_body)
    identity = validate_parent(take, report)
    current_wav = sha256(wave).hexdigest()
    difference = compare_waveforms(take["wav"], wave, original, report, request)
    remarks = []
    if not difference.same_clock:
        remarks.append("新旧监听使用不同的原片采样时间。两个 WAV 长度分别保留，不能逐样本相减。")
    if original["audio"]["frames"] != report["audio"]["frames"]:
        remarks.append("新旧监听的样本数不同。复算概要会保留各自的播放长度。")
    return {"take": take_id, "identity": identity, "source_rows": request_body["rows"], "mode": request.use_recipe,
            "old_sha256": actual_stored, "new_sha256": current_wav, "identical_wav": current_wav == actual_stored,
            "retained": original["audio"], "replayed": report["audio"], "comparison": difference.describe(), "warnings": remarks}, wave, comparison_figure(difference)
