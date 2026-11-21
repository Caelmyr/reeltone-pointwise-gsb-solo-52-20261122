from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from scipy import signal

from .audition import RATES, read_wav, wav_bytes, waveform_statistics
from .contracts import identifier, integer, mapping, option, real
from .errors import BenchError, Conflict, Limit
from .timing import clock_from_dict
from .visual import ACCENT, BLUE, TEAL, PlotSeries, line_plot


@dataclass(frozen=True)
class MonitorProvenance:
    id: str
    recipe_id: str
    source_rows: tuple[int, int]
    track: tuple[int, int]
    rate: int
    rows_per_second: float
    reverse: bool
    clock: object
    waveform: np.ndarray
    origin: str

    @classmethod
    def load(cls, stored, scan):
        if stored["scan_id"] != scan["id"] or stored["payload"]["parent"]["source_sha256"] != scan["sha256"]:
            raise Conflict("A/B 监听须引用同一张原始扫描。")
        if sha256(stored["wav"]).hexdigest() != stored["wav_sha256"]:
            raise Conflict(f"监听“{stored['title']}”的文件内容发生变化。")
        wave, rate = read_wav(stored["wav"])
        if wave.ndim != 1:
            raise BenchError("A/B 检查只处理档案中的单声道监听快照。")
        clock = stored["payload"]["clock"]
        sound = stored["payload"]["audio"]
        if rate != sound["output_rate"] or len(wave) != sound["frames"]:
            raise Conflict("监听快照记录的采样时标与文件不一致。")
        rail = stored["payload"]["aperture"]
        return cls(stored["id"], stored["recipe_id"], tuple(sound["source_rows"]), tuple(rail), rate, clock["rows_per_second"],
                   clock["reverse"], clock_from_dict(clock), wave, stored["wav_sha256"])

    def describe(self):
        return {"take": self.id, "recipe": self.recipe_id, "source_rows": list(self.source_rows), "guide": list(self.track),
                "rate": self.rate, "reverse": self.reverse, "rows_per_second": self.rows_per_second,
                "clock_mapping": "local_perforation" if self.clock.local else "uniform",
                "wav_sha256": self.origin}

    def sample_for_row_center(self, location):
        lo, hi = self.source_rows
        start = self.clock.edge_to_time(hi if self.reverse else lo)
        return (self.clock.edge_to_time(location) - start) * self.rate - 0.5

    def filter_for_grid(self, overlap, grid_count):
        span = overlap[1] - overlap[0]
        duration = self.clock.duration(overlap)
        input_samples_per_row = self.rate * duration / span if span else self.rate / self.rows_per_second
        output_samples_per_row = grid_count / span
        ratio = output_samples_per_row / input_samples_per_row
        if ratio > 0.98 or len(self.waveform) < 23:
            return self.waveform
        coefficients = signal.butter(6, min(0.95, ratio) * 0.82, output="sos")
        return signal.sosfiltfilt(coefficients, self.waveform, padlen=min(33, len(self.waveform) - 1))

    def sample_grid(self, grid, overlap):
        positions = self.sample_for_row_center(grid)
        filtered = self.filter_for_grid(overlap, len(grid))
        present = (positions >= -0.500001) & (positions <= len(filtered) - 0.499999)
        samples = np.interp(positions, np.arange(len(filtered)), filtered, left=filtered[0], right=filtered[-1])
        return samples, present


@dataclass(frozen=True)
class ListeningPair:
    left: MonitorProvenance
    right: MonitorProvenance
    rows: tuple[int, int]
    rate: int
    mode: str
    factor: float
    common_support: np.ndarray
    native_positions: np.ndarray
    pre_scaled: np.ndarray
    scaled: np.ndarray

    def archive(self):
        return wav_bytes(self.scaled, self.rate)

    def difference(self):
        return self.scaled[:, 1] - self.scaled[:, 0]

    def statistics(self):
        support = self.common_support
        actual = self.scaled[support]
        original = self.pre_scaled[support]
        delta = self.difference()[support]
        silence = np.max(np.abs(actual), axis=1) < 0.005
        return {"a": waveform_statistics(actual[:, 0]), "b": waveform_statistics(actual[:, 1]),
                "difference": waveform_statistics(delta), "unmatched_level_a": waveform_statistics(original[:, 0]),
                "unmatched_level_b": waveform_statistics(original[:, 1]), "silent_fraction": round(float(silence.mean()), 6),
                "difference_abs95": round(float(np.percentile(np.abs(delta), 95)), 6),
                "correlation": normalized_correlation(actual[:, 0], actual[:, 1]),
                "over_range": int(np.count_nonzero(np.abs(actual) > 1.0))}

    def describe(self):
        digest = sha256(self.archive()).hexdigest()
        return {"a": self.left.describe(), "b": self.right.describe(), "native_rows": list(self.rows), "reference": "take_a_read_clock",
                "sample_rate": self.rate, "match_mode": self.mode, "combined_sample_count": len(self.pre_scaled),
                "samples_in_both": int(self.common_support.sum()), "right_gain": round(self.factor, 6),
                "duration": round(len(self.pre_scaled) / self.rate, 7),
                "statistics": self.statistics(), "audio_sha256": digest}


def normalized_correlation(a, b):
    ac = a - a.mean()
    bc = b - b.mean()
    denominator = np.dot(ac, ac) ** 0.5 * np.dot(bc, bc) ** 0.5
    return round(float(np.dot(ac, bc) / denominator), 6) if denominator > 1e-12 else None


def equal_level(a, b, mask, mode, cap_db):
    if mode == "none":
        return 1.0
    aa, bb = a[mask], b[mask]
    if mode == "rms":
        measures = np.sqrt(np.array([np.mean(aa ** 2), np.mean(bb ** 2)]))
    else:
        measures = [np.percentile(np.abs(aa), 96), np.percentile(np.abs(bb), 96)]
    if min(measures) < 0.0015:
        raise BenchError("双方共看的片段至少一侧几乎静音，无法可靠进行等响比较。")
    cap = 10 ** (cap_db / 20.0)
    return float(np.clip(measures[0] / measures[1], 1 / cap, cap))


def intersect_rows(a, b, requested=None):
    left = max(a.source_rows[0], b.source_rows[0])
    right = min(a.source_rows[1], b.source_rows[1])
    if requested:
        left, right = max(left, requested[0]), min(right, requested[1])
    if right - left < 8:
        raise Conflict("两份监听没有足够的共同原片行供核对。")
    return left, right


def make_pair(a, b, count_limit, rate, gain_mode, gain_cap, requested=None):
    if a.id == b.id:
        raise BenchError("请选择两份不同的监听快照。")
    common = intersect_rows(a, b, requested)
    lo, hi = common
    count = max(1, round(a.clock.duration(common) * rate))
    if count > count_limit:
        raise Limit("双耳对听最多包含 16 秒的共同扫描区间。")
    if count < 8:
        raise BenchError("共同区间对指定采样率过短。")
    grid = (np.arange(count) + 0.5) * (hi - lo) / count
    locations = hi - grid if a.reverse else lo + grid
    left, present_a = a.sample_grid(locations, common)
    right, present_b = b.sample_grid(locations, common)
    admit = present_a & present_b
    if np.count_nonzero(admit) < 8:
        raise Conflict("两份监听快照在共同区间内的采样支持不足。")
    factor = equal_level(left, right, admit, gain_mode, gain_cap)
    original = np.stack((left, right), axis=-1)
    adjusted = np.stack((left, right * factor), axis=-1)
    adjusted[~admit] = 0.0
    return ListeningPair(a, b, common, rate, gain_mode, factor, admit, locations, original, adjusted)


def module_plot(pair):
    count = len(pair.scaled)
    boundaries = np.linspace(0, count, min(count, 350) + 1, dtype=int)
    records = []
    displacement = pair.difference()
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        values = pair.scaled[a:b]
        if pair.common_support[a:b].any():
            row = float(pair.native_positions[(a + b - 1) // 2])
            energy = np.sqrt(np.mean(values ** 2, axis=0))
            delta = float(np.sqrt(np.mean(displacement[a:b] ** 2)))
            records.append({"row": row, "a": float(energy[0]), "b": float(energy[1]), "difference": delta})
    lines = [PlotSeries(color, records, lambda r, name=key: r[name], lambda r: r["row"], label)
             for key, label, color in (("a", "Take A RMS", TEAL), ("b", "Take B RMS", ACCENT), ("difference", "Difference RMS", BLUE))]
    maximum = max(max(value["a"], value["b"], value["difference"]) for value in records)
    return line_plot("A/B AUDITION  ·  NATIVE-ROW ENERGY", lines, x_range=pair.rows, y_range=(0, max(0.15, maximum)),
                     formatter=lambda n: f"{n:.0f}", y_formatter=lambda n: f"{n:.2f}")


def retained_pair(catalog, scan_id, values):
    values = mapping(values, "A/B 监听请求")
    a_id = identifier(values.get("a"), "监听 A")
    b_id = identifier(values.get("b"), "监听 B")
    mode = option(values.get("gain", "none"), ("none", "rms", "percentile"), "响度匹配")
    cap = real(values.get("cap", 9), "匹配增益上限 dB", 0, 24)
    rate = integer(values.get("rate", 24000), "输出采样率", min(RATES), max(RATES))
    if rate not in RATES:
        raise BenchError("请选择支持的监听采样率。")
    scan = catalog.get_scan(scan_id)
    left = MonitorProvenance.load(catalog.get_take(a_id, raw=True), scan)
    right = MonitorProvenance.load(catalog.get_take(b_id, raw=True), scan)
    pair = make_pair(left, right, rate * 16, rate, mode, cap)
    return pair.describe(), pair.archive(), module_plot(pair)
