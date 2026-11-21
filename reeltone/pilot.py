from dataclasses import dataclass
from functools import cached_property

import numpy as np
from PIL import Image, ImageDraw
from scipy import signal

from .contracts import integer, real
from .errors import BenchError
from .raster import encode_image
from .visual import ACCENT, GRID, MUTED, PAGE, PANEL, RED, TEAL, TEXT, PlotSeries, line_plot, print_text


@dataclass(frozen=True)
class Standard:
    frequency: float
    band: float
    window_ms: float
    snr: float
    filter_fraction: float
    harmonics: int

    @classmethod
    def parse(cls, request, rate):
        highest = min(9000, rate * 0.38)
        if highest < 58:
            raise BenchError("原扫描行速过低，无法评估该频带的基准音。")
        nominal = real(request.get("frequency", 455), "基准音频率 Hz", 45, highest)
        band = real(request.get("band_percent", 7), "频率查找半宽 %", 2, 22) / 100
        window_ms = real(request.get("window_ms", 140), "分析窗长 ms", 30, 1100)
        snr = real(request.get("snr", 18), "最低峰值显著度 dB", 5, 50)
        filter_fraction = real(request.get("stability_percent", 1.5), "标定偏差提示 %", 0.2, 18) / 100
        harmonics = integer(request.get("harmonics", 3), "记录的谐波数", 1, 7)
        return cls(nominal, band, window_ms, snr, filter_fraction, harmonics)

    @property
    def range(self):
        return self.frequency * (1 - self.band), self.frequency * (1 + self.band)

    def as_dict(self):
        return {"reference_hz": self.frequency, "search_hz": list(self.range), "window_ms": self.window_ms,
                "snr_floor_db": self.snr, "flag_threshold_percent": round(self.filter_fraction * 100, 5),
                "harmonic_limit": self.harmonics}


@dataclass(frozen=True)
class Window:
    start: int
    stop: int
    rate: float
    delta_hz: float
    measured_frequency: float | None
    tone_level: float | None
    background_ratio: float | None
    energy: float
    skew: float | None
    harmonic_power: tuple
    state: str

    @property
    def valid(self):
        return self.state == "trusted"

    def summary(self, reference, native_start, native_end, reverse):
        if reverse:
            a, b = native_end - self.stop, native_end - self.start
        else:
            a, b = native_start + self.start, native_start + self.stop
        difference = (self.measured_frequency - reference) / reference if self.measured_frequency else None
        rounded = lambda value: round(value, 6) if value is not None else None
        return {"local_sample_edges": [self.start, self.stop], "native_rows": [a, b],
                "center_seconds": round((self.start + self.stop) / (2 * self.rate), 7), "state": self.state,
                "frequency": rounded(self.measured_frequency), "resolution": rounded(self.delta_hz),
                "relative_error": rounded(difference), "level_db": rounded(self.tone_level), "snr_db": rounded(self.background_ratio),
                "rms": rounded(self.energy), "spectrum_skew": rounded(self.skew),
                "harmonics": [[index, rounded(freq), rounded(relative)] for index, freq, relative in self.harmonic_power]}


@dataclass(frozen=True)
class AuditedTone:
    expected: Standard
    windows: tuple[Window, ...]
    rate: float
    source: tuple[int, int]
    reverse: bool

    @cached_property
    def trusted(self):
        return [item for item in self.windows if item.valid]

    def reports(self):
        return [window.summary(self.expected.frequency, *self.source, self.reverse) for window in self.windows]

    def summary(self):
        admitted = len(self.trusted)
        data = {"expected": self.expected.as_dict(), "source_rows": list(self.source), "windows": len(self.windows),
                "trusted": admitted, "suspect": len(self.windows) - admitted, "time_direction": "reverse" if self.reverse else "forward",
                "warnings": []}
        if not admitted:
            data["warnings"].append("片段内没有达到门限的基准音。先确认源片中实际录过此频率的正弦参考段。")
            return data
        actual = np.asarray([item.measured_frequency for item in self.trusted])
        level = np.asarray([item.tone_level for item in self.trusted])
        bias = (actual - self.expected.frequency) / self.expected.frequency
        bins = np.percentile(actual, [10, 50, 90])
        data.update({"measured_hz_q10_q50_q90": [round(float(x), 6) for x in bins],
                     "frequency_drift_ppm": round(float(np.median(bias)) * 1e6, 3),
                     "frequency_variation_ppm": round(float(np.std(bias)) * 1e6, 3),
                     "peak_hz_deviation": round(float(np.max(np.abs(actual - self.expected.frequency))), 6),
                     "mean_level_db": round(float(level.mean()), 5),
                     "warning_windows": int(np.count_nonzero(np.abs(bias) > self.expected.filter_fraction))})
        if admitted < max(2, len(self.windows) // 2):
            data["warnings"].append("只有少数分析窗通过基准音核查。总体统计仅使用明确的片段。")
        if data["warning_windows"]:
            data["warnings"].append("部分定位窗口超过了指定容差。波形传输速度、原始录音和扫描节奏均可影响读数。")
        return data


def parabola(values, index):
    if index <= 0 or index >= len(values) - 1:
        return 0.0
    a, b, c = np.log(np.maximum(values[index - 1:index + 2], 1e-30))
    denominator = a - 2 * b + c
    if abs(denominator) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (a - c) / denominator, -0.5, 0.5))


def window_levels(samples):
    length = len(samples)
    prepared = signal.windows.hann(length, sym=False)
    transformed = np.abs(np.fft.rfft((samples - samples.mean()) * prepared))
    return transformed * 2 / max(1e-12, prepared.sum())


def peak_level(values, closest):
    nearest = int(round(closest))
    if nearest >= len(values) or nearest < 1:
        return None
    a = max(1, nearest - 2)
    b = min(len(values), nearest + 3)
    place = a + int(np.argmax(values[a:b]))
    return place, float(values[place])


def tone_window(samples, start, stop, rate, guide):
    clip = samples[start:stop]
    energy = float(np.sqrt(np.mean(clip ** 2)))
    values = window_levels(clip)
    resolution = rate / len(clip)
    left, right = guide.range
    a = max(2, int(np.floor(left / resolution)))
    b = min(len(values) - 2, int(np.ceil(right / resolution)) + 1)
    if b <= a:
        raise BenchError("当前分析窗口不能分辨所选的基准频带。请增大窗口。")
    if energy < 0.0018:
        return Window(start, stop, rate, resolution, None, None, None, energy, None, (), "quiet")
    z = int(a + np.argmax(values[a:b]))
    center_frequency = (z + parabola(values, z)) * resolution
    peak = max(1e-12, float(values[z]))
    background = np.concatenate((values[max(1, a - 25):max(1, z - 2)], values[z + 3:min(len(values), b + 25)]))
    noise = max(1e-10, float(np.percentile(background, 64))) if background.size > 0 else 1e-8
    dominance = 20.0 * np.log10(peak / noise)
    level_db = 20.0 * np.log10(peak)
    skew = float(abs(values[z - 1] - values[z + 1]) / peak)
    if not left <= center_frequency <= right:
        state = "edge"
    elif dominance < guide.snr or peak < 0.003:
        state = "weak"
    else:
        state = "trusted"
    harmonic_power = []
    for harmonic in range(2, guide.harmonics + 1):
        target = center_frequency * harmonic
        found = peak_level(values, target / resolution)
        if found is not None:
            place, magnitude = found
            relative = 20.0 * np.log10(max(1e-12, magnitude) / peak)
            harmonic_power.append((harmonic, place * resolution, float(relative)))
    return Window(start, stop, rate, resolution, float(center_frequency), float(level_db), float(dominance), energy,
                  skew, tuple(harmonic_power), state)


def build_audit(samples, rate, rows, reverse, guide):
    maximum = min(len(samples), round(rate * guide.window_ms / 1000))
    if maximum < max(32, round(rate * 5.0 / guide.frequency)):
        raise BenchError("当前片段须至少包含基准音的五个完整周期。")
    if len(samples) / rate > 22:
        raise BenchError("基准音检测一次支持最多 22 秒。")
    stride = max(maximum // 2, (len(samples) - maximum) // 180)
    boundaries = list(range(0, len(samples) - maximum + 1, stride))
    if boundaries[-1] < len(samples) - maximum - stride // 3:
        boundaries.append(len(samples) - maximum)
    windows = tuple(tone_window(samples, start, start + maximum, rate, guide) for start in boundaries)
    return AuditedTone(guide, windows, rate, tuple(rows), reverse)


def guidance_plot(observation):
    reports = observation.reports()
    model = observation.expected
    series = [PlotSeries(TEAL, reports, lambda a: a["frequency"] if a["state"] == "trusted" else None,
                         lambda a: a["center_seconds"], "Trusted peak"),
              PlotSeries(RED, reports, lambda a: a["frequency"] if a["state"] != "trusted" else None,
                         lambda a: a["center_seconds"], "Withheld peak"),
              PlotSeries(ACCENT, reports, lambda a: model.frequency, lambda a: a["center_seconds"], "Reference tone")]
    return line_plot("REFERENCE TONE BY LISTENING WINDOW", series, x_range=(0, (observation.source[1] - observation.source[0]) / observation.rate),
                     y_range=model.range, formatter=lambda value: f"{value:.3f}s", y_formatter=lambda x: f"{x:.1f} Hz")


def pass_window_plate(observation):
    width, height = 930, 248
    margin, prefix = 25, 150
    page = Image.new("RGB", (width, height), PAGE)
    draw = ImageDraw.Draw(page)
    print_text(draw, (margin, 16), "REFERENCE TONE · WINDOW ADMISSION", 17, TEAL)
    print_text(draw, (margin, 56), f"{observation.expected.frequency:.2f} Hz reference", 13, TEXT)
    labels = (("trusted", TEAL), ("weak", ACCENT), ("edge", RED), ("quiet", GRID))
    current = observation.reports()
    stop = (observation.source[1] - observation.source[0]) / observation.rate
    for i, (state, tone) in enumerate(labels):
        y = 88 + i * 29
        print_text(draw, (margin, y + 9), state.upper(), 12, MUTED, anchor="lm")
        draw.rounded_rectangle((prefix, y, width - margin, y + 18), radius=5, fill=PANEL)
        for item in current:
            if item["state"] != state:
                continue
            a, b = item["local_sample_edges"]
            left = round(prefix + (width - prefix - margin) * a / observation.rate / stop)
            right = round(prefix + (width - prefix - margin) * b / observation.rate / stop)
            draw.rounded_rectangle((left, y + 1, max(left + 3, right), y + 17), radius=5, fill=tone)
    for fraction in np.linspace(0, 1, 8):
        x = round(prefix + fraction * (width - prefix - margin))
        print_text(draw, (x, 218), f"{fraction * stop:.3f}s", 11, MUTED, anchor="mt")
    return encode_image(page, "PNG")


def assay(material, request):
    from .audition import Monitor, normalize_signal
    from .optical import read_area

    rows = material.rows(request, min_span=18)
    clock, _ = material.clock()
    if (rows[1] - rows[0]) / clock.rows_per_second > 22:
        raise BenchError("基准音检测一次支持最多 22 秒。")
    expected = Standard.parse(request, clock.rows_per_second)
    optical = read_area(material.pixels, material.recipe.aperture, material.recipe.tone, rows)
    monitor = Monitor.parse(request.get("monitor", {}))
    base, _ = normalize_signal(optical, monitor, clock)
    observation = build_audit(base, clock.rows_per_second, rows, clock.reverse, expected)
    stats = observation.summary()
    stats.update({"parent": material.identity(), "sample_clock": clock.as_dict(rows), "windows_detail": observation.reports(),
                  "operating_note": "监测原生扫描行的基准音。WAV 重采样率和监听增益不会影响此判断。"})
    return stats, guidance_plot(observation), pass_window_plate(observation)
