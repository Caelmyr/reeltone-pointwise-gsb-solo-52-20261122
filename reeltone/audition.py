from dataclasses import dataclass
from fractions import Fraction
from io import BytesIO
import math
import wave

import numpy as np
from scipy import signal

from .contracts import integer, mapping, option, real
from .errors import BenchError
from .optical import OpticalRead
from .timing import RowClock


RATES = (8_000, 12_000, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000)
MAX_MONITOR_SECONDS = 20


@dataclass(frozen=True)
class Monitor:
    mode: str
    rate: int
    gain_db: float
    low_hz: float
    high_hz: float
    ramp_ms: float
    center: str

    @classmethod
    def parse(cls, values):
        values = mapping(values, "监听设置")
        rate = integer(values.get("rate", 24000), "WAV 采样率", min(RATES), max(RATES))
        if rate not in RATES:
            raise BenchError(f"监听采样率须为 {'、'.join(str(x) for x in RATES)} 之一。")
        gain = real(values.get("gain_db", 6), "增益 dB", -24, 30)
        high = real(values.get("high_hz", 0), "低通 Hz", 0, 20_000)
        low = real(values.get("low_hz", 40), "高通 Hz", 0, 1600)
        if high and high <= low + 10:
            raise BenchError("低通频率须高于高通频率。填 0 可关闭过滤。")
        ramp = real(values.get("ramp_ms", 6), "监听渐入渐出", 0, 100)
        center = option(values.get("center", "median"), ("median", "mean", "fixed"), "静音中心")
        mode = option(values.get("mode", "width"), ("width", "light"), "光学测读方式")
        return cls(mode, rate, gain, low, high, ramp, center)

    def as_dict(self):
        return {"mode": self.mode, "rate": self.rate, "gain_db": self.gain_db, "low_hz": self.low_hz,
                "high_hz": self.high_hz, "ramp_ms": self.ramp_ms, "center": self.center}


@dataclass(frozen=True)
class Audition:
    wave: np.ndarray
    scan_values: np.ndarray
    render_rate: int
    source_rows: tuple
    clock: RowClock
    options: Monitor
    dc: float
    monitor_clipping: int
    warnings: tuple[str, ...]

    @property
    def duration(self):
        return len(self.wave) / self.render_rate

    def summary(self):
        return {"duration_seconds": round(self.duration, 8), "source_rows": list(self.source_rows),
                "output_rate": self.render_rate, "frames": len(self.wave), "channels": 1,
                "row_rate": round(self.clock.rows_per_second, 6), "fixed_dc": round(self.dc, 6),
                "monitor_clipping": self.monitor_clipping, "statistics": waveform_statistics(self.wave),
                "warnings": [*self.clock.warnings, *self.warnings]}


def waveform_statistics(values):
    peak = float(np.max(np.abs(values)))
    rms = float(np.sqrt(np.mean(values ** 2)))
    level_db = lambda value: round(20 * math.log10(value), 2) if value > 1e-10 else None
    return {"peak": round(peak, 6), "rms": round(rms, 6), "peak_dbfs": level_db(peak),
            "rms_dbfs": level_db(rms), "dc": round(float(np.mean(values)), 7),
            "positive_count": int(np.count_nonzero(values > 0))}


def normalize_signal(read: OpticalRead, settings: Monitor, clock: RowClock):
    native = read.openness if settings.mode == "width" else read.mean_light
    if clock.reverse:
        native = native[::-1]
    if settings.center == "median":
        center = float(np.median(native))
    elif settings.center == "mean":
        center = float(np.mean(native))
    else:
        center = 0.5
    return np.asarray((native - center) * 2.0, dtype=np.float64), center


def shaping_filter(values, rows_per_second, options):
    nyquist = rows_per_second / 2
    low = options.low_hz or None
    high = options.high_hz or None
    remarks = []
    if low is not None and low >= nyquist * 0.80:
        raise BenchError("高通频率超过该扫描的原生采样能力。")
    if high is not None and high >= nyquist * 0.95:
        remarks.append("低通设置超出了源行速带宽，已保留源信号的可用带宽。")
        high = None
    if low is None and high is None:
        return values.copy(), remarks
    if len(values) < 16:
        remarks.append("所选片段过短，略过监试听滤波。")
        return values.copy(), remarks
    if low is not None and high is not None:
        coefficients = signal.butter(2, (low, high), "bandpass", fs=rows_per_second, output="sos")
    else:
        cutoff, kind = (low, "highpass") if low is not None else (high, "lowpass")
        coefficients = signal.butter(2, cutoff, kind, fs=rows_per_second, output="sos")
    filtered = signal.sosfiltfilt(coefficients, values, padlen=min(24, len(values) - 1))
    return filtered, remarks


def retime(filtered, row_rate, output_rate, allow_long=False):
    needed = max(1, round(len(filtered) * output_rate / row_rate))
    if not allow_long and needed > output_rate * MAX_MONITOR_SECONDS:
        raise BenchError(f"一次试听最多 {MAX_MONITOR_SECONDS} 秒。缩小原片行区间后重试。")
    ratio = Fraction(output_rate / row_rate).limit_denominator(1700)
    if ratio.numerator == ratio.denominator:
        resampled = filtered.copy()
    else:
        resampled = signal.resample_poly(filtered, ratio.numerator, ratio.denominator, padtype="line")
    if len(resampled) < needed:
        resampled = np.pad(resampled, (0, needed - len(resampled)), mode="edge")
    return resampled[:needed]


def apply_fade(values, count):
    if count < 2:
        return values
    fraction = np.linspace(0.0, np.pi / 2.0, count, endpoint=True)
    ramp = np.sin(fraction) ** 2
    values[:count] *= ramp
    values[-count:] *= ramp[::-1]
    return values


def audition(read: OpticalRead, clock: RowClock, options: Monitor, allow_long=False):
    scan, center = normalize_signal(read, options, clock)
    if not allow_long and len(scan) / clock.rows_per_second > MAX_MONITOR_SECONDS:
        raise BenchError(f"一次试听最多 {MAX_MONITOR_SECONDS} 秒。缩小原片行区间后重试。")
    filtered, warnings = shaping_filter(scan, clock.rows_per_second, options)
    formatted = retime(filtered, clock.rows_per_second, options.rate, allow_long=allow_long)
    formatted *= 10.0 ** (options.gain_db / 20.0)
    n_ramp = min(len(formatted) // 2, round(options.rate * options.ramp_ms / 1000.0))
    faded = apply_fade(formatted, n_ramp)
    clipping = int(np.count_nonzero(np.abs(faded) > 1.0))
    if clipping:
        warnings.append(f"试听增益造成 {clipping} 个样本超出 WAV 满幅。可降低增益。")
    if options.rate > clock.rows_per_second * 1.8:
        warnings.append("WAV 的扩展采样率不增加原扫描的可恢复带宽。")
    return Audition(faded, scan, options.rate, read.source_rows, clock, options, center, clipping, tuple(warnings))


def wav_bytes(values, sample_rate):
    sample = np.asarray(values)
    if sample.ndim == 1:
        channels = 1
    elif sample.ndim == 2 and 0 < sample.shape[1] <= 2:
        channels = sample.shape[1]
    else:
        raise BenchError("监听样本必须是单声道或双声道。")
    quantized = np.rint(np.clip(sample, -1.0, 1.0) * 32767.0).astype("<i2")
    output = BytesIO()
    with wave.open(output, "wb") as destination:
        destination.setnchannels(channels)
        destination.setsampwidth(2)
        destination.setframerate(sample_rate)
        destination.writeframes(quantized.tobytes(order="C"))
    return output.getvalue()


def read_wav(data, max_duration=MAX_MONITOR_SECONDS):
    try:
        with wave.open(BytesIO(data)) as recording:
            rate = recording.getframerate()
            channels = recording.getnchannels()
            frames = recording.getnframes()
            if rate not in RATES or channels not in (1, 2) or recording.getsampwidth() != 2:
                raise BenchError("监听文件须为标准 16 位 PCM WAV。")
            if frames / rate > max_duration or frames < 1:
                raise BenchError("监听文件时长超出范围。")
            audio = np.frombuffer(recording.readframes(frames), dtype="<i2")
            if len(audio) != frames * channels:
                raise BenchError("WAV 头和样本长度不一致。")
    except (wave.Error, EOFError, ValueError):
        raise BenchError("无法读取监听文件。") from None
    values = audio.reshape(frames, channels).astype(np.float64) / 32767.0
    return values[:, 0] if channels == 1 else values, rate


def waveform_envelope(result: Audition, size=600):
    data = result.wave
    bounds = np.linspace(0, len(data), min(len(data), size) + 1, dtype=int)
    start_time, _ = result.clock.extent(result.source_rows)
    output = []
    for first, stop in zip(bounds[:-1], bounds[1:]):
        values = data[first:stop]
        time0 = start_time + first / result.render_rate
        time1 = start_time + stop / result.render_rate
        native_edges = result.clock.native_center_at_time([time0, time1])
        output.append({"samples": [int(first), int(stop)], "start_seconds": round(first / result.render_rate, 7),
                       "source_edges": [round(float(edge), 3) for edge in native_edges],
                       "low": round(float(values.min()), 5), "high": round(float(values.max()), 5),
                       "rms": round(float(np.sqrt(np.mean(values ** 2))), 5)})
    return output
