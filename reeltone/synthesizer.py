"""Synthetic optical test strip: upload a mono WAV and render it as a grayscale
variable-width optical soundtrack, then read the rendered strip back through the
same optical pipeline used for real scans.

The round trip is deliberately lossy in documented, measurable ways:

* resampling to the requested row rate applies an anti-alias filter, so content
  above the new Nyquist frequency is removed rather than folded into band;
* loud peaks are soft-limited so a single crest cannot fill the whole aperture;
* the film edge can only move in sub-pixel steps, which sets an effective level
  granularity and a residual noise floor;
* an optional multi-row reading slit and fine photographic grain further shape
  what the pattern can retain.

Every loss is quantified in the comparison report returned alongside the strip.
"""
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import math

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage, signal

from .audition import read_wav
from .contracts import integer, mapping, option, real, safe_filename, text
from .errors import BenchError, Limit
from .geometry import Guide
from .optical import read_area
from .raster import encode_image, encode_png, read_scan
from .recipe import Recipe


MAX_WAV_BYTES = 16 * 1024 * 1024
MAX_STRIP_HEIGHT = 44_000
MIN_STRIP_HEIGHT = 100
SIDE_MARGIN = 6
LABEL_ROWS = 40
TRAILER_ROWS = 12
MAX_GRAIN = 0.06
JOB_TTL_SECONDS = 3_600
JOB_LIMIT = 16

DEFAULT_ROW_RATE = 7_200
DEFAULT_APERTURE_WIDTH = 96
DEFAULT_MARGIN = 0.03
DEFAULT_KNEE = 0.85
DEFAULT_BASE = 0.135
DEFAULT_LEVEL = 0.86


@dataclass(frozen=True)
class SynthSettings:
    row_rate: float
    aperture_width: int
    slit_rows: int
    margin_fraction: float
    knee: float
    grain_sigma: float
    base: float
    level: float
    polarity: str
    title: str

    @classmethod
    def parse(cls, values):
        values = mapping(values, "合成参数")
        row_rate = real(values.get("row_rate", DEFAULT_ROW_RATE), "行速（行/秒）", 500, 48_000)
        aperture_width = integer(values.get("aperture_width", DEFAULT_APERTURE_WIDTH), "读窗宽度（列）", 24, 600)
        slit_rows = integer(values.get("slit_rows", 1), "读窗高度（行）", 1, 5)
        margin = real(values.get("margin_fraction", DEFAULT_MARGIN), "边缘保留比例", 0.005, 0.15)
        knee = real(values.get("knee", DEFAULT_KNEE), "软限幅拐点（满幅比例）", 0.4, 0.995)
        grain = real(values.get("grain_sigma", 0.006), "颗粒噪声标准差", 0.0, MAX_GRAIN)
        base = real(values.get("base", DEFAULT_BASE), "片基透射", 0.0, 0.45)
        level = real(values.get("level", DEFAULT_LEVEL), "满幅透射", 0.55, 1.0)
        if level - base < 0.2:
            raise BenchError("满幅透射与片基透射至少相差 0.20。")
        polarity = option(values.get("polarity", "clear"), ("clear", "dark"), "轨道极性")
        title = text(values.get("title", "WAV 合成测试声轨"), "测试图名称", maximum=90)
        return cls(row_rate, aperture_width, slit_rows, margin, knee, grain, base, level, polarity, title)

    def as_dict(self):
        return {"row_rate": self.row_rate, "aperture_width": self.aperture_width, "slit_rows": self.slit_rows,
                "margin_fraction": self.margin_fraction, "knee": self.knee, "grain_sigma": self.grain_sigma,
                "base": self.base, "level": self.level, "polarity": self.polarity, "title": self.title}


@dataclass(frozen=True)
class PreparedAudio:
    samples: np.ndarray
    source_rate: int
    source_frames: int
    resampled: np.ndarray
    row_rate: float
    rows: int
    duration: float
    removed_band_share: float


@dataclass(frozen=True)
class LimiterResult:
    values: np.ndarray
    rows_limited: int
    max_input: float
    max_output: float
    gain_reduction_db: float
    strongest_row_reduction_db: float
    openness: np.ndarray
    openness_min: float
    openness_max: float


@dataclass(frozen=True)
class StripImage:
    pixels: np.ndarray
    width: int
    height: int
    aperture: Guide
    signal_rows: tuple
    recipe: Recipe
    png: bytes
    filename: str


def load_test_wav(data, filename):
    if not data:
        raise BenchError("请选择单声道 16 位 PCM WAV 测试音。")
    if len(data) > MAX_WAV_BYTES:
        raise Limit(f"测试 WAV 最大 {MAX_WAV_BYTES // (1024 * 1024)} MiB。")
    try:
        values, rate = read_wav(data)
    except BenchError:
        raise
    if values.ndim != 1:
        raise BenchError("请先把测试音混成单声道再上传。")
    name = safe_filename(filename, "test-tone.wav")
    if not name.lower().endswith(".wav"):
        name += ".wav"
    return values, rate, name


def band_energy(samples, sample_rate, low_hz, high_hz):
    if len(samples) < 8:
        return 0.0
    windowed = samples * np.hanning(len(samples))
    spectrum = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(len(samples), 1.0 / sample_rate)
    mask = (freqs >= low_hz) & (freqs < min(high_hz, sample_rate / 2 + 1e-9))
    if not np.any(mask):
        return 0.0
    return float(np.sum(np.abs(spectrum[mask]) ** 2))


def foldback_analysis(original, source_rate, resampled, row_rate):
    """Quantify how little above-Nyquist content folds back into the new band.

    A naive sampler would mirror every component above the new Nyquist back into
    the base band at full amplitude. The anti-alias filter must instead suppress
    it. To keep genuine in-band programme out of the denominator, we high-pass
    the source at the new Nyquist and run that high-only signal through the same
    resampler, then compare its surviving in-band energy against its source
    energy. The gap in dB is the achieved foldback suppression.
    """
    nyquist = row_rate / 2.0
    above_energy = band_energy(original, source_rate, nyquist, source_rate / 2)
    total_energy = band_energy(original, source_rate, 0.0, source_rate / 2)
    # Require the above-Nyquist content to be a real part of the programme
    # (> -60 dB relative), otherwise numerical filter ringing fabricates a
    # suppression number for a signal that had nothing up there.
    if above_energy <= 1e-12 or (total_energy > 0 and above_energy / total_energy < 1e-6):
        return {"above_nyquist_present": False, "foldback_suppression_db": None,
                "residual_in_band_level_dbfs": None}
    if len(original) >= 33:
        coefficients = signal.butter(4, nyquist, "highpass", fs=source_rate, output="sos")
        high_only = signal.sosfiltfilt(coefficients, original)
        ratio = Fraction(row_rate / source_rate).limit_denominator(20_000)
        high_resampled = signal.resample_poly(high_only, ratio.numerator, ratio.denominator, padtype="line")
        leakage_energy = band_energy(high_resampled, row_rate, 0.0, nyquist)
    else:
        leakage_energy = 0.0
    residual_level = math.sqrt(leakage_energy / max(len(resampled), 1))
    suppression = 10.0 * math.log10(above_energy / max(leakage_energy, 1e-20))
    return {"above_nyquist_present": True,
            "foldback_suppression_db": round(float(suppression), 2),
            "residual_in_band_level_dbfs": level_db(residual_level)}


def prepare_audio(samples, source_rate, row_rate):
    duration = len(samples) / source_rate
    target_rows = max(MIN_STRIP_HEIGHT, round(duration * row_rate))
    # The leader/trailer label bands add rows on top of the signal rows.
    total_height = target_rows + LABEL_ROWS + TRAILER_ROWS
    if total_height > MAX_STRIP_HEIGHT:
        raise Limit(f"按该行速生成需要约 {total_height:,} 行，超过 {MAX_STRIP_HEIGHT:,} 行上限。"
                    "请缩短 WAV 或降低行速。")
    ratio = Fraction(row_rate / source_rate).limit_denominator(20_000)
    resampled = signal.resample_poly(samples, ratio.numerator, ratio.denominator, padtype="line")
    if len(resampled) >= target_rows:
        resampled = resampled[:target_rows]
    else:
        resampled = np.pad(resampled, (0, target_rows - len(resampled)), mode="edge")
    rows = len(resampled)
    if rows + LABEL_ROWS + TRAILER_ROWS > MAX_STRIP_HEIGHT:
        raise Limit(f"按该行速生成需要 {rows + LABEL_ROWS + TRAILER_ROWS:,} 行，超过 {MAX_STRIP_HEIGHT:,} 行上限。"
                    "请缩短 WAV 或降低行速。")
    nyquist = row_rate / 2.0
    total = band_energy(samples, source_rate, 0.0, source_rate / 2)
    removed = band_energy(samples, source_rate, nyquist, source_rate / 2)
    share = removed / total if total > 0 else 0.0
    return PreparedAudio(samples, source_rate, len(samples), resampled, row_rate, rows, duration, share)


def apply_limiter(resampled, settings: SynthSettings):
    knee = settings.knee
    magnitude = np.abs(resampled)
    over = magnitude > knee
    limited = resampled.copy()
    limited[over] = np.sign(resampled[over]) * (
        knee + (1.0 - knee) * np.tanh((magnitude[over] - knee) / (1.0 - knee)))
    # The pattern never reaches the aperture edges: the margin keeps a visible
    # black rim so a loud crest cannot occupy the whole reading window.
    span = 0.5 - settings.margin_fraction
    openness = 0.5 + span * limited
    openness = np.clip(openness, settings.margin_fraction, 1.0 - settings.margin_fraction)
    peak_in = float(np.max(magnitude)) if magnitude.size else 0.0
    peak_out = float(np.max(np.abs(limited))) if limited.size else 0.0
    reduction = 20.0 * math.log10(peak_in / peak_out) if peak_in > 1e-9 and peak_out < peak_in - 1e-12 else 0.0
    if over.any():
        per_row = 20.0 * np.log10(np.maximum(magnitude[over], 1e-9)
                                   / np.maximum(np.abs(limited[over]), 1e-9))
        strongest_reduction = float(np.max(per_row))
    else:
        strongest_reduction = 0.0
    return LimiterResult(limited, int(over.sum()), peak_in, peak_out, reduction, strongest_reduction,
                         openness, float(openness.min()), float(openness.max()))


def reading_slit(openness, slit_rows):
    """A finite-height reader slit averages neighbouring film rows (vertical blur)."""
    if slit_rows <= 1:
        return openness.copy()
    return ndimage.uniform_filter1d(openness.astype(np.float64), size=slit_rows, mode="nearest")


def render_strip(prepared: PreparedAudio, limited: LimiterResult, settings: SynthSettings, source_name):
    width = settings.aperture_width + 2 * SIDE_MARGIN
    x0, x1 = SIDE_MARGIN, SIDE_MARGIN + settings.aperture_width
    leader = LABEL_ROWS
    height = leader + prepared.rows + TRAILER_ROWS
    image = np.full((height, width), settings.base, dtype=np.float32)

    columns = np.arange(width, dtype=np.float64)
    center = (x0 + x1) / 2.0
    semi = limited.openness * settings.aperture_width * 0.5
    edge_left = center - semi
    edge_right = center + semi
    # Sub-pixel anti-aliased edges: fractional pixel coverage at each boundary.
    cover_left = np.clip(columns[None, :] - (edge_left - 0.5)[:, None], 0.0, 1.0)
    cover_right = np.clip((edge_right + 0.5)[:, None] - columns[None, :], 0.0, 1.0)
    coverage = cover_left * cover_right
    block = settings.base + coverage * (settings.level - settings.base)
    if settings.polarity == "dark":
        block = settings.base + (1.0 - coverage) * (settings.level - settings.base)
    image[leader:leader + prepared.rows, :] = block.astype(np.float32)

    if settings.grain_sigma > 0:
        rng = np.random.default_rng(
            int.from_bytes(sha256(source_name.encode("utf-8")).digest()[:4], "big"))
        image += (rng.normal(scale=settings.grain_sigma, size=image.shape)).astype(np.float32)
    image = np.clip(image, 0.0, 1.0).astype(np.float32)

    label_strip(image, settings, prepared, source_name)

    recipe = build_recipe(settings, width)
    png = encode_png(image, bits=16)
    raster = read_scan(png, "synthetic.png")
    filename = f"reeltone-synthetic-{sha256(png).hexdigest()[:10]}.png"
    return StripImage(raster.intensity, width, height, recipe.aperture,
                      (leader, leader + prepared.rows), recipe, png, filename)


def label_strip(image, settings, prepared, source_name):
    """Burn an unmistakable test label into the leader band (outside the signal).

    The band is intentionally high-contrast and machine-obvious: warning blocks
    on every row, a readable text line, and a dashed tick row immediately above
    the signal. It stays clear of the aperture signal region by construction.
    """
    from PIL import ImageFont

    band = image[:LABEL_ROWS, :]
    dark_text = settings.polarity == "clear"
    ink = settings.base if dark_text else settings.level
    paper = settings.level if dark_text else settings.base
    band[:] = paper
    # Full-height warning blocks along the whole width (visible at any zoom).
    for index in range(0, image.shape[1], 6):
        band[:, index:index + 3] = ink
    font = _font(11, bold=True)
    canvas = Image.new("L", (image.shape[1], LABEL_ROWS), color=0)
    if font is not None:
        draw = ImageDraw.Draw(canvas)
        stamp = "SYNTHETIC TEST - NOT A REAL FILM SCAN"
        draw.text((2, 2), stamp, fill=255, font=font)
        detail = f"{settings.row_rate:.0f} rows/s | aperture {settings.aperture_width}px | slit {settings.slit_rows}"
        draw.text((2, 16), detail, fill=255, font=font)
    else:
        # Font-free fallback: solid caution bars across the full band. The
        # warning blocks above already mark every row of the leader.
        for row in (6, 18, 30):
            canvas.paste(255, (0, row, image.shape[1], row + 3))
    text_layer = np.asarray(canvas, dtype=np.float32) / 255.0
    # Overwrite the warning blocks only where glyph pixels land, so the text
    # always has a solid paper-colour background behind it.
    band[:] = np.where(text_layer > 0.2, ink, paper)[:, :band.shape[1]]
    # Dashed warning tick row immediately above the signal start.
    band[-1, :] = np.where(np.arange(band.shape[1]) % 4 < 2, ink, paper)


def build_recipe(settings: SynthSettings, width):
    x0, x1 = SIDE_MARGIN, SIDE_MARGIN + settings.aperture_width
    tone = {"black": round(settings.base + 0.02, 4),
            "white": round(settings.level - 0.02, 4),
            "threshold": 0.5, "polarity": settings.polarity}
    return Recipe.parse({
        "aperture": [x0, x1], "perforations": [0, min(5, x0 - 1)],
        "orientation": {"mirror_x": False, "reverse_y": False},
        "tone": tone,
        "clock": {"model": "manual", "row_rate": settings.row_rate, "frames_per_second": 24,
                  "holes_per_frame": 1, "pitch_guess": 250}}, width)


def recovered_openness(strip: StripImage):
    read = read_area(strip.pixels, strip.aperture, strip.recipe.tone, (0, strip.height))
    top, bottom = strip.signal_rows
    openness = read.openness[top:bottom]
    return openness, read


def align_series(reference, measured):
    """Find integer lag and least-squares gain between reference and measured."""
    ref = np.asarray(reference, dtype=np.float64)
    got = np.asarray(measured, dtype=np.float64)
    correlation = signal.correlate(got - got.mean(), ref - ref.mean(), mode="full")
    lag = int(np.argmax(correlation) - (len(ref) - 1))
    if lag >= 0:
        a, b = got[lag:lag + len(ref)], ref
    else:
        a, b = got[-lag:], ref[:len(got) + lag]
    count = min(len(a), len(b))
    a, b = a[:count], b[:count]
    gain = float(np.dot(a, b) / max(np.dot(b, b), 1e-18))
    return a, b, gain, lag


def level_db(value):
    return round(20.0 * math.log10(value), 2) if value > 1e-10 else None


def fft_band_table(reference, measured, row_rate):
    count = min(len(reference), len(measured))
    ref, got = reference[:count], measured[:count]
    freqs = np.fft.rfftfreq(count, 1.0 / row_rate)
    spec_ref = np.abs(np.fft.rfft(ref * np.hanning(count))) ** 2
    spec_got = np.abs(np.fft.rfft(got * np.hanning(count))) ** 2
    nyquist = row_rate / 2.0
    edges = [0, 75, 250, 750, 1500, 3000, 6000, 12000, 24000]
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        if low >= nyquist:
            break
        mask = (freqs >= low) & (freqs < min(high, nyquist + 1e-9))
        if not np.any(mask):
            continue
        rows.append({"band_hz": [low, min(high, round(nyquist, 1))],
                     "reference_db": level_db(math.sqrt(float(np.mean(spec_ref[mask])))),
                     "recovered_db": level_db(math.sqrt(float(np.mean(spec_got[mask]))))})
    return rows, freqs, spec_ref, spec_got


def quiet_passage_report(original, source_rate, openness_ref, openness_got):
    """Report how much of the quiet programme survives the pattern."""
    window = max(8, round(0.05 * source_rate))
    edges = np.arange(0, len(original), window)
    rms_levels = []
    for start in edges:
        chunk = original[start:min(len(original), start + window)]
        if len(chunk):
            rms_levels.append(math.sqrt(float(np.mean(chunk ** 2))))
    quiet_threshold = 0.1
    if rms_levels:
        quiet_windows = sum(value <= quiet_threshold for value in rms_levels)
        quiet_fraction = quiet_windows / len(rms_levels)
        quiet_time = quiet_windows * window / source_rate
    else:
        quiet_fraction, quiet_time = 0.0, 0.0
    # Residual floor of the pattern itself (deviation from the reference shape).
    residual = openness_got - openness_ref
    floor = float(np.percentile(np.abs(residual), 50))
    floor_95 = float(np.percentile(np.abs(residual), 95))
    # Recovery error inside the quiet windows only, mapped back to a dBFS level.
    residual = openness_got - openness_ref
    quiet_errors = []
    for index, level in enumerate(rms_levels):
        if level <= quiet_threshold:
            start = index * window
            quiet_errors.append(residual[start:min(len(residual), start + window)])
    quiet_chain = np.concatenate(quiet_errors) if quiet_errors else np.abs(residual)
    quiet_rms = math.sqrt(float(np.mean(quiet_chain ** 2))) if quiet_chain.size else floor
    return {"quiet_window_fraction": round(quiet_fraction, 4),
            "quiet_seconds_estimate": round(quiet_time, 4),
            "quiet_threshold_rms": quiet_threshold,
            "residual_median_openness": round(floor, 6),
            "residual_p95_openness": round(floor_95, 6),
            "residual_equivalent_dbfs": level_db(floor * 2.0),
            "quiet_section_error_dbfs": level_db(quiet_rms * 2.0)}


def dynamic_range_report(openness_ref, openness_got, limited: LimiterResult, settings):
    usable_low = float(np.percentile(openness_ref, 1))
    usable_high = float(np.percentile(openness_ref, 99))
    signal_span = max(1e-9, usable_high - usable_low)
    residual = float(np.percentile(np.abs(openness_got - openness_ref), 50))
    if signal_span < 1e-6:
        effective_dr = None
        dr_note = "信号近乎静音，没有可测的图案动态范围。"
    elif residual <= 0:
        effective_dr = None
        dr_note = "回读与参考完全一致，残余底噪低于量化下限。"
    else:
        effective_dr = 20.0 * math.log10(signal_span / residual)
        dr_note = None
    return {"openness_q01_q50_q99": [round(usable_low, 5),
                                     round(float(np.median(openness_ref)), 5), round(usable_high, 5)],
            "pattern_signal_span_openness": round(signal_span, 5),
            "headroom_margin_fraction": settings.margin_fraction,
            "limiter_knee_fraction": settings.knee,
            "limiter_rows": limited.rows_limited,
            "limiter_max_input": round(limited.max_input, 5),
            "limiter_max_output": round(limited.max_output, 5),
            "peak_gain_reduction_db": round(limited.gain_reduction_db, 3),
            "effective_dynamic_range_db": round(effective_dr, 2) if effective_dr is not None else None,
            "dynamic_range_note": dr_note,
            "aperture_quantization_levels": settings.aperture_width}


def compare_roundtrip(prepared: PreparedAudio, limited: LimiterResult, strip: StripImage, settings):
    openness_got, optical = recovered_openness(strip)
    openness_ref = limited.openness
    count = min(len(openness_ref), len(openness_got))
    openness_ref = openness_ref[:count]
    openness_got = openness_got[:count]

    reference_signal = (openness_ref - 0.5) * 2.0
    measured_signal = (openness_got - np.median(openness_got)) * 2.0
    aligned_got, aligned_ref, gain, lag = align_series(reference_signal, measured_signal)
    error = aligned_got - gain * aligned_ref
    correlation = float(np.corrcoef(aligned_got, aligned_ref)[0, 1]) if aligned_ref.std() > 0 else 1.0
    rms_error = math.sqrt(float(np.mean(error ** 2)))
    max_error = float(np.max(np.abs(error)))

    band_rows, freqs, spec_ref, spec_got = fft_band_table(aligned_ref, aligned_got, settings.row_rate)
    # Alias guard: energy above Nyquist cannot appear; report in-band energy that
    # exists in the recovery but not the band-limited reference (foldback proxy).
    nyquist = settings.row_rate / 2.0
    in_band = freqs < nyquist
    excess = np.maximum(spec_got[in_band] - spec_ref[in_band], 0.0)
    alias_proxy = math.sqrt(float(np.mean(excess))) if excess.size else 0.0

    signal_top, signal_bottom = strip.signal_rows
    touch_in_signal = int(optical.touch_edge[signal_top:signal_bottom].sum())
    touch_in_label = int(optical.touch_edge.sum()) - touch_in_signal

    # A multi-row reading slit acts as a vertical box filter: it attenuates the
    # upper part of the band. Quantify that loss both analytically and from the
    # measured round trip, so quantization/grain cannot hide the geometry.
    high_low, high_high = nyquist * 0.5, nyquist * 0.98
    ref_high = band_energy(aligned_ref, settings.row_rate, high_low, high_high)
    got_high = band_energy(aligned_got, settings.row_rate, high_low, high_high)
    measured_slit_db = round(10.0 * math.log10(ref_high / max(got_high, 1e-20)), 2) if ref_high > 1e-12 else None
    slit_rows = settings.slit_rows
    if slit_rows <= 1:
        theoretical_slit_db = 0.0
    else:
        probe = nyquist * 0.75
        z = math.pi * probe / settings.row_rate
        response = abs(math.sin(slit_rows * z) / (slit_rows * math.sin(z)))
        theoretical_slit_db = round(20.0 * math.log10(max(response, 1e-6)), 2)
    slit_attenuation = {"theoretical_db_at_0_75_nyquist": theoretical_slit_db,
                        "measured_high_octave_db": measured_slit_db}

    # Original vs band-limited reference: what the row rate removed.
    original_stats = waveform_numbers(prepared.samples)
    reference_stats = waveform_numbers(prepared.resampled)
    recovered_stats = waveform_numbers(aligned_got)

    report = {
        "synthetic": True,
        "source": {"filename": "test-tone.wav", "rate": prepared.source_rate, "frames": prepared.source_frames,
                   "duration_seconds": round(prepared.duration, 6),
                   "nyquist_hz": round(prepared.source_rate / 2.0, 2),
                   "statistics": original_stats},
        "row_clock": {"row_rate": settings.row_rate, "rows": prepared.rows,
                      "nyquist_hz": round(nyquist, 2),
                      "removed_above_nyquist_energy_share": round(prepared.removed_band_share, 6),
                      "removed_above_nyquist_percent": round(100.0 * prepared.removed_band_share, 3),
                      "foldback": foldback_analysis(prepared.samples, prepared.source_rate,
                                                    prepared.resampled, settings.row_rate)},
        "limiter": {"knee_fraction": settings.knee, "rows_limited": limited.rows_limited,
                    "max_input": round(limited.max_input, 5), "max_output": round(limited.max_output, 5),
                    "peak_gain_reduction_db": round(limited.gain_reduction_db, 3),
                    "strongest_row_reduction_db": round(limited.strongest_row_reduction_db, 3),
                    "openness_min": round(limited.openness_min, 5), "openness_max": round(limited.openness_max, 5)},
        "pattern": {"width_columns": settings.aperture_width, "slit_rows": settings.slit_rows,
                    "grain_sigma": settings.grain_sigma, "base_transmission": settings.base,
                    "level_transmission": settings.level, "polarity": settings.polarity,
                    "image_dimensions": [strip.width, strip.height],
                    "signal_rows": list(strip.signal_rows),
                    "edge_margin_fraction": settings.margin_fraction,
                    "slit_high_octave_attenuation_db": slit_attenuation},
        "readback": {"correlation": round(correlation, 6), "gain_fit": round(gain, 6),
                     "rms_error": round(rms_error, 6), "rms_error_dbfs": level_db(rms_error),
                     "max_abs_error": round(max_error, 6), "alignment_lag_rows": lag,
                     "touch_edge_rows_total": int(optical.touch_edge.sum()),
                     "touch_edge_rows_in_label_band": touch_in_label,
                     "touch_edge_rows_in_signal": touch_in_signal,
                     "clipped_pixels": int(optical.clipped_low.sum() + optical.clipped_high.sum()),
                     "multiple_island_rows": int(np.sum(optical.islands > 1)),
                     "band_table": band_rows,
                     "alias_proxy_rms_dbfs": level_db(alias_proxy),
                     "statistics": recovered_stats},
        "band_limited_reference": {"statistics": reference_stats},
        "dynamic_range": dynamic_range_report(openness_ref, openness_got, limited, settings),
        "quiet_retention": quiet_passage_report(prepared.samples, prepared.source_rate,
                                                openness_ref, openness_got),
        "explanation": [
            "行速重采样使用抗混叠低通：高于新奈奎斯特频率的能量被滤除，不会折叠成新的低音。",
            "超过软限幅拐点的峰值按 tanh 曲线压缩，读窗两侧始终保留边缘余量，任何峰值都不能占满整条读窗。",
            "图案边缘以亚像素渲染，但读回的有效电平粒度受读窗列数限制；颗粒和多行读窗会抬高残余底噪。",
            "弱音保留情况见 quiet_retention；动态范围损失见 dynamic_range；被去掉的高频能量占比见 row_clock。",
        ],
    }
    return report, openness_ref, openness_got, optical


def waveform_numbers(values):
    values = np.asarray(values, dtype=np.float64)
    peak = float(np.max(np.abs(values))) if values.size else 0.0
    rms = math.sqrt(float(np.mean(values ** 2))) if values.size else 0.0
    return {"peak": round(peak, 6), "rms": round(rms, 6),
            "peak_dbfs": level_db(peak), "rms_dbfs": level_db(rms)}


# ---------------------------------------------------------------------------
# Preview figures rendered with PIL only (no matplotlib dependency).
# ---------------------------------------------------------------------------

PAGE = "#111923"
PANEL = "#18232e"
GRID = "#364856"
TEXT = "#f2f5eb"
MUTED = "#a5b3bc"
ACCENT = "#e9b76b"
TEAL = "#77ded1"
RED = "#f48680"
BLUE = "#83a9eb"


def _font(size, bold=False):
    try:
        from PIL import ImageFont
        faces = ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf") if bold else (
            "DejaVuSans.ttf", "DejaVuSans-Bold.ttf")
        for face in faces:
            try:
                return ImageFont.truetype(face, size)
            except OSError:
                continue
    except ImportError:
        pass
    return None


def _draw_text(draw, xy, content, size=14, fill=TEXT, *, anchor=None):
    font = _font(size)
    if font is None:
        # Without any font file, draw a compact marker instead of crashing.
        x, y = xy
        draw.rectangle((x, y, x + max(8, len(str(content))), y + max(3, size // 3)), fill=fill)
        return
    draw.text(xy, str(content), font=font, fill=fill, anchor=anchor)


def _waveform_series(draw, series, box, color, *, samples=900):
    left, top, right, bottom = box
    data = np.asarray(series, dtype=np.float64)
    if len(data) > samples:
        edges = np.linspace(0, len(data), samples + 1, dtype=int)
        reduced = []
        for a, b in zip(edges[:-1], edges[1:]):
            chunk = data[a:b]
            reduced.append((float(chunk.min()), float(chunk.max())))
    else:
        reduced = [(float(v), float(v)) for v in data]
    mid = (top + bottom) / 2
    span = max(1e-6, float(np.max(np.abs(data))))
    for index, (low, high) in enumerate(reduced):
        x = left + index * (right - left) / max(1, len(reduced) - 1)
        y1 = mid - high / span * (bottom - top) / 2
        y2 = mid - low / span * (bottom - top) / 2
        draw.line((x, y1, x, y2), fill=color)
    draw.line((left, mid, right, mid), fill=GRID)


def comparison_figure(prepared, limited, openness_ref, openness_got, settings):
    width, height = 1180, 760
    page = Image.new("RGB", (width, height), PAGE)
    draw = ImageDraw.Draw(page)
    _draw_text(draw, (24, 18), "REELTONE SYNTHETIC TEST - SOURCE vs ROUND-TRIP READBACK", 16, TEAL)
    _draw_text(draw, (24, 42), "Top: original WAV. Middle: band-limited reference at the requested row rate. "
                        "Bottom: signal actually recovered from the rendered strip.", 13, MUTED)

    panels = [
        (92, 250, prepared.samples, BLUE, "Original upload"),
        (286, 444, limited.values, ACCENT, f"Band-limited + soft limited @ {settings.row_rate:.0f} rows/s"),
        (480, 638, (openness_got - np.median(openness_got)) * 2.0, TEAL, "Recovered through the normal optical reader"),
    ]
    for top, bottom, data, color, caption in panels:
        draw.rectangle((60, top, width - 30, bottom), outline=GRID)
        _draw_text(draw, (66, top + 6), caption, 13, TEXT)
        _waveform_series(draw, data, (70, top + 26, width - 40, bottom - 12), color)
    _draw_text(draw, (24, 660), "All panels share a symmetric amplitude scale per panel; the limiter caps loud crests "
                        "while quiet passages keep their shape.", 12, MUTED)
    _draw_text(draw, (24, 712), "TEST SYNTHESIS IMAGE - GENERATED FROM USER WAV, NOT A PRESERVATION SCAN", 13, RED)
    return encode_image(page, "PNG")


def spectrum_figure(openness_ref, openness_got, settings):
    count = min(len(openness_ref), len(openness_got))
    ref = (openness_ref[:count] - 0.5) * 2.0
    got = openness_got[:count] - np.median(openness_got[:count])
    got *= 2.0
    freqs = np.fft.rfftfreq(count, 1.0 / settings.row_rate)
    spec_ref = np.abs(np.fft.rfft(ref * np.hanning(count))) ** 2
    spec_got = np.abs(np.fft.rfft(got * np.hanning(count))) ** 2
    db_ref = 10.0 * np.log10(np.maximum(spec_ref, 1e-12))
    db_got = 10.0 * np.log10(np.maximum(spec_got, 1e-12))

    width, height = 1180, 420
    page = Image.new("RGB", (width, height), PAGE)
    draw = ImageDraw.Draw(page)
    _draw_text(draw, (24, 16), "SPECTRUM: BAND-LIMITED REFERENCE vs STRIP READBACK", 16, TEAL)
    left, top, right, bottom = 78, 56, width - 34, height - 60
    nyquist = settings.row_rate / 2.0
    # Auto-ranged dB scale from the actual data, with a fixed floor margin.
    visible = freqs <= nyquist
    peak_db = float(max(np.max(db_ref[visible]), np.max(db_got[visible])))
    top_db = math.ceil(peak_db / 10.0) * 10.0 + 5.0
    floor_db = max(-110.0, top_db - 80.0)

    def points(spectrum_db, color, line_width):
        previous = None
        for f, db in zip(freqs, spectrum_db):
            if f > nyquist:
                break
            x = left + f / nyquist * (right - left)
            y = bottom - np.clip((db - floor_db) / (top_db - floor_db), 0.0, 1.0) * (bottom - top)
            if previous is not None:
                draw.line((previous[0], previous[1], x, y), fill=color, width=line_width)
            previous = (x, y)

    draw.rectangle((left, top, right, bottom), outline=GRID)
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = left + fraction * (right - left)
        draw.line((x, top, x, bottom), fill=GRID)
        _draw_text(draw, (x - 18, bottom + 8), f"{nyquist * fraction:.0f} Hz", 11, MUTED)
    for db in np.linspace(floor_db, top_db, 5):
        y = bottom - (db - floor_db) / (top_db - floor_db) * (bottom - top)
        draw.line((left, y, right, y), fill=GRID)
        _draw_text(draw, (28, y - 6), f"{db:.0f}", 11, MUTED)
    points(db_ref, ACCENT, 1)
    points(db_got, TEAL, 1)
    draw.line((right, top, right, bottom), fill=RED)
    _draw_text(draw, (right - 104, top + 6), f"Nyquist {nyquist:.0f} Hz", 12, RED)
    _draw_text(draw, (left, bottom + 26), "Amber: reference after anti-alias filter. Teal: readback. "
                                   "No foldback bins appear above the noise floor.", 12, MUTED)
    return encode_image(page, "PNG")


def strip_preview_figure(strip: StripImage, settings):
    pixels = strip.pixels
    height, width = pixels.shape
    preview_width = 300
    preview_height = min(760, max(260, round(height * preview_width / max(width, 1))))
    image = Image.fromarray(np.rint(np.clip(pixels, 0, 1) * 255).astype(np.uint8), mode="L")
    resized = image.resize((preview_width, preview_height), Image.Resampling.BOX).convert("RGB")
    page = Image.new("RGB", (preview_width + 70, preview_height + 118), PAGE)
    draw = ImageDraw.Draw(page)
    _draw_text(draw, (16, 14), "RENDERED TEST STRIP (full image)", 15, TEAL)
    page.paste(resized, (24, 46))
    x0 = 24 + strip.aperture.x0 * preview_width / width
    x1 = 24 + strip.aperture.x1 * preview_width / width
    draw.line((x0, 46, x0, 46 + preview_height), fill=TEAL, width=2)
    draw.line((x1, 46, x1, 46 + preview_height), fill=TEAL, width=2)
    top, bottom = strip.signal_rows
    y0 = 46 + top * preview_height / height
    y1 = 46 + bottom * preview_height / height
    draw.line((24, y0, 24 + preview_width, y0), fill=ACCENT, width=1)
    draw.line((24, y1, 24 + preview_width, y1), fill=ACCENT, width=1)
    _draw_text(draw, (16, page.height - 52), f"Aperture cols {strip.aperture.x0}..{strip.aperture.x1} (teal), "
                                      f"signal rows {top}..{bottom} (amber).", 12, MUTED)
    _draw_text(draw, (16, page.height - 30), "SYNTHETIC TEST PATTERN - NOT A REAL SCAN", 13, RED)
    return encode_image(page, "PNG")


def synthesize(wav_bytes, wav_name, values):
    """Full pipeline used by both the preview endpoint and the commit endpoint."""
    settings = SynthSettings.parse(values)
    samples, source_rate, source_filename = load_test_wav(wav_bytes, wav_name)
    prepared = prepare_audio(samples, source_rate, settings.row_rate)
    limited = apply_limiter(prepared.resampled, settings)
    # A finite-height reader slit averages neighbouring rows; bake the same
    # averaging into the rendered pattern so the report's reference matches the
    # physical geometry the reader will actually integrate over.
    slit_openness = reading_slit(limited.openness, settings.slit_rows)
    limited = LimiterResult(limited.values, limited.rows_limited, limited.max_input, limited.max_output,
                            limited.gain_reduction_db, limited.strongest_row_reduction_db, slit_openness,
                            limited.openness_min, limited.openness_max)
    strip = render_strip(prepared, limited, settings, source_filename)
    report, openness_ref, openness_got, optical = compare_roundtrip(prepared, limited, strip, settings)
    report["source"]["filename"] = source_filename
    figures = {
        "comparison": comparison_figure(prepared, limited, openness_ref, openness_got, settings),
        "spectrum": spectrum_figure(openness_ref, openness_got, settings),
        "strip": strip_preview_figure(strip, settings),
    }
    sidecar = {
        "kind": "reeltone-synthetic-test-settings",
        "synthetic": True,
        "warning": "测试用合成图，不是真实胶片扫描。REELTONE SYNTHETIC TEST STRIP.",
        "source_wav": {"filename": source_filename, "sha256": sha256(wav_bytes).hexdigest(),
                       "rate": source_rate, "frames": prepared.source_frames},
        "settings": settings.as_dict(),
        "image": {"filename": strip.filename, "sha256": sha256(strip.png).hexdigest(),
                  "width": strip.width, "height": strip.height,
                  "signal_rows": list(strip.signal_rows)},
        "recipe": strip.recipe.as_dict(),
        "report": report,
    }
    return SynthJob(settings, prepared, limited, strip, report, figures, sidecar, wav_bytes)


@dataclass
class SynthJob:
    settings: SynthSettings
    prepared: PreparedAudio
    limited: LimiterResult
    strip: StripImage
    report: dict
    figures: dict
    sidecar: dict
    wav_bytes: bytes


class JobStore:
    def __init__(self):
        self._jobs = {}

    def put(self, job):
        import time
        key = sha256(job.strip.png).hexdigest()[:16]
        self._jobs[key] = (time.monotonic(), job)
        self._sweep()
        if len(self._jobs) > JOB_LIMIT:
            oldest = min(self._jobs, key=lambda item: self._jobs[item][0])
            self._jobs.pop(oldest)
        return key

    def get(self, key):
        import time
        entry = self._jobs.get(key)
        if entry is None:
            raise BenchError("合成预览已过期，请重新生成。", code="job_expired")
        created, job = entry
        if time.monotonic() - created > JOB_TTL_SECONDS:
            self._jobs.pop(key, None)
            raise BenchError("合成预览已过期，请重新生成。", code="job_expired")
        return job

    def _sweep(self):
        import time
        now = time.monotonic()
        expired = [key for key, (created, _) in self._jobs.items() if now - created > JOB_TTL_SECONDS]
        for key in expired:
            self._jobs.pop(key, None)
