"""Gallery redaction: native-row masks that replace both picture and sound.

The masks never remove rows.  Each held span keeps its place on the film, so
row numbers and playback time stay aligned with the untouched source scan.
"""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .audition import (
    Monitor, apply_fade, audition, normalize_signal, shaping_filter, waveform_envelope,
)
from .contracts import integer, mapping, sequence, split_rows, text
from .errors import BenchError
from .optical import read_area
from .raster import encode_png

MAX_SPANS = 20
MIN_FEATHER = 0
MAX_FEATHER = 600


@dataclass(frozen=True)
class HeldSpan:
    y0: int
    y1: int
    label: str
    note: str


def parse_spans(raw, height):
    entries = sequence(mapping(raw, "遮盖区间请求").get("spans", []), "遮盖区间", minimum=1, maximum=MAX_SPANS)
    spans = []
    for item in entries:
        data = mapping(item, "遮盖区间")
        y0, y1 = split_rows(data.get("rows"), height, min_span=1)
        label = text(data.get("label", f"遮盖 {len(spans) + 1}"), "遮盖标签", maximum=90)
        note = text(data.get("note", ""), "遮盖说明", maximum=400, required=False, multiline=True)
        spans.append(HeldSpan(y0, y1, label, note))
    spans.sort(key=lambda span: span.y0)
    for earlier, later in zip(spans, spans[1:]):
        if later.y0 < earlier.y1:
            raise BenchError(f"遮盖区间重叠：{earlier.y0}–{earlier.y1} 与 {later.y0}–{later.y1}。请先合并相邻区间。")
    held = sum(span.y1 - span.y0 for span in spans)
    if held >= height - 32:
        raise BenchError("遮盖范围不能覆盖整张扫描，至少保留 32 个原片行用于衔接基准和行号核对。")
    return tuple(spans)


def row_masks(height, spans, feather):
    hard = np.zeros(height, dtype=bool)
    for span in spans:
        hard[span.y0:span.y1] = True
    soft = hard.copy()
    if feather:
        controlled = np.zeros(height, dtype=bool)
        for span in spans:
            controlled[max(0, span.y0 - feather):min(height, span.y1 + feather)] = True
        soft = controlled
    return hard, soft


def _span_profile(height, span, feather):
    """Raised-cosine profile for one span on the full row axis."""
    profile = np.ones(height, dtype=np.float64)
    lo = max(0, span.y0 - feather)
    hi = min(height, span.y1 + feather)
    ramp_in = 0.5 * (1.0 - np.cos(np.pi * np.arange(span.y0 - lo) / feather))
    ramp_out = 0.5 * (1.0 + np.cos(np.pi * np.arange(hi - span.y1) / feather))
    profile[lo:span.y0] = ramp_in
    profile[span.y0:span.y1] = 0.0
    profile[span.y1:hi] = ramp_out
    return profile


def redaction_envelope(height, spans, feather):
    """1.0 outside, 0.0 on held rows, smooth raised-cosine ramps at the edges."""
    hard, soft = row_masks(height, spans, feather)
    if not feather:
        weight = np.ones(height, dtype=np.float64)
        weight[hard] = 0.0
        return weight, hard, soft
    # Combine full-axis profiles with a per-row minimum so overlapping feather
    # bands blend into one continuous dip instead of overwriting each other.
    weight = np.minimum.reduce([_span_profile(height, span, feather) for span in spans])
    weight[hard] = 0.0
    return weight, hard, soft


def column_baseline(pixels, controlled):
    """Median of the never-controlled rows per column, smoothed across columns."""
    height, width = pixels.shape
    if controlled.all():
        raise BenchError("遮盖范围不能覆盖整张扫描，至少保留一些原片行用于衔接基准。")
    open_rows = np.flatnonzero(~controlled)
    if len(open_rows) > 9000:
        open_rows = open_rows[np.linspace(0, len(open_rows) - 1, 9000).astype(int)]
    baseline = np.median(pixels[open_rows, :], axis=0)
    return ndimage.uniform_filter1d(baseline, size=max(3, width // 12), mode="nearest")


def redacted_pixels(pixels, spans, feather):
    height = pixels.shape[0]
    weight, hard, controlled = redaction_envelope(height, spans, feather)
    baseline = column_baseline(pixels, controlled)
    blended = pixels.astype(np.float64) * weight[:, None] + baseline[None, :] * (1.0 - weight[:, None])
    redacted = np.asarray(np.clip(blended, 0.0, 1.0), dtype=pixels.dtype)
    # Hard guarantee: every held row is synthetic baseline, none of the source remains.
    redacted[hard] = baseline[None, :]
    return redacted, weight, hard, controlled


def render_gallery(material, spans, feather, monitor_values):
    settings = Monitor.parse(monitor_values)
    pixels, weight, hard, controlled = redacted_pixels(material.pixels, spans, feather)
    clock, _ = material.clock()
    rate = settings.rate
    # Audio uses the unredacted optical reading, so the standard filter never
    # sees the synthetic strip.  Redaction is applied only after time stretching.
    plain_reading = read_area(material.pixels, material.recipe.aperture, material.recipe.tone,
                              (0, material.height))
    scan, center = normalize_signal(plain_reading, settings, clock)
    filtered, filter_warnings = shaping_filter(scan, clock.rows_per_second, settings)
    # Phase-preserving time stretch rather than a global FIR resampler: the
    # polyphase filter would smear held-block energy into distant samples.
    stretched = _stretch_rows(filtered, clock.rows_per_second, rate)
    # Raised-cosine envelope rebuilt on output samples guarantees the join lands
    # on an exact zero at each held block without a step on either boundary.
    output_weight, held_mask = _output_envelope(len(stretched), hard, clock, rate, feather)
    safe_wave = stretched * output_weight * 10.0 ** (settings.gain_db / 20.0)
    safe_wave[held_mask] = 0.0
    n_ramp = min(len(safe_wave) // 2, round(rate * settings.ramp_ms / 1000.0))
    safe_wave = apply_fade(safe_wave, n_ramp)
    held_edges = _mask_edges(held_mask)
    reference = audition(plain_reading, clock, settings, allow_long=True)
    report = {"span_count": len(spans), "feather_rows": feather, "held_rows": int(hard.sum()),
              "controlled_rows": int(controlled.sum()), "output_frames": len(safe_wave),
              "held_sample_edges": held_edges, "max_level_on_held": _held_level(safe_wave, held_mask),
              "max_envelope_at_boundary": _edge_envelope(weight, hard)}
    image_png = encode_png(pixels, bits=8)
    return safe_wave, rate, image_png, report, reference, waveform_envelope(reference)


def _output_envelope(frame_count, hard_rows, clock, rate, feather):
    """Raised-cosine blend on output samples; held rows map to an exact zero.

    Each block uses one continuous cosine lobe per side whose final point is
    exactly zero on the first/last held sample, so no step can form where the
    feather ramp meets the hard block.
    """
    times = np.arange(frame_count) / rate
    native = np.clip(np.rint(clock.native_center_at_time(times)).astype(int), 0, len(hard_rows) - 1)
    held = hard_rows[native]
    envelope = np.ones(frame_count, dtype=np.float64)
    feather_frames = max(1, int(round(feather * rate / clock.rows_per_second)))
    for y0, y1 in _row_edges(hard_rows):
        t0, t1 = clock.extent((y0, y1))
        entry = int(round(t0 * rate))
        exit_ = int(round(t1 * rate))
        left = max(0, entry - feather_frames)
        right = min(frame_count, exit_ + feather_frames)
        # Entry: 1 at ``left`` declining continuously to 0 at ``entry``.
        entry_len = entry - left
        if entry_len:
            phase = np.arange(entry_len, 0, -1) / entry_len
            envelope[left:entry] = np.minimum(envelope[left:entry],
                                              0.5 * (1.0 - np.cos(np.pi * phase)))
        # Exit: 0 at ``exit_`` rising continuously to 1 at ``right``.
        exit_len = right - exit_
        if exit_len:
            phase = np.arange(1, exit_len + 1) / exit_len
            envelope[exit_:right] = np.minimum(envelope[exit_:right],
                                               0.5 * (1.0 - np.cos(np.pi * phase)))
        envelope[entry:exit_] = 0.0
    envelope[held] = 0.0
    return envelope, held


def _row_edges(hard_rows):
    rows = np.flatnonzero(hard_rows)
    breaks = np.flatnonzero(np.diff(rows) > 1)
    starts = np.concatenate(([rows[0]], rows[breaks + 1]))
    stops = np.concatenate((rows[breaks], [rows[-1]])) + 1
    return list(zip(starts.tolist(), stops.tolist()))


def _mask_edges(mask):
    breaks = np.flatnonzero(np.diff(mask.view(np.int8), prepend=False, append=False))
    return [[int(a), int(b)] for a, b in zip(breaks[::2], breaks[1::2]) if mask[a]]


def _stretch_rows(values, row_rate, output_rate):
    """Map native rows on to the output clock with linear interpolation.

    Unlike polyphase resampling this has no filter memory, so replacing a row
    range cannot move energy outside that range's playback window.  The output
    length matches a conventional retime exactly, preserving total duration.
    """
    count = max(1, round(len(values) * output_rate / row_rate))
    positions = (np.arange(count) + 0.5) * row_rate / output_rate - 0.5
    positions = np.clip(positions, 0.0, len(values) - 1.0)
    lower = np.floor(positions).astype(int)
    upper = np.minimum(lower + 1, len(values) - 1)
    fraction = positions - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def _edge_envelope(weight, hard_rows):
    """Row-domain blend weight at the first/last controlled rows of each block."""
    indices = np.flatnonzero(hard_rows)
    if not len(indices):
        return 0.0
    worst = 0.0
    for edge in indices:
        for neighbor in (edge - 1, edge + 1):
            if 0 <= neighbor < len(weight) and not hard_rows[neighbor]:
                worst = max(worst, float(weight[neighbor]))
    return round(worst, 8)


def _held_level(wave, held_mask):
    if not held_mask.any():
        return 0.0
    return round(float(np.max(np.abs(wave[held_mask]))), 10)


def parse_feather(data):
    return integer(mapping(data, "遮盖请求").get("feather_rows", 24), "平顺衔接行数", MIN_FEATHER, MAX_FEATHER)
