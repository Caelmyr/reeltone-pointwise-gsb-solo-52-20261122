from dataclasses import dataclass

import numpy as np
from scipy import signal

from .audition import waveform_statistics
from .geometry import runs


@dataclass(frozen=True)
class Spectrum:
    hertz: np.ndarray
    seconds: np.ndarray
    power: np.ndarray
    band_limit: float
    source_rate: float
    monitor_rate: int
    window: int

    @property
    def bin_size(self):
        return self.monitor_rate / self.window

    def profile(self, maximum=320):
        measured = self.power.mean(axis=1)
        stride = max(1, int(np.ceil(len(measured) / maximum)))
        return [{"hz": round(float(self.hertz[i]), 3), "db": power_db(measured[i])}
                for i in range(0, len(measured), stride)]

    def bands(self):
        results = []
        boundaries = (0, 75, 250, 750, 1500, 3000, 6000, 12000, 24000)
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            admitted = self.hertz >= start
            admitted &= self.hertz < min(stop, self.band_limit + 0.001)
            if not np.any(admitted):
                continue
            selected = self.power[admitted]
            results.append({"hz": [start, min(stop, round(self.band_limit, 2))],
                            "mean_power_db": power_db(float(np.mean(np.sum(selected, axis=0)))),
                            "peak_power_db": power_db(float(np.max(np.sum(selected, axis=0))))})
        return results

    def summary(self):
        band = self.power.mean(axis=1)
        represented = self.hertz <= self.band_limit
        measure = (self.hertz >= self.bin_size * 1.25) & represented
        if measure.any():
            choices = np.flatnonzero(measure)
            strongest = float(self.hertz[choices[np.argmax(band[measure])]])
        else:
            strongest = None
        return {"resolution_hz": round(self.bin_size, 5), "window_samples": self.window,
                "native_nyquist_hz": round(self.source_rate / 2, 2), "inspected_band_hz": round(self.band_limit, 2),
                "monitor_rate": self.monitor_rate, "dominant_bin_hz": strongest,
                "bins": len(self.hertz), "time_windows": len(self.seconds), "bands": self.bands()}


def power_db(value, floor=-120.0):
    return round(max(floor, float(10.0 * np.log10(max(1e-18, value)))), 4)


def analyze_signal(values, rate, source_rate, window=384):
    count = min(len(values), window)
    count = max(8, count)
    prepared = values if len(values) >= count else np.pad(values, (0, count - len(values)))
    overlap = min(count - 1, max(1, 3 * count // 4))
    hz, secs, spectrum = signal.spectrogram(prepared, fs=rate, window="hann", nperseg=count,
                                           noverlap=overlap, detrend="constant", scaling="spectrum")
    band_limit = min(source_rate / 2, rate / 2)
    cap = max(2, min(len(hz), np.searchsorted(hz, band_limit, side="right")))
    return Spectrum(hz[:cap], secs, spectrum[:cap], band_limit, source_rate, rate, count)


def review_sections(values, rate, *, quiet_db=-42, sample_ms=50):
    count = max(4, round(sample_ms * rate / 1000))
    edges = np.arange(0, len(values), count, dtype=int)
    quiet = []
    loud = []
    sections = []
    for a in edges:
        b = min(len(values), a + count)
        level = waveform_statistics(values[a:b])
        quiet.append(level["rms_dbfs"] is None or level["rms_dbfs"] <= quiet_db)
        loud.append(level["peak"] >= 0.99)
        sections.append({"samples": [int(a), int(b)], "seconds": [round(a / rate, 6), round(b / rate, 6)],
                         "rms_dbfs": level["rms_dbfs"], "peak": level["peak"]})
    def group(mask, minimum):
        return [[sections[a]["seconds"][0], sections[b - 1]["seconds"][1]] for a, b in runs(mask, min_count=minimum)]
    return {"window_ms": sample_ms, "quiet_floor_dbfs": quiet_db, "quiet_seconds": group(quiet, 2),
            "limit_seconds": group(loud, 1), "sections": sections}


def narrowband_profile(values, rate, clock, choice, rows=None):
    window = {"compact": 256, "standard": 512, "detail": 1024}[choice]
    source_rate = clock.rate_range(rows)[0] if clock.local and rows else clock.rows_per_second
    spectrum = analyze_signal(values, rate, source_rate, window)
    return spectrum, {"spectrum": spectrum.summary(), "frequency_profile": spectrum.profile(),
                      "review_windows": review_sections(values, rate)}
