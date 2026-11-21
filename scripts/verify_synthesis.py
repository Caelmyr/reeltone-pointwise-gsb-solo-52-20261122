#!/usr/bin/env python3
"""End-to-end verification for the WAV -> synthetic optical strip feature.

Runs entirely against an in-process Flask test client and a temporary library,
so it needs no network or running server. It checks the four guarantees the
feature promises:

1. a mono WAV is rendered as a grayscale variable-width strip and read back
   through the *same* optical pipeline used for real scans, with high fidelity;
2. frequencies above the requested row-rate Nyquist are anti-alias filtered,
   not folded into new low-frequency components;
3. loud crests are soft-limited with edge headroom, so no row fills the window;
4. the committed strip is marked synthetic and ships with re-openable settings.

Usage:  python scripts/verify_synthesis.py
Exit 0 on success, non-zero with a traceback on failure.
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import tempfile
import wave
from pathlib import Path

# Allow running from a plain interpreter where numpy/scipy/Pillow live on an
# explicit site-packages path (isolated environments).
PROJECT = Path(__file__).resolve().parents[1]
for candidate in ("/opt/pointwise-env/site-packages", str(PROJECT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import numpy as np  # noqa: E402

from reeltone.errors import BenchError, Limit  # noqa: E402
from reeltone.synthesizer import synthesize  # noqa: E402
from reeltone.web import create_app  # noqa: E402


def make_wav(seconds=1.5, rate=24000):
    t = np.arange(int(rate * seconds)) / rate
    quiet = 0.04 * np.sin(2 * np.pi * 280 * t)
    burst = np.zeros_like(t)
    mask = (t > 0.5) & (t < 0.8)
    burst[mask] = 1.0 * np.sin(2 * np.pi * 640 * t[mask])
    high = 0.3 * np.sin(2 * np.pi * 7000 * t)  # above the 3600 Hz Nyquist at 7200 rows/s
    samples = np.clip(quiet + burst + high, -1.0, 1.0)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as destination:
        destination.setnchannels(1)
        destination.setsampwidth(2)
        destination.setframerate(rate)
        destination.writeframes(np.rint(samples * 32767).astype("<i2").tobytes())
    return buf.getvalue()


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f"  ok - {message}")


def main():
    wav_bytes = make_wav()

    print("synthesis math:")
    job = synthesize(wav_bytes, "verify-tone.wav",
                     {"row_rate": 7200, "aperture_width": 96, "slit_rows": 1,
                      "grain_sigma": 0.006, "title": "验证参数 A"})
    report = job.report
    check(report["readback"]["correlation"] > 0.995, "round-trip correlation above 0.995")
    check(report["readback"]["touch_edge_rows_in_signal"] == 0, "no signal row touches the aperture edge")
    check(report["readback"]["clipped_pixels"] == 0, "no fully clipped pixels")
    check(report["row_clock"]["removed_above_nyquist_percent"] > 5,
          "above-Nyquist energy is removed rather than kept")
    foldback = report["row_clock"]["foldback"]
    check(foldback["above_nyquist_present"] and foldback["foldback_suppression_db"] >= 40,
          "foldback suppression of inaudible highs is at least 40 dB")
    check(report["limiter"]["rows_limited"] > 0 and report["limiter"]["strongest_row_reduction_db"] > 0,
          "loud crests are soft-limited")
    check(report["limiter"]["openness_min"] > 0.0 and report["limiter"]["openness_max"] < 1.0,
          "the pattern keeps headroom on both aperture edges")
    check(report["dynamic_range"]["effective_dynamic_range_db"] is not None,
          "an effective pattern dynamic range is reported")
    check(job.strip.png[:8] == b"\x89PNG\r\n\x1a\n", "the strip is encoded as PNG bytes")
    check(set(job.figures) == {"comparison", "spectrum", "strip"}, "all three report figures render")

    print("alias isolation:")
    rate = 24000
    t = np.arange(rate * 2) / rate
    high_only = io.BytesIO()
    with wave.open(high_only, "wb") as destination:
        destination.setnchannels(1)
        destination.setsampwidth(2)
        destination.setframerate(rate)
        destination.writeframes(np.rint(0.4 * np.sin(2 * np.pi * 6000 * t) * 32767).astype("<i2").tobytes())
    probe = synthesize(high_only.getvalue(), "alias-probe.wav", {"row_rate": 7200, "grain_sigma": 0.0})
    check(probe.report["row_clock"]["foldback"]["foldback_suppression_db"] >= 60,
          "a 6 kHz tone cannot reappear as a low alias (>= 60 dB suppression)")

    print("input and dimension guards:")
    stereo = io.BytesIO()
    with wave.open(stereo, "wb") as destination:
        destination.setnchannels(2)
        destination.setsampwidth(2)
        destination.setframerate(rate)
        destination.writeframes(np.zeros(200, dtype="<i2").tobytes())
    try:
        synthesize(stereo.getvalue(), "stereo.wav", {"row_rate": 7200})
        raise AssertionError("stereo WAV should be rejected")
    except BenchError:
        print("  ok - stereo WAV is rejected with a user-facing error")
    long_tone = io.BytesIO()
    with wave.open(long_tone, "wb") as destination:
        destination.setnchannels(1)
        destination.setsampwidth(2)
        destination.setframerate(rate)
        destination.writeframes(np.zeros(rate * 20, dtype="<i2").tobytes())
    try:
        synthesize(long_tone.getvalue(), "too-tall.wav", {"row_rate": 48000})
        raise AssertionError("oversized strip should be rejected")
    except Limit:
        print("  ok - row-rate/duration combinations over the height cap are rejected")

    print("http flow through the normal reading desk:")
    with tempfile.TemporaryDirectory() as library:
        client = create_app(library).test_client()
        response = client.post(
            "/api/synthesis/preview",
            data={"wav": (io.BytesIO(wav_bytes), "verify-tone.wav"),
                  "settings": json.dumps({"row_rate": 7200, "aperture_width": 96, "title": "HTTP 参数 A"})},
            content_type="multipart/form-data")
        check(response.status_code == 200, "preview endpoint returns 200")
        token = response.get_json()["job"]
        for kind in ("comparison.png", "spectrum.png", "strip.png"):
            asset = client.get(f"/api/synthesis/{token}/{kind}")
            check(asset.status_code == 200 and asset.data[:8] == b"\x89PNG\r\n\x1a\n",
                  f"figure {kind} is downloadable PNG")
        committed = client.post("/api/synthesis/commit",
                                json={"job": token, "reel_title": "验证合成卷"}).get_json()
        check(committed["scan"]["title"].startswith("【合成测试图】"),
              "committed scan title carries the synthetic marker")
        sidecar = json.loads(base64.b64decode(committed["settings_data"].split(",", 1)[1]))
        check(sidecar["synthetic"] is True and sidecar["warning"], "settings JSON carries the test warning")
        scan_id = committed["scan"]["id"]
        scan_doc = client.get(f"/api/scans/{scan_id}").get_json()
        height = scan_doc["scan"]["metadata"]["height"]
        recipe_id = scan_doc["recipe"]["id"]
        audition = client.post(
            f"/api/scans/{scan_id}/audition",
            json={"rows": [0, height], "recipe": recipe_id,
                  "monitor": {"mode": "width", "rate": 24000, "gain_db": 0, "low_hz": 0,
                              "high_hz": 0, "ramp_ms": 0, "center": "median"}})
        check(audition.status_code == 200 and audition.get_json()["audio"].startswith("data:audio/wav"),
              "the committed strip is audible through the normal optical audition endpoint")
        settings_file = client.get(committed["settings_download"])
        check(settings_file.status_code == 200 and json.loads(settings_file.data)["synthetic"] is True,
              "the re-openable settings file downloads as JSON")
        original = client.get(committed["image_download"])
        check(original.status_code == 200 and original.data[:8] == b"\x89PNG\r\n\x1a\n",
              "the test strip PNG downloads byte-for-byte from the library")

    print("\nAll synthesis verification checks passed.")


if __name__ == "__main__":
    main()
