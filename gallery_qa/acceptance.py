#!/usr/bin/env python3
"""End-to-end acceptance checks for the gallery redaction feature.

Run in isolated mode from the repository root:
    python3 -I gallery_qa/acceptance.py
The script only creates data under a temporary directory; the seeded working
library and original scans are never modified.
"""
import io
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), "/opt/pointwise-env/site-packages"]

import numpy as np
from PIL import Image

from reeltone.audition import Monitor, audition, read_wav
from reeltone.bench import load_material
from reeltone.demo import seed_library
from reeltone.gallery import build_gallery, verify_gallery
from reeltone.optical import read_area
from reeltone.redact import parse_spans, redacted_pixels, render_gallery
from reeltone.store import Catalog
from reeltone.web import create_app

FAILURES = []


def check(name, condition, detail=""):
    print(f"[{'PASS' if condition else 'FAIL'}] {name} {detail}")
    if not condition:
        FAILURES.append(name)


def main():
    with tempfile.TemporaryDirectory() as directory:
        catalog = Catalog(directory)
        seeded = seed_library(catalog)
        scan = seeded["training"][0]["scans"][0]
        scan_id = scan["id"]
        height = scan["metadata"]["height"]
        material = load_material(catalog, scan_id)
        spans = parse_spans({"spans": [
            {"rows": [2000, 2600], "label": "暂不公开段一", "note": "版权核对"},
            {"rows": [5000, 5200], "label": "暂不公开段二"}]}, height)
        feather = 40
        monitor = {"mode": "width", "rate": 24000, "gain_db": 6, "low_hz": 40,
                   "high_hz": 0, "ramp_ms": 6, "center": "median"}

        # 1. Image redaction: dimensions preserved, held rows fully replaced,
        #    outside rows bit-identical, feather rows smoothly blended.
        redacted, weight, hard, controlled = redacted_pixels(material.pixels, spans, feather)
        check("图像尺寸与原片一致", redacted.shape == material.pixels.shape, str(redacted.shape))
        held_rows = np.flatnonzero(hard)
        outside = np.flatnonzero(~controlled)
        check("遮盖行像素与原图完全不同", not np.allclose(redacted[held_rows], material.pixels[held_rows]))
        check("非控制区像素逐位未改", np.array_equal(redacted[outside], material.pixels[outside]))
        smooth = controlled & ~hard
        diff = np.abs(redacted.astype(np.float64) - material.pixels.astype(np.float64))
        check("衔接区为部分混合(非全替换也非零差)", 0 < float(diff[smooth].mean()) < float(np.abs(
            redacted[hard].astype(np.float64) - material.pixels[hard].astype(np.float64)).mean()))

        # 2. Audio: same length as the unredacted full-strip monitor, held
        #    samples exactly zero, elsewhere identical to an unredacted monitor
        #    built with the same phase-preserving stretch pipeline.
        from reeltone.audition import apply_fade, normalize_signal, shaping_filter
        from reeltone.redact import _stretch_rows
        clock, _ = material.clock()
        plain_reading = read_area(material.pixels, material.recipe.aperture, material.recipe.tone,
                                  (0, height))
        settings = Monitor.parse(monitor)
        native, _ = normalize_signal(plain_reading, settings, clock)
        filtered, _ = shaping_filter(native, clock.rows_per_second, settings)
        plain_wave = apply_fade(_stretch_rows(filtered, clock.rows_per_second, settings.rate)
                                * 10.0 ** (settings.gain_db / 20.0),
                                round(settings.rate * settings.ramp_ms / 1000.0))
        wave, rate, image_png, safety, result, _ = render_gallery(material, spans, feather, monitor)
        check("监听总帧数与未遮盖全片一致", len(wave) == len(plain_wave), f"{len(wave)} vs {len(plain_wave)}")
        check("遮盖区监听严格静音", safety["max_level_on_held"] == 0.0)
        held_samples = np.zeros(len(wave), dtype=bool)
        for at, stop in safety["held_sample_edges"]:
            held_samples[at:stop] = True
        # Output-domain feather windows (raised cosine rebuilt on output samples).
        feather_frames = max(1, round(feather * rate / clock.rows_per_second))
        blend_rows = np.zeros(len(wave), dtype=bool)
        for span in spans:
            t0, t1 = clock.extent((span.y0, span.y1))
            lo = max(0, int(round(t0 * rate)) - feather_frames - 2)
            hi = min(len(wave), int(round(t1 * rate)) + feather_frames + 2)
            blend_rows[lo:hi] = True
        untouched = ~blend_rows
        check("衔接区外样本与原监听一致", np.allclose(wave[untouched], plain_wave[untouched], atol=1e-9, rtol=0),
              f"max diff {float(np.max(np.abs(wave[untouched] - plain_wave[untouched]))):.2e}")
        # Inside the controlled playback window the raised-cosine envelope keeps
        # joins smooth: the largest step must be no worse than the source scale.
        step_plain = float(np.max(np.abs(np.diff(plain_wave[blend_rows]))))
        step_gated = float(np.max(np.abs(np.diff(wave[blend_rows]))))
        check("衔接区无超过原信号尺度的突变", step_gated <= step_plain * 1.05 + 1e-9,
              f"max step {step_gated:.4f} vs {step_plain:.4f}")
        check("遮盖块时间保留(非零时长占位)", held_samples.sum() > 0)

        # 3. Packet contents: only redacted strip, monitor, row ranges, manifest.
        built = build_gallery(catalog, scan_id, {"title": "展厅阅览包", "spans": [
            {"rows": [2000, 2600], "label": "暂不公开段一", "note": "版权核对"},
            {"rows": [5000, 5200], "label": "暂不公开段二"}], "feather_rows": feather, "monitor": monitor})
        with zipfile.ZipFile(io.BytesIO(built["archive"])) as archive:
            names = archive.namelist()
            check("ZIP 只含四类文件", set(names) == {"manifest.json", "redacted-row-ranges.csv",
                  next(n for n in names if n.endswith("-redacted.png")),
                  next(n for n in names if n.endswith("-monitor.wav"))}, str(names))
            check("ZIP 不含原始扫描字节",
                  scan["sha256"] not in {h for n in names for h in [__import__("hashlib").sha256(
                      archive.read(n)).hexdigest()]})
            png = Image.open(io.BytesIO(archive.read(next(n for n in names if n.endswith(".png")))))
            check("包内边带图尺寸与原图一致", png.size == (material.width, height))
            csv_text = archive.read("redacted-row-ranges.csv").decode("utf-8-sig")
            check("行区间表保留原片行号", "2000,2600" in csv_text and "5000,5200" in csv_text)

        # 4. Verification: clean packet verifies; tampering is detected.
        verdict = verify_gallery(built["archive"])
        check("干净阅览包通过核验", verdict["verified"])
        tampered = bytearray(built["archive"])
        tampered[3000] ^= 0x01
        detected = False
        try:
            detected = not verify_gallery(bytes(tampered))["verified"]
        except Exception:
            detected = True
        check("篡改后核验失败", detected)

        # 5. Original library untouched: stored raw bytes digest unchanged.
        fresh = catalog.get_scan(scan_id, raw=True)
        check("馆内原始扫描摘要未变", fresh["sha256"] == scan["sha256"])
        check("现有监听数量未受影响", len(catalog.list_takes(scan_id)) == 0)

        # 6. Overlapping spans rejected; full-strip span rejected.
        try:
            parse_spans({"spans": [{"rows": [10, 30]}, {"rows": [25, 40]}]}, height)
            check("重叠遮盖区间被拒绝", False)
        except Exception:
            check("重叠遮盖区间被拒绝", True)
        try:
            parse_spans({"spans": [{"rows": [0, height]}]}, height)
            check("整卷遮盖被拒绝", False)
        except Exception:
            check("整卷遮盖被拒绝", True)

        # 7. Reverse orientation (second demo scan) keeps row mapping native.
        reverse_scan = seeded["training"][0]["scans"][1]
        rev_material = load_material(catalog, reverse_scan["id"])
        rev_spans = parse_spans({"spans": [{"rows": [1000, 1400], "label": "反向片遮盖"}]},
                                rev_material.height)
        rev_wave, _, _, rev_safety, _, _ = render_gallery(rev_material, rev_spans, feather, monitor)
        rev_clock, _ = rev_material.clock()
        expected_lo = round((rev_material.height - 1400) / rev_clock.rows_per_second * 24000)
        check("反向走片仍按原片行映射静音", rev_safety["held_sample_edges"][0][0] in range(expected_lo - 3, expected_lo + 4),
              str(rev_safety["held_sample_edges"][0]))

        # 8. HTTP end-to-end through the Flask app.
        app = create_app(directory)
        client = app.test_client()
        response = client.post(f"/api/scans/{scan_id}/redaction/preview", json={
            "spans": [{"rows": [3000, 3100], "label": "HTTP 遮盖"}], "feather_rows": 24, "monitor": monitor})
        check("预览接口 200 且静音", response.status_code == 200 and
              response.get_json()["safety"]["max_level_on_held"] == 0.0, str(response.status_code))
        reel_id = scan["reel_id"]
        response = client.post(f"/api/reels/{reel_id}/gallery", json={
            "scan": scan_id, "title": "HTTP 阅览包",
            "spans": [{"rows": [3000, 3100], "label": "HTTP 遮盖"}], "feather_rows": 24, "monitor": monitor})
        check("阅览包接口 201", response.status_code == 201, str(response.status_code))
        packet_id = response.get_json()["packet"]["id"]
        response = client.get(f"/api/packets/{packet_id}/verify")
        check("已保存阅览包校验通过", response.status_code == 200 and response.get_json()["verified"])
        response = client.post("/api/packets/verify", data={"packet": (io.BytesIO(built["archive"]), "g.zip")},
                               content_type="multipart/form-data")
        check("统一核验接口按 schema 分流", response.status_code == 200 and response.get_json()["schema"]
              == "reeltone-gallery-v1")

    print("\n结果:", "全部通过" if not FAILURES else f"{len(FAILURES)} 项失败: {FAILURES}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
