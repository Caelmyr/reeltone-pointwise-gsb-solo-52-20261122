"""Behavioural checks for the multi-band splice line inspector.

Run with:  python3 -I -m unittest discover -s tests -t .
The synthetic scenes make one slanted brightness line visible in chosen bands, so the
three evidence grades can be checked without any demo library.
"""
import sys
import unittest

import numpy as np

sys.path.insert(0, ".")

from reeltone.bench import Material
from reeltone.demo import INK_CART, sample_recipe
from reeltone.raster import encode_png, read_scan
from reeltone.splice_bands import inspect_splice_bands


HEIGHT, WIDTH = 2000, 244


def bands(columns=None):
    columns = columns or ((6, 35), (52, 145), (154, 244))
    return [[left, 1400, right, 1600] for left, right in columns]


BANDS = bands()


def recipe():
    return sample_recipe([item for item in INK_CART if item.variant == "wear"][0])


def material(image):
    raster = read_scan(encode_png(image, bits=8), "synthetic.png")
    settings = recipe()
    record = {"id": "a" * 32, "sha256": "b" * 64, "metadata": raster.metadata()}
    stored = {"id": "c" * 32, "seq": 1, "fingerprint": settings.fingerprint(),
              "settings": settings.as_dict()}
    return Material(record, raster, stored, settings)


def scene(visible=(0, WIDTH), slope=0.12, centre=1500, seed=7):
    rng = np.random.default_rng(seed)
    image = 0.25 + 0.02 * rng.normal(size=(HEIGHT, WIDTH))
    rows = np.arange(HEIGHT)[:, None]
    columns = np.arange(WIDTH)[None, :]
    for frame in np.arange(100, HEIGHT, 250):
        image[frame:frame + 12, 154:] = 0.45
    mask = np.abs(rows - (centre + slope * columns)) <= 4
    mask[:, :visible[0]] = False
    mask[:, visible[1]:] = False
    image = np.where(mask, 0.78, image)
    return np.clip(image, 0, 1).astype(np.float32)


def inspect(image, bands=BANDS, **extra):
    request = {"bands": bands, "window": 8, "contrast": 0.08, "max_slant": 0.35, **extra}
    summary, plate, chart = inspect_splice_bands(material(image), request)
    return summary, plate, chart


class SpliceBandTests(unittest.TestCase):
    def test_slanted_line_seen_by_all_bands_is_corroborated(self):
        summary, plate, chart = inspect(scene())
        self.assertEqual(summary["band_count"], 3)
        self.assertGreaterEqual(summary["corroborated_count"], 1)
        best = summary["findings"][0]
        self.assertEqual(best["classification"], "corroborated")
        self.assertGreaterEqual(best["support_count"], 2)
        self.assertAlmostEqual(best["slope_rows_per_column"], 0.12, delta=0.02)
        self.assertTrue(all(item["seen"] for item in best["observations"]))
        self.assertGreater(len(plate), 1000)
        self.assertGreater(len(chart), 1000)

    def test_line_missing_on_reflective_side_is_reported_per_band(self):
        summary, _plate, _chart = inspect(scene(visible=(0, 180)))
        best = summary["findings"][0]
        self.assertEqual(best["classification"], "corroborated")
        picture = next(item for item in best["observations"] if item["role"] == "picture")
        self.assertFalse(picture["seen"])
        self.assertIn(picture["evidence_state"], ("partial", "absent"))

    def test_picture_edge_only_is_kept_but_separated(self):
        summary, _plate, _chart = inspect(scene(visible=(154, 195)))
        self.assertEqual(summary["candidate_count"], len(summary["findings"]))
        self.assertGreaterEqual(summary["picture_only_count"], 1)
        picture_lines = [item for item in summary["findings"] if item["classification"] == "picture_only"]
        self.assertTrue(picture_lines)
        self.assertTrue(all(item["roles"] == ["picture"] for item in picture_lines))
        self.assertNotIn("corroborated", {item["classification"] for item in picture_lines})

    def test_single_band_support_is_limited_but_retained(self):
        summary, _plate, _chart = inspect(scene(visible=(52, 145)))
        self.assertGreaterEqual(summary["limited_count"], 1)
        limited = [item for item in summary["findings"] if item["classification"] == "single_band"]
        self.assertTrue(limited)
        self.assertEqual(limited[0]["support_count"], 1)

    def test_frame_edges_alone_never_count_as_complete_evidence(self):
        summary, _plate, _chart = inspect(scene(visible=(1, 1)))
        # Frame edges may still be retained as picture-only candidates, but never as a
        # splice with complete evidence, and never with support outside the picture band.
        self.assertEqual(summary["corroborated_count"], 0)
        self.assertEqual(summary["limited_count"], summary["candidate_count"])
        for finding in summary["findings"]:
            self.assertEqual(finding["roles"], ["picture"])

    def test_clean_picture_area_without_edges_reports_nothing(self):
        summary, _plate, _chart = inspect(scene(visible=(1, 1)))
        clean = scene(visible=(1, 1), centre=1400)
        summary, _plate, _chart = inspect(clean)
        self.assertEqual(summary["corroborated_count"], 0)

    def test_bands_must_span_different_columns(self):
        with self.assertRaises(Exception):
            inspect(scene(), bands=[[6, 1400, 35, 1600], [20, 1400, 40, 1600]])

    def test_bands_need_a_shared_row_range(self):
        with self.assertRaises(Exception):
            inspect(scene(), bands=[[6, 100, 35, 200], [52, 1500, 145, 1600]])

    def test_line_extent_is_reported_in_source_rows(self):
        summary, _plate, _chart = inspect(scene())
        best = summary["findings"][0]
        left, right = best["line_rows"]
        self.assertLess(left, right)
        self.assertGreater(best["focus_column"], 0)
        self.assertGreaterEqual(best["focus_row"], 0)


if __name__ == "__main__":
    unittest.main()
