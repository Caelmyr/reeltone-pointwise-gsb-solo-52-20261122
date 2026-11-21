"""End-to-end and unit tests for the longitudinal overlap join feature.

Run with: python -B -m unittest discover -s tests -t .
The tests build synthetic overlapping scans in a temporary catalog; no network
or fixture files are required.
"""
import shutil
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from reeltone import overlap
from reeltone.errors import BenchError, Conflict
from reeltone.geometry import Guide, Orientation
from reeltone.join import confirm_join, find_candidates, preview_plan
from reeltone.recipe import ClockSetting, Recipe, Tone
from reeltone.raster import encode_png, read_scan
from reeltone.store import Catalog
from reeltone.web import create_app


WIDTH = 60
TRACK = (5, 45)
PERF = (0, 4)


def recipe():
    settings = {"aperture": list(TRACK), "perforations": list(PERF),
                "orientation": {"mirror_x": False, "reverse_y": False},
                "tone": {"black": 0.1, "white": 0.9, "threshold": 0.5, "polarity": "clear"},
                "clock": {"model": "manual", "row_rate": 6000, "frames_per_second": 24,
                          "holes_per_frame": 1, "pitch_guess": 250}}
    return Recipe.parse(settings, WIDTH)


def master_strip(rows=1200, seed=7):
    rng = np.random.default_rng(seed)
    r = np.arange(rows)
    chirp = np.sin(2 * np.pi * (r / 40.0 + 0.0009 * r ** 2))
    sig = 0.6 * chirp + 0.5 * rng.normal(0, 1, rows) + 0.3 * np.sin(2 * np.pi * r / 127.0) * np.linspace(0, 1, rows)
    strip = np.repeat((0.5 + 0.18 * sig / np.max(np.abs(sig)))[:, None], WIDTH, axis=1).astype(np.float32)
    return strip + rng.normal(0, 0.008, strip.shape).astype(np.float32)


def material(pixels, scan_id, title):
    return SimpleNamespace(pixels=np.clip(pixels, 0, 1).astype(np.float32), width=WIDTH, height=pixels.shape[0],
                           recipe=recipe(), recipe_record={"id": "r1", "seq": 1, "fingerprint": "f"},
                           record={"id": scan_id, "title": title, "sha256": "x" * 6, "reel_id": "reel",
                                   "metadata": {"height": pixels.shape[0], "width": WIDTH}})


class EngineTests(unittest.TestCase):
    def setUp(self):
        base = np.clip(master_strip(), 0, 1)
        self.offset = 300
        head = base[:900].copy()
        tail = base[300:].copy()
        head[500:540, 10:30] = 0.98
        tail[400:440, 20:40] = 0.03
        self.head = material(head, "h", "head")
        self.tail = material(tail, "t", "tail")

    def test_offset_recovered_from_rough_marks(self):
        candidates, _ = overlap.search_alignments(self.head, self.tail, (250, 850), (0, 600))
        self.assertEqual(candidates[0].offset, self.offset)
        self.assertGreater(candidates[0].score, 0.2)

    def test_reversed_pair_gives_negative_offset(self):
        candidates, _ = overlap.search_alignments(self.tail, self.head, (0, 600), (250, 850))
        self.assertEqual(candidates[0].offset, -self.offset)

    def test_width_and_direction_guards(self):
        narrow_pixels = np.pad(self.tail.pixels, ((0, 0), (0, 10)), mode="edge")[:, :50]
        narrow = material(narrow_pixels, "n", "narrow")
        narrow.recipe = Recipe.parse({"aperture": [5, 40], "perforations": [0, 4],
                                      "orientation": {"mirror_x": False, "reverse_y": False},
                                      "tone": {"black": 0.1, "white": 0.9, "threshold": 0.5, "polarity": "clear"},
                                      "clock": {"model": "manual", "row_rate": 6000, "frames_per_second": 24,
                                                "holes_per_frame": 1, "pitch_guess": 250}}, 50)
        with self.assertRaises(BenchError):
            overlap.search_alignments(self.head, narrow)
        mirrored = SimpleNamespace(**{**self.tail.__dict__})
        mirrored.recipe = Recipe.parse({"aperture": list(TRACK), "perforations": list(PERF),
                                        "orientation": {"mirror_x": True, "reverse_y": False},
                                        "tone": {"black": 0.1, "white": 0.9, "threshold": 0.5, "polarity": "clear"},
                                        "clock": {"model": "manual", "row_rate": 6000, "frames_per_second": 24,
                                                  "holes_per_frame": 1, "pitch_guess": 250}}, WIDTH)
        with self.assertRaises(Conflict):
            overlap.search_alignments(self.head, mirrored)

    def test_quality_plan_routes_defect_to_clearer_scan(self):
        alignment = overlap.Alignment(self.offset, 0.9, 0.9, 600, 1200)
        segments, blends, _stats, _clarity, _spans = overlap.plan_seams(
            self.head, self.tail, alignment, mode="quality", min_segment=20, crossfade=0)
        # Head rows 500-540 are damaged; those output positions must come from tail.
        damaged = [s for s in segments if s.start < 540 and s.stop > 500]
        self.assertTrue(damaged)
        self.assertTrue(all(s.source == "tail" for s in damaged))
        self.assertEqual(sum(s.stop - s.start for s in segments), 1200)

    def test_single_seam_has_two_sources_and_no_duplicates(self):
        from reeltone.join import _assert_no_duplicate_film_rows
        alignment = overlap.Alignment(self.offset, 0.9, 0.9, 600, 1200)
        segments, blends, _stats, _clarity, _spans = overlap.plan_seams(
            self.head, self.tail, alignment, mode="single", crossfade=0)
        sources = [s.source for s in segments]
        self.assertEqual(sources, ["head", "tail"])
        _assert_no_duplicate_film_rows(segments, alignment, 900, 900, blends)

    def test_assembly_pixels_match_declared_sources(self):
        alignment = overlap.Alignment(self.offset, 0.9, 0.9, 600, 1200)
        segments, blends, _stats, _clarity, _spans = overlap.plan_seams(
            self.head, self.tail, alignment, mode="quality", min_segment=20, crossfade=0)
        band, tonal = overlap.assemble_band(self.head, self.tail, alignment, segments, blends)
        self.assertEqual(band.shape, (1200, WIDTH))
        for segment in segments:
            a, b = segment.source_rows
            if segment.source == "head":
                source = self.head.pixels[a:b]
            else:
                source = np.clip(self.tail.pixels[a:b] * tonal["tonal_gain"] + tonal["tonal_bias"], 0, 1)
            np.testing.assert_allclose(band[segment.start:segment.stop], source, atol=1e-3)

    def test_overlap_physical_rows_never_repeated(self):
        from reeltone.join import _assert_no_duplicate_film_rows
        alignment = overlap.Alignment(self.offset, 0.9, 0.9, 600, 1200)
        for mode, fade in (("quality", 0), ("quality", 6), ("single", 0), ("single", 8)):
            segments, blends, _stats, _clarity, _spans = overlap.plan_seams(
                self.head, self.tail, alignment, mode=mode, min_segment=20, crossfade=fade)
            _assert_no_duplicate_film_rows(segments, alignment, 900, 900, blends)


class CatalogFlowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.catalog = Catalog(self.tmp)
        self.reel = self.catalog.create_reel({"title": "断接卷", "accession": "RT-T-1", "gauge": "16 mm", "note": ""})
        base = np.clip(master_strip(), 0, 1)
        head = base[:900].copy()
        tail = base[300:].copy()
        head[500:540, 10:30] = 0.98
        tail[400:440, 20:40] = 0.03
        self.head_id = self._import("head-pass.png", "片头扫描", head)
        self.tail_id = self._import("tail-pass.png", "片尾扫描", tail)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _import(self, name, title, pixels):
        raster = read_scan(encode_png(pixels, bits=16), name)
        scan, _ = self.catalog.add_scan(self.reel["id"], raster, title, "", recipe())
        return scan["id"]

    def _body(self, **extra):
        body = {"head_scan": self.head_id, "tail_scan": self.tail_id,
                "head_rows": [250, 850], "tail_rows": [0, 600]}
        body.update(extra)
        return body

    def test_candidates_preview_and_confirm_flow(self):
        report = find_candidates(self.catalog, self.reel["id"], self._body())
        self.assertEqual(report["candidates"][0]["offset"], 300)
        preview, plate = preview_plan(self.catalog, self.reel["id"], self._body(mode="quality", crossfade=6))
        self.assertEqual(preview["segment_count"] + 0, len(preview["manifest"]["segments"]))
        self.assertTrue(plate[:8] == b"\x89PNG\r\n\x1a\n")
        record, _ = confirm_join(self.catalog, self.reel["id"],
                                 self._body(title="接续边带", mode="quality", crossfade=6))
        self.assertEqual(record["output_scan_id"] is not None, True)
        # Derived scan is in the reel, originals are still present.
        scans = self.catalog.list_scans(self.reel["id"])
        kinds = {scan["id"]: scan["origin_kind"] for scan in scans}
        self.assertEqual(kinds[self.head_id], "imported")
        self.assertEqual(kinds[self.tail_id], "imported")
        self.assertEqual(kinds[record["output_scan_id"]], "derived_join")
        stored = self.catalog.join_for_scan(record["output_scan_id"])
        self.assertEqual(stored["id"], record["id"])
        # Provenance segments cover the whole output and reference real rows.
        manifest = stored["manifest"]
        output_scan = self.catalog.get_scan(record["output_scan_id"])
        height = output_scan["metadata"]["height"]
        covered = sum(s["output_stop"] - s["output_start"] for s in manifest["segments"])
        self.assertEqual(covered, height)
        for segment in manifest["segments"]:
            self.assertIn(segment["source_scan_id"], (self.head_id, self.tail_id))

    def test_derived_band_opens_in_existing_reader(self):
        record, _ = confirm_join(self.catalog, self.reel["id"], self._body(mode="single", crossfade=0))
        from reeltone.bench import load_material
        material = load_material(self.catalog, record["output_scan_id"])
        self.assertEqual(material.height, 1200)
        self.assertEqual(material.width, WIDTH)

    def test_http_routes_and_originals_untouched(self):
        app = create_app(self.tmp)
        client = app.test_client()
        response = client.post(f"/api/reels/{self.reel['id']}/joins/candidates", json=self._body())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["candidates"][0]["offset"], 300)
        response = client.post(f"/api/reels/{self.reel['id']}/joins",
                               json=self._body(title="HTTP 接续", mode="single", crossfade=0))
        self.assertEqual(response.status_code, 201)
        join_id = response.get_json()["join"]["id"]
        response = client.get(f"/api/reels/{self.reel['id']}/joins")
        self.assertTrue(any(item["id"] == join_id for item in response.get_json()["joins"]))
        # The two source scans remain downloadable byte-identical.
        for scan_id in (self.head_id, self.tail_id):
            raw = self.catalog.get_scan(scan_id, raw=True)
            response = client.get(f"/api/scans/{scan_id}/original")
            self.assertEqual(response.data, raw["raw"])


if __name__ == "__main__":
    unittest.main()
