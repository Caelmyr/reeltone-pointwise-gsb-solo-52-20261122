"""Application service for two-pass overlap joins.

Keeps the HTTP layer and the numeric overlap engine apart: candidates and seam
plans are computed on demand and never persisted; confirmation renders a new
continuous raster, stores it as a derived scan alongside the untouched source
scans, and records full per-segment provenance.
"""
from hashlib import sha256
from base64 import b64encode
from io import BytesIO

import numpy as np
from PIL import Image

from .bench import load_material
from .contracts import identifier, safe_filename, text
from .errors import BenchError, Conflict
from . import overlap
from .raster import read_scan
from .recipe import Recipe


def _pair(catalog, reel_id, values):
    head_id = identifier(values.get("head_scan"), "片头扫描")
    tail_id = identifier(values.get("tail_scan"), "片尾扫描")
    if head_id == tail_id:
        raise BenchError("请选择同卷的两张不同扫描。")
    head = load_material(catalog, head_id, values.get("head_recipe") or None)
    tail = load_material(catalog, tail_id, values.get("tail_recipe") or None)
    if head.record["reel_id"] != reel_id or tail.record["reel_id"] != reel_id:
        raise Conflict("两张扫描须属于同一个档案卷。")
    return head, tail


def find_candidates(catalog, reel_id, values):
    head, tail = _pair(catalog, reel_id, values)
    head_mark, tail_mark, _, _, _ = overlap.parse_join_request(values, head.height, tail.height)
    candidates, curve = overlap.search_alignments(head, tail, head_mark, tail_mark)
    best, second = candidates[0].score, candidates[1].score if len(candidates) > 1 else None
    ambiguous = second is not None and (best - second) < 0.06
    curve_png = overlap.score_curve_png(curve, candidates)
    return {
        "head": {"scan_id": head.record["id"], "title": head.record["title"], "height": head.height},
        "tail": {"scan_id": tail.record["id"], "title": tail.record["title"], "height": tail.height},
        "candidates": [item.as_dict() for item in candidates],
        "ambiguous": bool(ambiguous),
        "ambiguity_gap": round(best - second, 6) if second is not None else None,
        "curve": curve,
        "curve_image": "data:image/png;base64," + b64encode(curve_png).decode("ascii"),
        "notice": "前两个候选的相关系数很接近，请在预览中核对后再确认。" if ambiguous else
                  "候选按相关系数排序；确认前请核对原片行对应关系。",
    }


def preview_plan(catalog, reel_id, values):
    head, tail = _pair(catalog, reel_id, values)
    head_mark, tail_mark, mode, crossfade, min_segment = overlap.parse_join_request(values, head.height, tail.height)
    alignment = _selected_alignment(head, tail, values, head_mark, tail_mark)
    segments, blends, stats, clarity, spans = overlap.plan_seams(
        head, tail, alignment, mode=mode, min_segment=min_segment, crossfade=crossfade)
    manifest = _manifest(head, tail, alignment, segments, blends, stats, spans, values)
    plate = overlap.join_plate(head, tail, alignment, segments, clarity)
    return {"manifest": manifest, "preview_sha256": sha256(plate).hexdigest(),
            "segment_count": len(segments), "switch_count": len(blends)}, plate


def _selected_alignment(head, tail, values, head_mark, tail_mark):
    offset = values.get("offset")
    if offset is None:
        candidates, _ = overlap.search_alignments(head, tail, head_mark, tail_mark)
        return candidates[0]
    offset = int(offset)
    head_rows, tail_rows, _output = overlap.overlap_extent(head.height, tail.height, offset)
    track_a = overlap._row_feature(head.pixels, head.recipe.aperture.as_list())
    track_b = overlap._row_feature(tail.pixels, tail.recipe.aperture.as_list())
    score, overlap_rows = overlap._ncc_at(track_a, track_b, offset)
    if score is None:
        raise BenchError("选定的偏移共同行不足。")
    perf_a = overlap._row_feature(head.pixels, head.recipe.perforations.as_list())
    perf_b = overlap._row_feature(tail.pixels, tail.recipe.perforations.as_list())
    second, _ = overlap._ncc_at(perf_a, perf_b, offset)
    return overlap.Alignment(offset, score, second or 0.0, overlap_rows,
                             head.height + tail.height - overlap_rows)


def _manifest(head, tail, alignment, segments, blends, stats, spans, values):
    head_rows, tail_rows = spans
    return {
        "schema": "reeltone-overlap-join-v1",
        "head_scan_id": head.record["id"], "tail_scan_id": tail.record["id"],
        "head_title": head.record["title"], "tail_title": tail.record["title"],
        "head_sha256": head.record["sha256"], "tail_sha256": tail.record["sha256"],
        "head_recipe_id": head.recipe_record["id"], "tail_recipe_id": tail.recipe_record["id"],
        "offset_rows": alignment.offset, "ncc_score": round(alignment.score, 6),
        "overlap": {"head_rows": list(head_rows), "tail_rows": list(tail_rows), "rows": alignment.overlap_rows},
        "mode": values.get("mode", "quality"), "crossfade_rows": stats["crossfade_rows"],
        "clarity": stats, "tonal": None,
        "segments": [item.as_dict() for item in segments], "blends": blends,
        "orientation": head.recipe.orientation.as_dict(),
        "aperture": head.recipe.aperture.as_list(),
        "warranty": "每个输出行仅引用一张原扫描的一行；原片物理行不会因换源而重复进入。",
    }


def _encode_band(band):
    pixels = np.rint(np.clip(band, 0.0, 1.0) * 65535.0).astype("<u2")
    output = BytesIO()
    Image.fromarray(pixels).save(output, "PNG", compress_level=4)
    return output.getvalue()


def confirm_join(catalog, reel_id, values):
    head, tail = _pair(catalog, reel_id, values)
    head_mark, tail_mark, mode, crossfade, min_segment = overlap.parse_join_request(values, head.height, tail.height)
    alignment = _selected_alignment(head, tail, values, head_mark, tail_mark)
    segments, blends, stats, clarity, spans = overlap.plan_seams(
        head, tail, alignment, mode=mode, min_segment=min_segment, crossfade=crossfade)
    band, tonal = overlap.assemble_band(head, tail, alignment, segments, blends)
    _assert_no_duplicate_film_rows(segments, alignment, head.height, tail.height, blends)
    manifest = _manifest(head, tail, alignment, segments, blends, stats, spans, values)
    manifest["tonal"] = tonal
    png = _encode_band(band)
    filename = safe_filename(values.get("filename") or "joined-strip.png", "joined-strip.png")
    raster = read_scan(png, filename)
    title = text(values.get("title", "接续连续边带"), "接续边带名称", maximum=110)
    note = text(values.get("note", "由两张同卷扫描按确认的纵向重叠接续生成。"), "接续说明",
                maximum=700, required=False, multiline=True)
    recipe = Recipe.parse(head.recipe_record["settings"], raster.width)
    record = catalog.save_join(reel_id, raster=raster, title=title, note=note, recipe=recipe,
                               head_id=head.record["id"], tail_id=tail.record["id"], manifest=manifest)
    return record, overlap.join_plate(head, tail, alignment, segments, clarity)


def _assert_no_duplicate_film_rows(segments, alignment, head_h, tail_h, blends=()):
    """Verify physical-row uniqueness across the single-source segments.

    Segments partition the output without overlap.  Blend windows are short
    crossfades at switches and intentionally read both scans for a few rows, so
    they are checked for bounds and a capped width instead of uniqueness.
    """
    seen_head, seen_tail = set(), set()
    for segment in segments:
        a, b = segment.source_rows
        bucket, limit = (seen_head, head_h) if segment.source == "head" else (seen_tail, tail_h)
        if not (0 <= a < b <= limit):
            raise Conflict("来源行越界。")
        if bucket.intersection(range(a, b)):
            raise Conflict("同一段原片行被重复选用。")
        bucket.update(range(a, b))
    head_rows, tail_rows, _output = overlap.overlap_extent(head_h, tail_h, alignment.offset)
    # Physical positions expressed in head coordinates: tail row t maps to head
    # row t + offset.  Only rows actually selected by a segment participate, and
    # every overlap position must be owned by exactly one of the two scans.
    tail_selected = {t for segment in segments if segment.source == "tail"
                     for t in range(*segment.source_rows)}
    head_selected = seen_head
    physical_from_tail = {t + alignment.offset for t in tail_selected}
    overlap_positions = set(range(*head_rows))
    covered = head_selected.intersection(overlap_positions) | physical_from_tail.intersection(overlap_positions)
    if covered != overlap_positions:
        raise Conflict("重叠区存在未被任何来源覆盖的原片行。")
    if head_selected.intersection(physical_from_tail):
        raise Conflict("重叠区的同一物理行被两张扫描重复选用。")
    for blend in blends:
        if blend["rows"] > 40:
            raise Conflict("过渡段过长，可能掩盖重复内容。")
