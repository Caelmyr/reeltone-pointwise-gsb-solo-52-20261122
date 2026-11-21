"""Fragment tracing: locate where an uploaded fragment may have been cropped from.

The query image is compared only in memory; nothing in this module writes it to
the catalog.  Matching uses zero-mean normalized cross correlation so global
exposure differences between the fragment and the scan do not dominate.  Both
upload orientations are tried because a surviving fragment may be upside down.
Every position above the threshold is returned as a candidate and the caller
(an archivist) decides which one, if any, is the real source.
"""
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import maximum_filter
from scipy.signal import fftconvolve

from .errors import BenchError
from .geometry import Box
from .raster import byte_preview, encode_image, read_scan
from .visual import ACCENT, GRID, MUTED, PAGE, PANEL, TEAL, TEXT, print_text

MIN_SIDE = 12
COARSE_FACTOR = 2
COARSE_MAX_ROWS = 12000
NATIVE_STRIP_ROWS = 2600
REFINE_RADIUS = 2
MIN_WINDOW_STD = 0.008
MIN_TEMPLATE_STD = 0.006
COARSE_PEAK_THRESHOLD = 0.55
MAX_CANDIDATES = 24
PEAK_CAP = 40
ORIENTATIONS = (("native", False), ("reverse_y", True))


@dataclass(frozen=True)
class TraceCandidate:
    scan_id: str
    box: Box
    score: float
    orientation: str

    def as_dict(self):
        return {"scan_id": self.scan_id, "box": self.box.as_list(), "score": round(self.score, 5),
                "orientation": self.orientation, "coordinate_resolution": "native", "accepted": False}


def _shrink(values, factor):
    if factor <= 1:
        return values
    image = byte_preview(values)
    size = (max(1, round(values.shape[1] / factor)), max(1, round(values.shape[0] / factor)))
    return np.asarray(image.resize(size, Image.Resampling.BOX), dtype=np.float32) / 255.0


def ncc_map(image, template, min_std=MIN_WINDOW_STD):
    """Zero-mean NCC over every fully contained window; flat windows stay NaN.

    Window sums come from zero-padded integral images, so every fully
    contained window uses its own real pixels (no reflected border values).
    Correlation still uses FFT convolution.
    """
    height, width = image.shape
    th, tw = template.shape
    if th > height or tw > width:
        return None
    out_h, out_w = height - th + 1, width - tw + 1
    pixels = th * tw
    image64 = image.astype(np.float64)
    cumulative = np.pad(image64, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    cumulative_sq = np.pad(image64 ** 2, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    ys = np.arange(out_h)[:, None]
    xs = np.arange(out_w)[None, :]
    y1, x1 = ys + th, xs + tw
    sums = cumulative[y1, x1] - cumulative[ys, x1] - cumulative[y1, xs] + cumulative[ys, xs]
    squares = cumulative_sq[y1, x1] - cumulative_sq[ys, x1] - cumulative_sq[y1, xs] + cumulative_sq[ys, xs]
    variance = squares - sums * sums / pixels
    correlation = fftconvolve(image64, template[::-1, ::-1].astype(np.float64), mode="valid")[:out_h, :out_w]
    centered = template - template.mean()
    template_den = np.sqrt((centered * centered).sum())
    result = np.full((out_h, out_w), np.nan, dtype=np.float64)
    usable = variance > min_std ** 2 * pixels
    result[usable] = np.clip(
        (correlation[usable] - sums[usable] * template.mean())
        / (np.sqrt(np.maximum(variance[usable], 1e-18)) * template_den), -1.0, 1.0)
    return result


def _peaks(score_map, threshold, min_y, min_x, cap=None):
    filled = np.nan_to_num(score_map, nan=-2.0)
    footprint = np.ones((max(3, min_y), max(3, min_x)), dtype=bool)
    local_max = maximum_filter(filled, footprint=footprint, mode="nearest")
    positions = np.argwhere((filled == local_max) & (score_map > threshold))
    if cap is not None:
        positions = positions[np.argsort(score_map[positions[:, 0], positions[:, 1]])[::-1]][:cap]
    return [(int(y), int(x), float(score_map[y, x])) for y, x in positions]


def _dedup_peaks(peaks, min_y, min_x, cap=PEAK_CAP):
    """Global NMS across strip boundaries, strongest first."""
    ordered = sorted(peaks, key=lambda item: item[2], reverse=True)
    kept = []
    for y, x, score in ordered:
        if all(abs(y - ky) >= min_y or abs(x - kx) >= min_x for ky, kx, _ in kept):
            kept.append((y, x, score))
        if len(kept) >= cap:
            break
    return kept


def _subpixel_peak(score_map, y, x):
    """Parabolic fit on a 3x3 coarse neighbourhood to beat cell quantization."""
    fy, fx = float(y), float(x)
    if 0 < y < score_map.shape[0] - 1:
        a, b, c = score_map[y - 1, x], score_map[y, x], score_map[y + 1, x]
        if np.isfinite(a) and np.isfinite(b) and np.isfinite(c) and abs(a + c - 2 * b) > 1e-9:
            fy = y + 0.5 * (a - c) / (a - 2 * b + c)
    if 0 < x < score_map.shape[1] - 1:
        a, b, c = score_map[y, x - 1], score_map[y, x], score_map[y, x + 1]
        if np.isfinite(a) and np.isfinite(b) and np.isfinite(c) and abs(a + c - 2 * b) > 1e-9:
            fx = x + 0.5 * (a - c) / (a - 2 * b + c)
    return fy, fx


def _native_strip_peaks(scan_pixels, template, threshold):
    """Native-resolution NCC in overlapping vertical strips; returns (y, x, score)."""
    height = scan_pixels.shape[0]
    th, tw = template.shape
    step_y = max(MIN_SIDE, th // 2)
    step_x = max(MIN_SIDE, tw // 2)
    # Every fully contained window is valid, so the overlap only has to stop
    # windows from being split across the strip boundary.
    overlap = th - 1
    found = []
    top = 0
    while top < height:
        bottom = min(height, top + NATIVE_STRIP_ROWS)
        score_map = ncc_map(scan_pixels[top:bottom], template)
        if score_map is not None and np.isfinite(score_map).any():
            # Keep every local maximum inside the strip; global NMS runs afterwards
            # so a strip full of repeated perfs cannot push out the true position.
            found.extend((top + y, x, score)
                         for y, x, score in _peaks(score_map, threshold, step_y, step_x))
        if bottom == height:
            break
        top = bottom - overlap
    return _dedup_peaks(found, step_y, step_x)


def _coarse_peaks(scan_pixels, template, threshold):
    """Downsampled NCC followed by native refinement for very tall scans.

    The coarse stage only locates candidate cells, so it uses a deliberately
    lower threshold; native refinement re-scores each one at full resolution
    and the caller's real threshold decides whether a candidate survives.
    """
    factor = COARSE_FACTOR
    coarse_scan = _shrink(scan_pixels, factor)
    coarse_template = _shrink(template, factor)
    if coarse_template.shape[0] < MIN_SIDE or coarse_template.shape[1] < MIN_SIDE:
        return []
    score_map = ncc_map(coarse_scan, coarse_template)
    if score_map is None or not np.isfinite(score_map).any():
        return []
    step_y = max(MIN_SIDE, coarse_template.shape[0] // 2)
    step_x = max(MIN_SIDE, coarse_template.shape[1] // 2)
    found = []
    for cy, cx, _ in _peaks(score_map, COARSE_PEAK_THRESHOLD, step_y, step_x):
        fy, fx = _subpixel_peak(score_map, cy, cx)
        refined = _refine(scan_pixels, template, fy, fx, factor)
        if refined is not None and refined[2] > threshold:
            found.append(refined)
    return found


def _refine(scan, template, coarse_y, coarse_x, factor):
    """Recompute NCC at native resolution around a (subpixel) coarse peak."""
    height, width = scan.shape
    th, tw = template.shape
    center_y, center_x = coarse_y * factor, coarse_x * factor
    margin = (REFINE_RADIUS + 2) * factor
    y0 = max(0, int(center_y) - margin)
    x0 = max(0, int(center_x) - margin)
    y1 = min(height - th, int(center_y) + margin) + 1
    x1 = min(width - tw, int(center_x) + margin) + 1
    region = scan[y0:y1 + th - 1, x0:x1 + tw - 1]
    score_map = ncc_map(region, template)
    if score_map is None or not np.isfinite(score_map).any():
        return None
    iy, ix = np.unravel_index(np.nanargmax(np.nan_to_num(score_map, nan=-2.0)), score_map.shape)
    return y0 + int(iy), x0 + int(ix), float(score_map[iy, ix])


def search_scan(scan_id, scan_pixels, fragment, threshold):
    """Return every plausible crop position of ``fragment`` inside one scan."""
    height, width = scan_pixels.shape
    fh, fw = fragment.shape
    if fh > height or fw > width:
        return [], "fragment_larger"
    if float(fragment.std()) < MIN_TEMPLATE_STD:
        raise BenchError("查询图几乎没有纹理，无法用像素图案定位。请换一张包含声轨或齿孔变化的局部图。")
    use_coarse = height > COARSE_MAX_ROWS and fh // COARSE_FACTOR >= MIN_SIDE
    candidates = []
    for orientation, flipped in ORIENTATIONS:
        template = fragment[::-1] if flipped else fragment
        peaks = _coarse_peaks(scan_pixels, template, threshold) if use_coarse else \
            _native_strip_peaks(scan_pixels, template, threshold)
        for y, x, score in peaks:
            # Both paths finish at full-resolution pixel boxes; coarse is only a locator.
            candidates.append(TraceCandidate(scan_id, Box(x, y, x + fw, y + fh), float(score), orientation))
    candidates.sort(key=lambda item: item.score, reverse=True)
    return _suppress(candidates)[:MAX_CANDIDATES], None


def _suppress(candidates):
    # Same box but different orientation stays: each is a distinct hypothesis
    # for an archivist to check, not an automatic answer.
    kept = []
    for candidate in candidates:
        box = candidate.box
        if any(candidate.orientation == other.orientation
               and box.x0 - other.box.x1 < box.width // 2 and other.box.x0 - box.x1 < box.width // 2
               and box.y0 - other.box.y1 < box.height // 2 and other.box.y0 - box.y1 < box.height // 2
               for other in kept):
            continue
        kept.append(candidate)
    return kept


def trace_reel(catalog, reel_id, fragment, threshold):
    """Compare an in-memory fragment against every scan of one reel.

    The fragment is never inserted into the catalog.  All threshold-passing
    positions are returned so repeated perforations or reference tones do not
    collapse into a single automatic answer.
    """
    scans = catalog.list_scans(reel_id)
    results, skipped = [], []
    for record in scans:
        raw = catalog.get_scan(record["id"], raw=True)
        raster = read_scan(raw["raw"], record["metadata"]["filename"])
        candidates, reason = search_scan(record["id"], raster.intensity, fragment, threshold)
        if reason:
            skipped.append({"scan_id": record["id"], "title": record["title"], "reason": reason})
        for candidate in candidates:
            plate = trace_plate(fragment, candidate, raster.intensity, record["title"])
            results.append({"candidate": candidate.as_dict(), "scan_title": record["title"],
                            "plate": plate})
    results.sort(key=lambda item: item["candidate"]["score"], reverse=True)
    return results[:MAX_CANDIDATES], skipped


def _fit(bitmap, area):
    canvas = bitmap.copy()
    canvas.thumbnail(area, Image.Resampling.BOX)
    return canvas


def trace_plate(fragment, candidate, scan_pixels, scan_title):
    """Side-by-side plate: query fragment next to the candidate region in the scan."""
    box = candidate.box
    region = box.subimage(scan_pixels)
    paper = Image.new("RGB", (900, 470), PAGE)
    draw = ImageDraw.Draw(paper)
    print_text(draw, (22, 18), "REELTONE  /  FRAGMENT TRACE", 16, TEAL)
    print_text(draw, (22, 44), "Query fragment (not archived) vs candidate position in scan", 12, MUTED)
    card_w, card_h, top = 410, 330, 82
    for index, (pixels, tag, sub) in enumerate((
            (fragment, "QUERY FRAGMENT", f"{fragment.shape[1]} x {fragment.shape[0]} px"),
            (region, "CANDIDATE IN SCAN", f"native box {box.x0},{box.y0} .. {box.x1},{box.y1}"))):
        x0 = 24 + index * (card_w + 32)
        draw.rectangle((x0, top, x0 + card_w, top + card_h), fill=PANEL, outline=GRID)
        print_text(draw, (x0 + 12, top + 10), tag, 13, ACCENT if index else TEAL)
        bitmap = _fit(byte_preview(pixels), (card_w - 24, card_h - 78))
        px = x0 + (card_w - bitmap.width) // 2
        py = top + 44 + (card_h - 78 - bitmap.height) // 2
        paper.paste(bitmap.convert("RGB"), (px, py))
        draw.rectangle((px - 1, py - 1, px + bitmap.width, py + bitmap.height), outline=ACCENT, width=2)
        print_text(draw, (x0 + 12, top + card_h - 24), sub, 11, MUTED)
    note_y = top + card_h + 14
    details = (f"scan: {scan_title[:52]}   orientation: {candidate.orientation}   "
               f"NCC score: {candidate.score:.4f}   resolution: native pixels")
    print_text(draw, (24, note_y), details, 12, TEXT)
    print_text(draw, (24, note_y + 24),
               "Candidates are ranked but never auto-accepted; repeated perfs or tone can match several places.",
               11, MUTED)
    return encode_image(paper, "PNG")
