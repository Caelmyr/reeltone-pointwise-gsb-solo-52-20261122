"""Frame-level evidence cross checks.

A retained original-film crop (the native pixel companion of a saved crop proof,
or a native region of a source scan) is compared against a same-size image sent
back by a colleague. The returned image may be stored in another container
(TIFF), reduced from the archive 16-bit master to an 8-bit viewing copy, or
turned over (one of the eight dihedral orientations). Every orientation is
tested against the untouched native reference and the result is classified as:

* ``identical``     - pixels agree exactly (within lossless code tolerance);
* ``depth_only``    - differences are only those expected from a normal bit
                      depth reduction / lifting;
* ``different``     - an orientation aligns but identifiable pixel differences
                      remain, located in native source coordinates;
* ``inconclusive``  - no orientation can be aligned reliably.

The retained reference pixels and the archive source files are never modified.
"""
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import warnings

import numpy as np
from PIL import Image, ImageDraw, UnidentifiedImageError

from .contracts import safe_filename
from .errors import BenchError, Limit
from .geometry import Box, Orientation
from .raster import FORMATS, byte_preview, encode_image
from .visual import ACCENT, BLUE, GRID, MUTED, PAGE, RED, TEAL, TEXT, print_text


# (key, Orientation applied to the returned image to align it with native order)
ORIENTATIONS = (
    ("native", Orientation(False, False)),
    ("mirror_x", Orientation(True, False)),
    ("reverse_y", Orientation(False, True)),
    ("rotate_180", Orientation(True, True)),
    ("transpose", Orientation(False, False)),      # handled specially: transpose
    ("rotate_90_cw", Orientation(False, False)),   # handled specially: rot90
    ("rotate_90_ccw", Orientation(False, False)),
    ("transverse", Orientation(False, False)),
)
ORIENTATION_LABELS = {
    "native": "原方向（无需转向）",
    "mirror_x": "左右镜像",
    "reverse_y": "上下翻转",
    "rotate_180": "旋转 180°（上下左右同时颠倒）",
    "transpose": "主对角线转置（顺时针 90° 后上下翻转）",
    "rotate_90_cw": "顺时针旋转 90°",
    "rotate_90_ccw": "逆时针旋转 90°",
    "transverse": "副对角线转置",
}
# Orientation tag values 5..8 imply the pixel array itself is rotated 90/270.
ROTATED_TAGS = frozenset({5, 6, 7, 8})

VERDICT_IDENTICAL = "identical"
VERDICT_DEPTH_ONLY = "depth_only"
VERDICT_DIFFERENT = "different"
VERDICT_INCONCLUSIVE = "inconclusive"
VERDICT_LABELS = {
    VERDICT_IDENTICAL: "像素完全一致",
    VERDICT_DEPTH_ONLY: "与正常位深转换相符",
    VERDICT_DIFFERENT: "存在可以指认的差异",
    VERDICT_INCONCLUSIVE: "看不准，分开处理",
}
# A same-depth pair whose residual exceeds this cannot be a lossless copy.
EXACT_TOLERANCE = 1.5 / 65535.0
# Quantization floor of an 8-bit code expressed in normalized intensity.
EIGHT_BIT_STEP = 1.0 / 255.0
# Rounding half a code plus a small margin for container re-quantization.
DEPATH_TOLERANCE = 0.75 * EIGHT_BIT_STEP
# Above this normalized RMSE even a weak orientation match is rejected.
ALIGN_RMSE_LIMIT = 0.18
# A frame with almost no tonal modulation cannot disambiguate orientation.
FLAT_STD_LIMIT = 0.02
MAX_RETURNED_BYTES = 32 * 1024 * 1024
MAX_DIFFERENCE_REGIONS = 12


@dataclass(frozen=True)
class ReturnedImage:
    filename: str
    image_format: str
    bit_depth: int
    channel_count: int
    digest: str
    width: int
    height: int
    intensity: np.ndarray
    orientation_tag: int
    lossy: bool
    tag_applied: bool

    def describe(self):
        return {
            "filename": self.filename, "format": self.image_format, "bit_depth": self.bit_depth,
            "channel_count": self.channel_count, "sha256": self.digest,
            "width": self.width, "height": self.height, "lossy_container": self.lossy,
            "embedded_orientation_tag": self.orientation_tag,
            "embedded_orientation_note": ("采集端方向标签已在读取时按物理像素顺序归正"
                                          if self.tag_applied else "无需要归正的方向标签"),
        }


def _pixels(image):
    if "transparency" in image.info or image.mode in {"LA", "RGBA", "PA", "RGBa", "La"}:
        raise BenchError("核对图含有透明通道，无法与无透明度的原片裁片比较。请导出无透明度的图像。")
    if image.mode in {"I;16", "I;16L", "I;16B", "I;16N", "I"}:
        array = np.asarray(image)
        if array.dtype.kind not in "ui" or np.min(array) < 0 or np.max(array) > 65535:
            raise BenchError("仅支持 16 位无符号灰度图。")
        return np.asarray(array, dtype=np.float32) / 65535.0, 16, 1
    if image.mode in {"1", "L"}:
        return np.asarray(image.convert("L"), dtype=np.float32) / 255.0, 8, 1
    if image.mode in {"RGB", "P"}:
        source = np.asarray(image.convert("RGB"), dtype=np.float32)
        intensity = (source[:, :, 0] * 0.2126 + source[:, :, 1] * 0.7152
                     + source[:, :, 2] * 0.0722) / 255.0
        return intensity, 8, 3
    raise BenchError("核对图须为标准 8 位 RGB、8 位灰度或 16 位灰度图像。")


def read_returned(data, name):
    if not data:
        raise BenchError("请选择同事回传的同尺寸图像。")
    if len(data) > MAX_RETURNED_BYTES:
        raise Limit(f"核对图最多 {MAX_RETURNED_BYTES // (1024 * 1024)} MiB。")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                image_format = image.format
                if image_format not in FORMATS:
                    raise BenchError("核对图接收 PNG、单页 TIFF 或 JPEG。")
                if getattr(image, "n_frames", 1) != 1:
                    raise BenchError("多页 TIFF 请逐页导出后再核对。")
                orientation_tag = int(image.getexif().get(274, 1) or 1)
                tag_applied = False
                if image_format == "TIFF" and orientation_tag in {2, 3, 4}:
                    # Keep physical pixel order, mirroring the intake pipeline.
                    image.getexif()[274] = 1
                    image.tag_v2[274] = 1
                    tag_applied = True
                intensity, depth, channels = _pixels(image)
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise Limit("核对图超过解码安全上限。") from None
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise BenchError("读取核对图失败，请重新导出该图像。") from None
    height, width = intensity.shape
    lossy = image_format == "JPEG"
    return ReturnedImage(safe_filename(name, "returned.tiff"), image_format, depth, channels,
                         sha256(data).hexdigest(), width, height, intensity, orientation_tag,
                         lossy, tag_applied)


def _turn(intensity, key):
    if key == "native":
        return intensity
    if key == "mirror_x":
        return intensity[:, ::-1]
    if key == "reverse_y":
        return intensity[::-1, :]
    if key == "rotate_180":
        return intensity[::-1, ::-1]
    if key == "transpose":
        return intensity.T
    if key == "rotate_90_cw":
        return np.rot90(intensity, -1)
    if key == "rotate_90_ccw":
        return np.rot90(intensity, 1)
    if key == "transverse":
        return np.fliplr(intensity.T)
    raise BenchError("未知的转向方式。")


def _candidate_keys(returned, reference_shape):
    """Only orientations whose turned array has the reference dimensions fit."""
    keys = []
    for key, _ in ORIENTATIONS:
        turned = _turn(returned.intensity, key)
        if turned.shape == reference_shape:
            keys.append(key)
    return keys


def _residual_stats(reference, candidate):
    delta = candidate.astype(np.float64) - reference.astype(np.float64)
    absolute = np.abs(delta)
    return {
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "mean_absolute": float(absolute.mean()),
        "max_absolute": float(absolute.max()),
        "q95_absolute": float(np.percentile(absolute, 95)),
        "fraction_over_exact_tolerance": float(np.mean(absolute > EXACT_TOLERANCE)),
        "fraction_over_depth_tolerance": float(np.mean(absolute > DEPATH_TOLERANCE)),
    }


def _difference_regions(reference, candidate, native_box, orientation_key):
    """Cluster pixels exceeding depth-conversion tolerance into native boxes."""
    absolute = np.abs(candidate.astype(np.float64) - reference.astype(np.float64))
    mask = absolute > DEPATH_TOLERANCE
    if not mask.any():
        return []
    # Downstream plates only need coarse regions; cap work on large frames.
    height, width = mask.shape
    cell = max(1, int(np.ceil(max(height, width) / 220)))
    if cell > 1:
        trimmed_h = height - height % cell
        trimmed_w = width - width % cell
        pooled = mask[:trimmed_h, :trimmed_w].reshape(
            trimmed_h // cell, cell, trimmed_w // cell, cell).any(axis=(1, 3))
    else:
        pooled = mask
    rows, cols = np.nonzero(pooled)
    if len(rows) == 0:
        return []
    # Label connected cells with a simple two-pass union-find on the grid.
    labels = np.full(pooled.shape, -1, dtype=np.int32)
    parents = []

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(a, b):
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parents[max(root_a, root_b)] = min(root_a, root_b)

    for row, col in zip(rows, cols):
        neighbor_labels = []
        for dy, dx in ((-1, 0), (0, -1), (-1, -1), (-1, 1)):
            ny, nx = row + dy, col + dx
            if 0 <= ny < pooled.shape[0] and 0 <= nx < pooled.shape[1] and labels[ny, nx] >= 0:
                neighbor_labels.append(find(labels[ny, nx]))
        if not neighbor_labels:
            new_label = len(parents)
            parents.append(new_label)
            labels[row, col] = new_label
        else:
            root = min(neighbor_labels)
            labels[row, col] = root
            for other in neighbor_labels:
                union(root, other)

    components = {}
    for row, col in zip(rows, cols):
        root = find(labels[row, col])
        components.setdefault(root, []).append((row, col))
    regions = []
    for cells in components.values():
        cell_rows = np.array([item[0] for item in cells])
        cell_cols = np.array([item[1] for item in cells])
        gy0, gx0 = cell_rows.min() * cell, cell_cols.min() * cell
        gy1, gx1 = min(height, (cell_rows.max() + 1) * cell), min(width, (cell_cols.max() + 1) * cell)
        local = Box(gx0, gy0, gx1, gy1)
        native = _native_box(local, width, height, orientation_key)
        absolute_local = absolute[gy0:gy1, gx0:gx1]
        # Worst pixel inside the component, mapped back to native coordinates.
        worst_flat = int(np.argmax(absolute_local))
        wy, wx = np.unravel_index(worst_flat, absolute_local.shape)
        native_point = _native_point(gx0 + wx, gy0 + wy, width, height, orientation_key)
        regions.append({
            "native_box": [int(native_box.x0 + native.x0), int(native_box.y0 + native.y0),
                           int(native_box.x0 + native.x1), int(native_box.y0 + native.y1)],
            "native_worst_point": [int(native_box.x0 + native_point[0]),
                                   int(native_box.y0 + native_point[1])],
            "cell_count": int(len(cells)),
            "pixel_count_estimate": int(mask[gy0:gy1, gx0:gx1].sum()),
            "max_absolute_delta": round(float(absolute_local.max()), 6),
        })
    regions.sort(key=lambda item: item["max_absolute_delta"], reverse=True)
    return regions[:MAX_DIFFERENCE_REGIONS]


def _native_box(local: Box, width, height, orientation_key):
    """Convert a box in aligned (native-ordered) candidate coordinates back."""
    points = [(local.x0, local.y0), (local.x1 - 1, local.y0), (local.x0, local.y1 - 1),
              (local.x1 - 1, local.y1 - 1)]
    turned = [_native_point(x, y, width, height, orientation_key) for x, y in points]
    xs = [point[0] for point in turned]
    ys = [point[1] for point in turned]
    return Box(min(xs), min(ys), max(xs) + 1, max(ys) + 1)


def _native_point(x, y, width, height, orientation_key):
    """Inverse of _turn for a single point of the candidate array."""
    if orientation_key == "native":
        return x, y
    if orientation_key == "mirror_x":
        return width - 1 - x, y
    if orientation_key == "reverse_y":
        return x, height - 1 - y
    if orientation_key == "rotate_180":
        return width - 1 - x, height - 1 - y
    if orientation_key == "transpose":
        return y, x
    if orientation_key == "rotate_90_cw":
        return height - 1 - y, x
    if orientation_key == "rotate_90_ccw":
        return y, width - 1 - x
    if orientation_key == "transverse":
        return height - 1 - y, width - 1 - x
    raise BenchError("未知的转向方式。")


def _classify(reference, candidate, returned, stats, reference_depth):
    depth_reduced = returned.bit_depth < reference_depth
    if stats["fraction_over_exact_tolerance"] == 0 and stats["max_absolute"] <= EXACT_TOLERANCE:
        return VERDICT_IDENTICAL
    if depth_reduced and stats["fraction_over_depth_tolerance"] == 0:
        return VERDICT_DEPTH_ONLY
    if returned.lossy:
        # Lossy containers spread small transform artifacts over the whole
        # frame. A tight residual still only demonstrates container effects;
        # a larger residual cannot be separated from a genuinely different
        # frame by pixel comparison and is left inconclusive.
        if stats["rmse"] <= DEPATH_TOLERANCE * 1.6 and stats["fraction_over_depth_tolerance"] <= 0.004:
            return VERDICT_DEPTH_ONLY
        if stats["rmse"] <= 0.05:
            return VERDICT_INCONCLUSIVE
    if stats["rmse"] > ALIGN_RMSE_LIMIT:
        return VERDICT_INCONCLUSIVE
    return VERDICT_DIFFERENT


@dataclass(frozen=True)
class FrameCheck:
    verdict: str
    orientation_key: str | None
    orientation_label: str | None
    reference: dict
    returned: dict
    candidates: list
    stats: dict
    regions: list
    interpretation: str
    primary_png: bytes
    secondary_png: bytes

    def reading(self):
        return {
            "verdict": self.verdict,
            "verdict_label": VERDICT_LABELS[self.verdict],
            "matched_orientation": self.orientation_key,
            "matched_orientation_label": self.orientation_label,
            "reference": self.reference,
            "returned_image": self.returned,
            "orientation_candidates": self.candidates,
            "alignment_statistics": self.stats,
            "difference_regions_native": self.regions,
            "interpretation": self.interpretation,
        }


def frame_check(reference_pixels, reference_meta, native_box, returned: ReturnedImage):
    reference_pixels = np.asarray(reference_pixels, dtype=np.float32)
    ref_h, ref_w = reference_pixels.shape
    if (returned.height, returned.width) not in {(ref_h, ref_w), (ref_w, ref_h)}:
        raise BenchError(
            f"回传图像为 {returned.width} × {returned.height}，原片裁片为 {ref_w} × {ref_h}。"
            "请上传与原片框同尺寸（含整体转 90°）的图像。")
    keys = _candidate_keys(returned, reference_pixels.shape)
    if not keys:
        raise BenchError("没有任何转向能让回传图像与原片裁片尺寸对齐。")
    candidate_rows = []
    for key in keys:
        candidate = _turn(returned.intensity, key)
        stats = _residual_stats(reference_pixels, candidate)
        candidate_rows.append({"key": key, "label": ORIENTATION_LABELS[key], **stats})
    candidate_rows.sort(key=lambda row: (row["rmse"], row["mean_absolute"]))
    best = candidate_rows[0]
    aligned = _turn(returned.intensity, best["key"])
    flat = float(np.std(reference_pixels)) <= FLAT_STD_LIMIT
    runner_up = candidate_rows[1]["rmse"] if len(candidate_rows) > 1 else best["rmse"]
    # A reliable orientation match needs the best alignment to clearly beat the
    # next one; near-ties mean the frame content cannot disambiguate direction.
    orientation_clear = (runner_up - best["rmse"]) > 0.25 * max(best["rmse"], 1e-6)
    ambiguous = flat or not orientation_clear
    verdict = _classify(reference_pixels, aligned, returned, best, reference_meta["bit_depth"])
    if ambiguous and verdict == VERDICT_DIFFERENT:
        verdict = VERDICT_INCONCLUSIVE
    regions = (_difference_regions(reference_pixels, aligned, native_box, best["key"])
               if verdict == VERDICT_DIFFERENT else [])
    if verdict == VERDICT_INCONCLUSIVE:
        orientation_key, orientation_label = None, None
        interpretation = ("八种转向均无法与原片框可靠对齐，或画面本身过于平坦无法区分方向。"
                          "不要把该回传图作为同一原片框的证据，应分开留存并人工复查。")
    else:
        orientation_key, orientation_label = best["key"], ORIENTATION_LABELS[best["key"]]
        interpretation = {
            VERDICT_IDENTICAL: "在对正的朝向下，回传图与原片裁片的归一化像素逐点一致，可作为同一原片框的证据。",
            VERDICT_DEPTH_ONLY: ("全部像素差都落在一次正常位深转换的量化范围内，"
                                 "没有超出转换误差的可指认差异，可作为同一原片框的查看稿证据。"),
            VERDICT_DIFFERENT: ("画面能够对正，但存在超出位深转换量化范围的像素差异，"
                                "差异位置已按未翻转的原片坐标标出，不能直接视为同一原片框的无损证据。"),
        }[verdict]
    reference_info = {
        "source_sha256": reference_meta["source_sha256"], "scan_id": reference_meta["scan_id"],
        "title": reference_meta["title"], "native_box": native_box.as_list(),
        "bit_depth": reference_meta["bit_depth"], "kind": reference_meta["kind"],
        "proof_id": reference_meta.get("proof_id"),
    }
    check = FrameCheck(verdict, orientation_key, orientation_label, reference_info, returned.describe(),
                       candidate_rows, best, regions, interpretation,
                       evidence_plate(reference_pixels, aligned, native_box, best["key"], verdict,
                                      reference_info, returned.describe(), regions),
                       difference_map(reference_pixels, aligned, native_box, best["key"], regions))
    return check


# ---------------------------------------------------------------------------
# Evidence rendering
# ---------------------------------------------------------------------------

def _fit(values, width_cap, height_cap):
    if values.ndim == 3:
        image = Image.fromarray(np.asarray(values, dtype=np.uint8), mode="RGB")
    else:
        image = byte_preview(values)
    scale = min(width_cap / image.width, height_cap / image.height, 1.0)
    if scale < 1.0:
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                             Image.Resampling.BOX)
    return image.convert("RGB")


def evidence_plate(reference, candidate, native_box, orientation_key, verdict, reference_info, returned_info, regions):
    width_cap, height_cap = 300, 430
    ref_view = _fit(reference, width_cap, height_cap)
    ret_view = _fit(candidate, width_cap, height_cap)
    page = Image.new("RGB", (980, 720), PAGE)
    draw = ImageDraw.Draw(page)
    print_text(draw, (26, 24), "REELTONE  /  FRAME EVIDENCE CROSS CHECK", 18, TEAL)
    verdict_color = {"identical": TEAL, "depth_only": BLUE, "different": RED,
                     "inconclusive": ACCENT}[verdict]
    print_text(draw, (26, 56), VERDICT_LABELS[verdict], 22, verdict_color)
    positions = (("ARCHIVE NATIVE CROP", (36, 108), ref_view),
                 ("RETURNED · ALIGNED", (356, 108), ret_view))
    for caption, (x, y), view in positions:
        print_text(draw, (x, y - 24), caption, 13, MUTED)
        page.paste(view, (x, y))
        draw.rectangle((x - 2, y - 2, x + view.width + 1, y + view.height + 1), outline=GRID)
    # Difference panel on the right.
    diff = np.clip(np.abs(candidate.astype(np.float64) - reference.astype(np.float64)), 0, 1)
    heat = np.zeros((*diff.shape, 3), dtype=np.uint8)
    heat[:, :, 0] = np.rint(60 + diff * 195).astype(np.uint8)
    heat[:, :, 1] = np.rint(58 + (1 - diff) * 60).astype(np.uint8)
    heat[:, :, 2] = np.rint(60 + (1 - diff) * 90).astype(np.uint8)
    diff_view = _fit(heat, width_cap, height_cap)
    dx, dy = 676, 108
    print_text(draw, (dx, dy - 24), "ABSOLUTE DIFFERENCE", 13, MUTED)
    page.paste(diff_view, (dx, dy))
    draw.rectangle((dx - 2, dy - 2, dx + diff_view.width + 1, dy + diff_view.height + 1), outline=GRID)
    # Mark difference regions in native coordinates on both image panels.
    for panel_x, panel_y, view in ((36, 108, ref_view), (356, 108, ret_view)):
        for index, region in enumerate(regions[:6], start=1):
            bx = region["native_box"]
            local = Box(bx[0] - native_box.x0, bx[1] - native_box.y0,
                        bx[2] - native_box.x0, bx[3] - native_box.y0)
            scale_x = view.width / reference.shape[1]
            scale_y = view.height / reference.shape[0]
            rectangle = (panel_x + local.x0 * scale_x, panel_y + local.y0 * scale_y,
                         panel_x + local.x1 * scale_x, panel_y + local.y1 * scale_y)
            draw.rectangle(tuple(round(v) for v in rectangle), outline=ACCENT, width=2)
            print_text(draw, (rectangle[0] + 3, rectangle[1] + 1), index, 11, TEXT)
    sidebar_y = 560
    lines = [
        ("ORIENTATION", ORIENTATION_LABELS[orientation_key]),
        ("NATIVE BOX", " ".join(str(v) for v in native_box.as_list())),
        ("REFERENCE", f"{reference_info['bit_depth']} bit · {reference_info['source_sha256'][:16]}…"),
        ("RETURNED", f"{returned_info['format']} · {returned_info['bit_depth']} bit · {returned_info['filename']}"),
        ("RETURN SHA256", returned_info["sha256"][:24] + "…"),
    ]
    for label, value in lines:
        print_text(draw, (36, sidebar_y), label, 12, MUTED)
        print_text(draw, (190, sidebar_y), str(value)[:78], 13)
        sidebar_y += 26
    if regions:
        print_text(draw, (36, sidebar_y + 4), "差异框按未翻转原片坐标给出，序号与读数 JSON 一致。", 12, ACCENT)
    return encode_image(page, "PNG")


def difference_map(reference, candidate, native_box, orientation_key, regions):
    diff = np.abs(candidate.astype(np.float64) - reference.astype(np.float64))
    amplified = np.clip(diff / max(DEPATH_TOLERANCE, 1e-9), 0, 1)
    heat = np.zeros((*diff.shape, 3), dtype=np.uint8)
    heat[:, :, 0] = np.rint(40 + amplified * 215).astype(np.uint8)
    heat[:, :, 1] = np.rint(40 + (1 - amplified) * 120).astype(np.uint8)
    heat[:, :, 2] = np.rint(48 + (1 - amplified) * 150).astype(np.uint8)
    image = Image.fromarray(heat)
    image = image.resize((min(1200, max(240, image.width)), min(900, max(180, image.height))),
                         Image.Resampling.NEAREST)
    page = Image.new("RGB", (image.width + 48, image.height + 96), PAGE)
    draw = ImageDraw.Draw(page)
    print_text(draw, (22, 18), "DIFFERENCE HEAT MAP · NATIVE FILM ORDER", 16, TEAL)
    print_text(draw, (22, 44), f"orientation: {ORIENTATION_LABELS[orientation_key]} · native box "
                               f"{native_box.as_list()}", 12, MUTED)
    page.paste(image.convert("RGB"), (24, 72))
    return encode_image(page, "PNG")
