from dataclasses import dataclass
from hashlib import sha256
import math
import struct

from .errors import BenchError


ORIENTATION = {
    1: "standard_native", 2: "horizontal_flip", 3: "rotate_180", 4: "vertical_flip",
    5: "transpose", 6: "rotate_90_cw", 7: "transverse", 8: "rotate_90_ccw",
}
TIFF_LABELS = {1: "uncompressed", 5: "lzw", 7: "jpeg", 8: "deflate", 32773: "packbits", 32946: "deflate"}
PNG_COLORS = {0: "gray", 2: "truecolor", 3: "palette", 4: "gray_alpha", 6: "truecolor_alpha"}


@dataclass(frozen=True)
class SourceEncoding:
    format: str
    orientation: int
    source_bits: tuple[int, ...]
    channel_model: str
    compressed: str | None
    dpi: tuple[float, float] | None
    transfer: dict
    color_profile: str | None
    container_details: dict

    def describe(self):
        return {"format": self.format, "embedded_orientation": ORIENTATION.get(self.orientation, "unknown"),
                "orientation_tag": self.orientation, "source_bits": list(self.source_bits), "channel_model": self.channel_model,
                "compression": self.compressed, "metadata_dpi": list(self.dpi) if self.dpi else None,
                "transfer_tags": self.transfer, "icc_profile_sha256": self.color_profile, "container": self.container_details}


def _num(value, *, low=0.0001, high=900000):
    if isinstance(value, (tuple, list)) and len(value) == 2:
        try:
            ratio = float(value[0]) / float(value[1])
        except (ValueError, TypeError, ZeroDivisionError, OverflowError):
            return None
        return ratio if math.isfinite(ratio) and low <= ratio <= high else None
    try:
        numeric = float(value)
    except (ValueError, TypeError, ZeroDivisionError, OverflowError):
        return None
    if not math.isfinite(numeric) or not low <= numeric <= high:
        return None
    return numeric


def _first_int(value, fallback):
    if isinstance(value, (tuple, list)):
        value = value[0] if value else None
    if isinstance(value, int):
        return int(value)
    return fallback


def _bits(field, fallback):
    if isinstance(field, int):
        return (field,)
    if isinstance(field, (list, tuple)) and 1 <= len(field) <= 8:
        return tuple(x if isinstance(x, int) else fallback[0] for x in field)
    return fallback


def _scale_dpi(info, tags):
    values = info.get("dpi")
    if isinstance(values, (list, tuple)) and len(values) == 2:
        x, y = [_num(value) for value in values]
        if x and y:
            return round(x, 3), round(y, 3)
    if tags:
        unit = _first_int(tags.get(296), 1)
        if unit in (2, 3):
            x, y = _num(tags.get(282)), _num(tags.get(283))
            if x and y:
                factor = 2.54 if unit == 3 else 1.0
                return round(x * factor, 3), round(y * factor, 3)
    return None


def png_fields(data):
    if len(data) < 33 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise BenchError("PNG 头不完整。")
    length = struct.unpack(">I", data[8:12])[0]
    if length != 13:
        raise BenchError("PNG 文件头与像素读入信息不一致。")
    w, h, code_depth, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", data[16:29])
    count = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(color, 0)
    if count == 0 or compression != 0 or filtering != 0 or interlace not in (0, 1):
        raise BenchError("不支持该 PNG 颜色编码。")
    return (code_depth,) * count, PNG_COLORS[color], "deflate", {"interlaced": bool(interlace), "width": w, "height": h}


def tiff_fields(image):
    tags = image.tag_v2
    bits = _bits(tags.get(258), (16,) if image.mode.startswith("I") else (8,))
    photometric = _first_int(tags.get(262), -1)
    planar = _first_int(tags.get(284), 1)
    compressor = _first_int(tags.get(259), 1)
    format_code = _first_int(tags.get(339), 1)
    mono_map = {0: "white_is_zero", 1: "black_is_zero", 2: "rgb", 3: "palette", 5: "separated", 6: "ycbcr"}
    details = {"photometric": mono_map.get(photometric, "unknown"), "planar": planar,
               "sample_format": format_code, "tiled": 322 in tags and 323 in tags,
               "image_planes": _first_int(tags.get(277), len(bits))}
    if details["sample_format"] != 1 and image.mode not in {"L", "RGB", "P", "1"}:
        raise BenchError("TIFF 图像须包含无符号整数扫描码。")
    return bits, image.mode, TIFF_LABELS.get(compressor, f"compression_{compressor}"), details


def jpeg_fields(image):
    details = {"progressive": bool(image.info.get("progressive") or image.info.get("progression")),
               "components": image.layers if hasattr(image, "layers") else len(image.getbands())}
    subsampling = getattr(image, "layer", ())
    if subsampling and len(subsampling) <= 4:
        details["sampling_factors"] = [[int(item[1]), int(item[2])] for item in subsampling if len(item) == 4]
    return (8,) * len(image.getbands()), image.mode, "jpeg", details


def collect_source_encoding(image, original):
    container = image.format
    if container == "PNG":
        bits, model, compressor, details = png_fields(original)
    elif container == "TIFF":
        bits, model, compressor, details = tiff_fields(image)
    else:
        bits, model, compressor, details = jpeg_fields(image)
    if len(bits) > 1 and max(bits) > 8:
        raise BenchError("彩色 16 位输入须先在扫描流程中转换为灰度 16 位或 RGB 8 位。")
    orientation = _first_int(image.getexif().get(274, 1), 1)
    profile = image.info.get("icc_profile")
    checksum = sha256(profile).hexdigest() if isinstance(profile, bytes) and profile else None
    transfer = {}
    gamma = _num(image.info.get("gamma"), high=5)
    if gamma:
        transfer["png_gamma"] = round(gamma, 7)
    if "srgb" in image.info:
        transfer["srgb_intent"] = _first_int(image.info["srgb"], -1)
    density = _scale_dpi(image.info, getattr(image, "tag_v2", None))
    return SourceEncoding(container, orientation, bits, model, compressor, density, transfer, checksum, details)


def estimate_native_dimensions(encoding, width, height):
    if not encoding.dpi:
        return None
    return {"width_mm": round(25.4 * width / encoding.dpi[0], 5), "length_mm": round(25.4 * height / encoding.dpi[1], 5),
            "basis": "embedded_dpi_only"}


def source_notices(encoding, bits, width, height):
    notices = []
    if encoding.format == "JPEG" or encoding.compressed == "jpeg":
        notices.append("有损 JPEG 可能让声轨边缘附近出现轻微纹波，保留源文件便于复核。")
    if encoding.orientation != 1:
        notices.append("图像保留原始像素顺序。需要时可在读片设置里确认镜像和走片方向。")
    if encoding.transfer or encoding.color_profile:
        notices.append("色彩标签被记录为来源证据。扫描码值按文件像素读取，无自动色彩空间转换。")
    if bits < 16:
        notices.append("工作像素解析度低于 16 位。二值化、阈值和光学曲线须结合原分辨率检查。")
    if width / height > 0.75:
        notices.append("图像偏短。请确认走片方向沿行延伸且已保留足够的连续声轨。")
    return notices


def inspect_source(material, sample_columns=80):
    import numpy as np

    metadata = material.raster.metadata()
    values = material.pixels
    row_step = max(1, values.shape[0] // 2200)
    stride = values[::row_step]
    x_positions = np.linspace(0, values.shape[1], min(sample_columns, values.shape[1]) + 1, dtype=int)
    column_profile = []
    for x, right in zip(x_positions[:-1], x_positions[1:]):
        data = stride[:, x:right]
        level_min, level_center, level_max = np.percentile(data, [5, 50, 95])
        column_profile.append({"columns": [int(x), int(right)], "shadow": round(float(level_min), 6),
                               "midpoint": round(float(level_center), 6), "highlight": round(float(level_max), 6)})
    witnesses = []
    positions = np.linspace(0, values.shape[0] - 1, min(4, values.shape[0]), dtype=int)
    for row in positions:
        codes = np.rint(values[row] * ((1 << material.raster.bit_depth) - 1)).astype("<u2")
        witnesses.append({"native_row": int(row), "normalized_codes_sha256": sha256(codes.tobytes()).hexdigest(),
                          "normalized_first_code": int(codes[0]), "normalized_last_code": int(codes[-1]),
                          "normalized_min_code": int(codes.min()), "normalized_max_code": int(codes.max())})
    return {"parent": material.identity(), "metadata": metadata, "native_order_preserved": True,
            "intrinsic_width": values.shape[1], "intrinsic_height": values.shape[0], "sampling_step": row_step,
            "normalized_channels": 1, "profile": column_profile, "witness_rows": witnesses}
