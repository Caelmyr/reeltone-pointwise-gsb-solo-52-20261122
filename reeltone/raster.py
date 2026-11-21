from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import warnings

import numpy as np
from PIL import Image, UnidentifiedImageError

from .contracts import integer, safe_filename
from .errors import BenchError, Limit
from .source_info import SourceEncoding, collect_source_encoding, estimate_native_dimensions, source_notices


FILE_BYTES = 32 * 1024 * 1024
MAX_PIXELS = 14_000_000
MAX_WIDTH = 2048
MAX_HEIGHT = 44_000
MIN_WIDTH = 24
MIN_HEIGHT = 100
FORMATS = {"PNG": ("image/png", "png"), "TIFF": ("image/tiff", "tiff"), "JPEG": ("image/jpeg", "jpg")}


@dataclass(frozen=True)
class Raster:
    width: int
    height: int
    filename: str
    image_format: str
    mime: str
    bit_depth: int
    channel_count: int
    digest: str
    raw: bytes
    intensity: np.ndarray
    encoding: SourceEncoding

    def metadata(self):
        return {
            "width": self.width,
            "height": self.height,
            "filename": self.filename,
            "format": self.image_format,
            "mime": self.mime,
            "bit_depth": self.bit_depth,
            "channel_count": self.channel_count,
            "sha256": self.digest,
            "file_size": len(self.raw),
            "encoding": self.encoding.describe(),
            "reported_dimensions": estimate_native_dimensions(self.encoding, self.width, self.height),
            "notices": source_notices(self.encoding, self.bit_depth, self.width, self.height),
        }


def scan_dimensions(width, height):
    if width < MIN_WIDTH or height < MIN_HEIGHT:
        raise BenchError(f"声轨边带至少需要 {MIN_WIDTH} 列 × {MIN_HEIGHT} 行。")
    if width > MAX_WIDTH or height > MAX_HEIGHT or width * height > MAX_PIXELS:
        raise Limit(f"图像最多 {MAX_WIDTH} 列、{MAX_HEIGHT} 行、{MAX_PIXELS:,} 像素。")


def intensity_pixels(image):
    if "transparency" in image.info or image.mode in {"LA", "RGBA", "PA", "RGBa", "La"}:
        raise BenchError("透明度会改变胶片透射值。请提交已检测且无透明度的扫描。")
    if image.mode in {"I;16", "I;16L", "I;16B", "I;16N", "I"}:
        array = np.asarray(image)
        if array.dtype.kind not in "ui" or np.min(array) < 0 or np.max(array) > 65535:
            raise BenchError("仅支持 16 位无符号灰度图。")
        return np.asarray(array, dtype=np.float32) / 65535.0, 16, 1
    if image.mode in {"1", "L"}:
        return np.asarray(image.convert("L"), dtype=np.float32) / 255.0, 8, 1
    if image.mode == "RGB" or image.mode == "P":
        source = np.asarray(image.convert("RGB"), dtype=np.float32)
        intensity = (source[:, :, 0] * 0.2126 + source[:, :, 1] * 0.7152 + source[:, :, 2] * 0.0722) / 255.0
        return intensity, 8, 3
    raise BenchError("扫描须为标准独立 8 位 RGB、8 位灰度或 16 位灰度图像。")


def read_scan(data, name):
    if not data:
        raise BenchError("请选择声轨边带图像。")
    if len(data) > FILE_BYTES:
        raise Limit("每次导入最多 32 MiB。")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                image_format = image.format
                if image_format not in FORMATS:
                    raise BenchError("接收 PNG、单页 TIFF 或 JPEG。")
                width, height = image.size
                scan_dimensions(width, height)
                if getattr(image, "n_frames", 1) != 1:
                    raise BenchError("请选择单张影像，多页 TIFF 须逐页导出。")
                encoding = collect_source_encoding(image, data)
                exif_orientation = encoding.orientation
                if exif_orientation in {5, 6, 7, 8}:
                    raise BenchError("请在扫描软件中将声轨长度方向沿行展开后重存。")
                if image_format == "TIFF" and exif_orientation in {2, 3, 4}:
                    # Preserve the physical pixel order while retaining the original tag in SourceEncoding.
                    image.getexif()[274] = 1
                    image.tag_v2[274] = 1
                intensity, depth, channels = intensity_pixels(image)
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise Limit("图像超过解码安全上限。") from None
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        raise BenchError("读取扫描失败，请重新导出该图像。") from None
    mime, extension = FORMATS[image_format]
    return Raster(width, height, safe_filename(name, f"scan.{extension}"), image_format, mime,
                  depth, channels, sha256(data).hexdigest(), bytes(data), intensity, encoding)


def encode_png(values, bits=8):
    if bits == 16:
        pixels = np.rint(np.clip(values, 0.0, 1.0) * 65535.0).astype("<u2")
        image = Image.fromarray(pixels)
    else:
        pixels = np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)
        image = Image.fromarray(pixels)
    return encode_image(image, "PNG")


def encode_image(image, format):
    output = BytesIO()
    if format == "PNG":
        image.save(output, format, compress_level=4)
    elif format == "JPEG":
        image.convert("RGB").save(output, format, quality=88, optimize=True)
    else:
        image.save(output, format)
    return output.getvalue()


def byte_preview(values, *, lo=0, hi=1, inverted=False):
    stretch = np.clip((values - lo) / max(hi - lo, 0.0001), 0.0, 1.0)
    if inverted:
        stretch = 1.0 - stretch
    return Image.fromarray(np.rint(stretch * 255).astype(np.uint8), mode="L")


def fitted_preview(image, cap_width, cap_height, *, filter=Image.Resampling.BOX):
    canvas = image.copy()
    canvas.thumbnail((cap_width, cap_height), filter)
    return canvas


def aggregated_rows(values, size=600):
    rows = values.shape[0]
    if rows <= size:
        limits = np.column_stack((np.arange(rows), np.arange(1, rows + 1)))
        return values.copy(), limits
    edges = np.linspace(0, rows, size + 1, dtype=np.int64)
    cumulative = np.concatenate((np.zeros((1, *values.shape[1:])), np.cumsum(values, axis=0, dtype=np.float64)))
    sums = cumulative[edges[1:]] - cumulative[edges[:-1]]
    lengths = np.diff(edges).reshape((-1,) + (1,) * (values.ndim - 1))
    return sums / lengths, np.column_stack((edges[:-1], edges[1:]))


def query_box(args, width, height, limit):
    from .geometry import Box

    left = integer(args.get("x", 0), "左列", 0, width - 1)
    top = integer(args.get("y", 0), "首行", 0, height - 1)
    right = integer(args.get("right", width), "右边界", left + 1, width)
    bottom = integer(args.get("bottom", min(height, top + limit)), "末行之后", top + 1, height)
    return Box(left, top, right, bottom)
