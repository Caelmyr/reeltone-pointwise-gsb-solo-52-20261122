import math
from pathlib import PurePath
import re
import unicodedata

from .errors import BenchError


ID_RE = re.compile(r"[0-9a-f]{32}\Z")
SINGLE_LINE = re.compile(r"[\r\n\t]+")


def mapping(value, name="提交数据"):
    if not isinstance(value, dict):
        raise BenchError(f"{name}必须是一个对象。")
    return value


def sequence(value, name, *, minimum=0, maximum=1000):
    if not isinstance(value, (list, tuple)) or not minimum <= len(value) <= maximum:
        raise BenchError(f"{name}须包含 {minimum}–{maximum} 项。")
    return value


def text(value, name, *, maximum=120, required=True, multiline=False):
    if not isinstance(value, str):
        raise BenchError(f"{name}必须是文字。")
    cleaned = value.strip()
    if not multiline:
        cleaned = SINGLE_LINE.sub(" ", cleaned)
    if any(0 <= ord(character) < 32 and character not in "\r\n\t" for character in cleaned):
        raise BenchError(f"{name}含有无效字符。")
    if len(cleaned) > maximum or (required and not cleaned):
        raise BenchError(f"{name}须为{' 1–' if required else '不超过 '}{maximum} 字。")
    return cleaned


def integer(value, name, lo, hi):
    if isinstance(value, bool):
        raise BenchError(f"{name}须为整数。")
    if isinstance(value, str):
        if not re.fullmatch(r"[+-]?\d{1,12}", value.strip()):
            raise BenchError(f"{name}须为整数。")
        value = int(value)
    if not isinstance(value, int) or not lo <= value <= hi:
        raise BenchError(f"{name}须为 {lo}–{hi} 的整数。")
    return value


def real(value, name, lo, hi):
    if isinstance(value, bool):
        raise BenchError(f"{name}须为数字。")
    try:
        value = float(value)
    except (ValueError, TypeError, OverflowError):
        raise BenchError(f"{name}须为数字。") from None
    if not math.isfinite(value) or not lo <= value <= hi:
        raise BenchError(f"{name}须在 {lo:g}–{hi:g} 范围内。")
    return value


def option(value, values, name):
    if not isinstance(value, str) or value not in values:
        raise BenchError(f"{name}须为 {' / '.join(values)}。")
    return value


def flag(value, name):
    if not isinstance(value, bool):
        raise BenchError(f"{name}须为布尔值。")
    return value


def identifier(value, name="记录编号"):
    if not isinstance(value, str) or not ID_RE.fullmatch(value):
        raise BenchError(f"{name}格式无效。")
    return value


def safe_filename(value, fallback="scan"):
    leaf = PurePath(str(value).replace("\\", "/")).name
    cleaned = unicodedata.normalize("NFKC", leaf)
    cleaned = "".join(x for x in cleaned if x.isalnum() or x in ".-_ ()[]")
    return cleaned.strip(". ")[:100] or fallback


def shipment_stem(title, fallback="reeltone"):
    candidate = safe_filename(title, fallback)
    return re.sub(r"[ .]+", "-", candidate).rstrip("-")[:72] or fallback


def split_rows(value, height, *, min_span=1, default=None):
    if value is None:
        if default is None:
            raise BenchError("请选择行区间。")
        return default
    pair = sequence(value, "行区间", minimum=2, maximum=2)
    top = integer(pair[0], "起始行", 0, height - 1)
    bottom = integer(pair[1], "停止行", top + 1, height)
    if bottom - top < min_span:
        raise BenchError(f"行区间至少要包含 {min_span} 行。")
    return top, bottom
