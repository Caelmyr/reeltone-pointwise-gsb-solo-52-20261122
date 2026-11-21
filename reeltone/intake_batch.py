from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import PurePosixPath
import unicodedata
from zipfile import BadZipFile, ZIP_DEFLATED, ZIP_STORED, ZipFile

import numpy as np

from .contracts import mapping, option, safe_filename, text
from .errors import BenchError, Conflict, Limit
from .raster import FILE_BYTES, MAX_PIXELS, read_scan
from .recipe import Recipe
from .visual import source_overview


BATCH_BYTES = 58 * 1024 * 1024
BATCH_PIXELS = 27_000_000
BATCH_MEMBERS = 14
SUPPORTED_SUFFIXES = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}
SIDE_MANIFEST = "intake.json"


@dataclass(frozen=True)
class ImportEntry:
    path: str
    raster: object
    title: str
    note: str
    recipe: Recipe
    archive_digest: str

    def draft(self):
        return {"raster": self.raster, "title": self.title, "note": self.note, "recipe": self.recipe}

    def compact(self, duplicates, *, preview=False):
        meta = self.raster.metadata()
        matched = duplicates.get(meta["sha256"])
        payload = {"path": self.path, "title": self.title, "derived_guide": self.recipe.aperture.as_list(),
                   "metadata": meta, "receipt": self.archive_digest, "prior_scan": matched, "byte_duplicate": matched is not None}
        if preview:
            payload["preview"] = source_overview(self.raster.intensity, self.recipe.aperture, self.recipe.perforations,
                                                 self.recipe.orientation, size=(176, 330))
        return payload


def member_path(name):
    if not isinstance(name, str) or not name or len(name) > 160 or name.startswith("/") or "\\" in name:
        raise BenchError("导入 ZIP 内部路径无效。")
    components = name.split("/")
    if any(part in {"", ".", ".."} or ":" in part or any(ord(char) < 32 for char in part) for part in components):
        raise BenchError("导入 ZIP 内部路径无效。")
    return str(PurePosixPath(name))


def accepted_entries(archive):
    files = []
    listed = set()
    total = 0
    sheet = None
    for entry in archive.infolist():
        if entry.is_dir():
            member_path(entry.filename.rstrip("/"))
            continue
        name = member_path(entry.filename)
        normalized = unicodedata.normalize("NFKC", name).casefold()
        if normalized in listed:
            raise BenchError("导入 ZIP 含有名称重复或仅大小写不同的文件。")
        listed.add(normalized)
        if entry.flag_bits & 0x01 or entry.compress_type not in {ZIP_STORED, ZIP_DEFLATED}:
            raise BenchError("导入 ZIP 须使用不加密的常见压缩。")
        if name == SIDE_MANIFEST:
            if entry.file_size > 48_000:
                raise Limit("intake.json 超过 48 KB。")
            sheet = entry
            continue
        if PurePosixPath(name).suffix.lower() not in SUPPORTED_SUFFIXES:
            raise BenchError(f"ZIP 中含有无法作为声轨扫描导入的文件：{name[:75]}。")
        if entry.file_size <= 0 or entry.file_size > FILE_BYTES:
            raise Limit("单张扫描需大于零且最多为 32 MiB。")
        total += entry.file_size
        if total > BATCH_BYTES:
            raise Limit("同批展开文件大小合计最多 58 MiB。")
        files.append(entry)
    if not files or len(files) > BATCH_MEMBERS:
        raise Limit(f"每个导入批次须包含 1–{BATCH_MEMBERS} 张扫描。")
    return files, sheet


def sheet_entries(blob, allowed):
    if blob is None:
        return {}
    try:
        data = json.loads(blob.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        raise BenchError("intake.json 不是可读的 UTF-8 JSON。") from None
    document = mapping(data, "导入资料单")
    records = mapping(document.get("scans", {}), "导入资料单的 scans 项")
    entries = {}
    for path, row in records.items():
        name = member_path(path)
        if name not in allowed:
            raise BenchError(f"资料单引用了 ZIP 中不存在的扫描：{name[:75]}。")
        metadata = mapping(row, "扫描资料")
        entries[name] = {"title": text(metadata.get("title", ""), "扫描备注标题", maximum=100, required=False),
                         "note": text(metadata.get("note", ""), "扫描说明", maximum=660, required=False, multiline=True),
                         "direction": option(metadata.get("direction", "forward"), ("forward", "reverse"), "走片方向")}
    return entries


def read_intake(data, reel):
    if not data or len(data) > BATCH_BYTES:
        raise Limit("请选择不超过 58 MiB 的扫描 ZIP。")
    digest = sha256(data).hexdigest()
    try:
        with ZipFile(BytesIO(data)) as archive:
            files, sheet = accepted_entries(archive)
            lookup = sheet_entries(archive.read(sheet) if sheet else None, {file.filename for file in files})
            result = []
            pixel_count = 0
            for file in files:
                path = file.filename
                record = lookup.get(path, {})
                filename = safe_filename(path)
                blob = archive.read(file)
                raster = read_scan(blob, filename)
                pixel_count += raster.width * raster.height
                if pixel_count > BATCH_PIXELS:
                    raise Limit("同一导入批次包含超过 2,700 万扫描像素。")
                draft = Recipe.default(raster.width).as_dict()
                draft["clock"]["holes_per_frame"] = 4 if reel["gauge"] == "35 mm" else 1
                draft["orientation"]["reverse_y"] = record.get("direction") == "reverse"
                title = record.get("title") or filename.rsplit(".", 1)[0]
                result.append(ImportEntry(path, raster, title, record.get("note", ""), Recipe.parse(draft, raster.width), digest))
    except (BadZipFile, KeyError, RuntimeError, NotImplementedError, OSError):
        raise BenchError("无法读取导入 ZIP 或其成员。") from None
    return result, digest


def payload_for(entries, prior, allow_previews):
    known = {scan["sha256"]: scan["id"] for scan in prior}
    by_digest = defaultdict(list)
    summaries = []
    total = 0
    for n, entry in enumerate(entries):
        key = entry.raster.digest
        identical = by_digest[key]
        payload = entry.compact(known, preview=allow_previews and n < 5)
        payload["duplicates_inside_batch"] = identical.copy()
        payload["entry_index"] = n + 1
        payload["row_interval"] = [0, entry.raster.height]
        summaries.append(payload)
        identical.append(entry.path)
        total += int(key not in known and len(identical) == 1)
    return {"files": summaries, "ready_to_add": total, "entries": len(entries), "original_bytes": sum(len(item.raster.raw) for item in entries)}


def sample_columns(record, bins=18):
    values = record.raster.intensity
    row_stride = max(1, values.shape[0] // 1600)
    width = values.shape[1]
    partitions = np.linspace(0, width, min(bins, width) + 1, dtype=int)
    return [{"columns": [int(a), int(b)], "dark_fraction": round(float(np.mean(values[::row_stride, a:b] < 0.015)), 5),
             "bright_fraction": round(float(np.mean(values[::row_stride, a:b] > 0.985)), 5),
             "q20_q80": [round(float(x), 5) for x in np.percentile(values[::row_stride, a:b], [20, 80])]}
            for a, b in zip(partitions[:-1], partitions[1:])]


def inspect_upload(catalog, reel_id, data, commit_digest=None, making=False):
    reel = catalog.get_reel(reel_id)
    entries, digest = read_intake(data, reel)
    prior = catalog.list_scans(reel_id)
    summary = payload_for(entries, prior, allow_previews=not making)
    summary.update({"sha256": digest, "reel_id": reel_id, "draftable": True,
                    "observed_pixels": sum(entry.raster.width * entry.raster.height for entry in entries),
                    "declared_limit_pixels": min(MAX_PIXELS * len(entries), BATCH_PIXELS)})
    if making:
        if commit_digest != digest:
            raise Conflict("待导入的 ZIP 与最近预检的文件内容不一致。")
        summary["imported"] = catalog.add_scan_batch(reel_id, (entry.draft() for entry in entries), digest)
        summary["added_count"] = sum(item["created"] for item in summary["imported"])
    else:
        for index, row in enumerate(summary["files"]):
            if index < 5:
                row["exposure_columns"] = sample_columns(entries[index])
    return summary
