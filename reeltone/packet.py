from collections import Counter
from contextlib import closing
import csv
from dataclasses import dataclass
from hashlib import sha256
import html
from io import BytesIO, StringIO
import json
from pathlib import PurePosixPath
import re
from zipfile import BadZipFile, ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo

from .contracts import shipment_stem
from .cues import PacketRequest, cue_coverage, expand_items
from .errors import BenchError, Limit
from .store import utc_text


SCHEMA = "reeltone-review-v1"
MAX_PACKET_BYTES = 62 * 1024 * 1024
MAX_EXPANDED_BYTES = 115 * 1024 * 1024
MAX_MEMBERS = 175
MANIFEST_NAME = "manifest.json"
HEX64 = re.compile(r"[a-f0-9]{64}\Z")


@dataclass(frozen=True)
class Resource:
    path: str
    body: bytes
    media: str
    role: str

    def describe(self):
        return {"path": self.path, "bytes": len(self.body), "sha256": sha256(self.body).hexdigest(),
                "media": self.media, "role": self.role}


@dataclass(frozen=True)
class Packet:
    archive: bytes
    filename: str
    summary: dict


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def csv_bytes(rows):
    output = StringIO(newline="")
    writer = csv.writer(output, lineterminator="\r\n")
    writer.writerows(rows)
    return output.getvalue().encode("utf-8-sig")


def cue_document(scan_index, cues):
    header = ("scan", "scan_sha256", "kind", "status", "first_native_row", "after_last_native_row", "title", "note", "take_id")
    output = [header]
    for cue in cues:
        scan = scan_index[cue["scan_id"]]
        anchor = cue["anchor"] or {}
        output.append((scan["title"], scan["sha256"], cue["kind"], cue["status"], cue["y0"], cue["y1"],
                       cue["title"], cue["note"], anchor.get("take", "")))
    return csv_bytes(output)


def clip_document(positions, scans):
    header = ("item", "title", "audio", "scanner_source", "source_sha256", "first_row", "after_last_row", "take_id",
              "take_sample_start", "take_sample_stop", "audition_rate_hz", "playback_start", "playback_stop",
              "estimated_time_spans")
    rows = [header]
    for clip in positions:
        source = scans[clip["scan"]]
        movie = clip["continuous"]
        time = movie["timecodes"] if movie else ("", "")
        estimated = " ".join(f"{span['native_rows'][0]}-{span['native_rows'][1]}"
                             for span in clip.get("estimated_spans") or ())
        rows.append((clip["index"], clip["title"], clip["file"], source["title"], source["sha256"],
                     *clip["source_rows"], clip["take"], *clip["take_samples"], clip["clip_sample_rate"], *time,
                     estimated))
    return csv_bytes(rows)


def print_document(reel, title, clips, proofs, cues, already):
    esc = lambda value: html.escape(str(value), quote=True)
    paper = [
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">',
        f"<title>{esc(title)} · ReelTone</title>",
        "<style>body{background:#fff;color:#17252c;font:15px system-ui;margin:4rem auto;max-width:1040px;padding:0 1rem}",
        "h1,h2{font-weight:600} h2{margin-top:2rem}p,td{line-height:1.5}table{border-collapse:collapse;width:100%}",
        "td,th{text-align:left;padding:.65rem;border-bottom:1px solid #c2c9cb}th{background:#edf4f4}",
        "small{color:#41545a}code{overflow-wrap:anywhere}img{max-width:100%;max-height:290px;display:block;margin-top:.5rem}",
        "article{border-bottom:1px solid #cdd8d8;padding:1rem 0} @media print{body{margin:0} article,tr{break-inside:avoid}}</style>",
        "<body><header><small>REELTONE / 审听交接</small>",
        f"<h1>{esc(title)}</h1><p>{esc(reel['title'])} · {esc(reel['gauge'])} · {esc(reel['accession'] or '未填入藏号')}</p>",
        f"<p><small>{esc(already)}</small></p></header>",
    ]
    if clips:
        paper.append("<h2>监听顺序</h2><table><thead><tr><th>片段</th><th>原片行区间</th><th>串听起止</th><th>时间依据</th></tr></thead><tbody>")
        for clip in clips:
            start, stop = clip["source_rows"]
            label = f"{clip['index']}. {clip['title']}"
            playback = " – ".join(clip["continuous"]["timecodes"]) if clip["continuous"] else "独立 WAV"
            estimated = clip.get("estimated_spans") or []
            basis = ("含估算：" + "、".join(f"{span['native_rows'][0]}–{span['native_rows'][1]}" for span in estimated)
                     + " 行") if estimated else "按保存时的定时映射"
            paper.append(f"<tr><td><a href='{esc(clip['file'])}'>{esc(label)}</a></td><td>{start} ≤ 行 &lt; {stop}</td>"
                         f"<td>{esc(playback)}</td><td>{esc(basis)}</td></tr>")
        paper.append("</tbody></table>")
    if proofs:
        paper.append("<h2>图像证据</h2>")
        for frozen in proofs:
            paper.append(f"<article><strong>{esc(frozen['title'])}</strong><p>{esc(frozen['note'])}</p>")
            for link in frozen["files"]:
                paper.append(f"<a href='{esc(link)}'><img alt='{esc(frozen['title'])}' src='{esc(link)}'></a>")
            paper.append("</article>")
    if cues:
        paper.append("<h2>原片审听标记</h2><table><thead><tr><th>标签</th><th>行</th><th>状态</th><th>备注</th></tr></thead><tbody>")
        for cue in cues:
            paper.append(f"<tr><td>{esc(cue['title'])}</td><td>{cue['y0']} ≤ 行 &lt; {cue['y1']}</td>"
                         f"<td>{esc(cue['status'])}</td><td>{esc(cue['note'])}</td></tr>")
        paper.append("</tbody></table>")
    paper.append("<p><small>源校验与采样位置见 manifest.json。</small></p></body></html>")
    return "\n".join(paper).encode("utf-8")


def proof_assets(proofs):
    records = []
    assets = []
    for i, proof in enumerate(proofs, 1):
        stem = f"evidence/{i:02d}-{shipment_stem(proof['title'])}"
        first = f"{stem}.png"
        assets.append(Resource(first, proof["primary_file"], "image/png", "evidence"))
        files = [first]
        if proof["secondary_file"]:
            extra = f"{stem}-native.png"
            assets.append(Resource(extra, proof["secondary_file"], "image/png", "evidence_detail"))
            files.append(extra)
        records.append({"id": proof["id"], "title": proof["title"], "category": proof["category"], "note": proof["note"],
                        "scan": proof["scan_id"], "recipe_id": proof["recipe_id"], "files": files, "reading": proof["evidence"]})
    return records, assets


def asset_origins(scans, recipe_map, include_sources):
    assets = []
    manifest = []
    for i, scan in enumerate(scans.values(), 1):
        meta = scan["metadata"]
        stored_native = None
        if include_sources:
            stored_native = f"sources/{i:02d}-{meta['filename']}"
            assets.append(Resource(stored_native, scan["raw"], meta["mime"], "original_scan"))
        linked = [recipe for recipe in recipe_map.values() if recipe["scan_id"] == scan["id"]]
        manifest.append({"id": scan["id"], "title": scan["title"], "native": stored_native, "sha256": scan["sha256"],
                         "metadata": meta, "acquisition_note": scan["source_note"], "recipe_ids": [recipe["id"] for recipe in linked]})
    return manifest, assets


def put_files(resources):
    output = BytesIO()
    if len(resources) > MAX_MEMBERS:
        raise Limit("交付内容文件数超出上限。")
    expanded = sum(len(resource.body) for resource in resources)
    if expanded > MAX_EXPANDED_BYTES:
        raise Limit("交付内容展开后体积超出上限。减少原始扫描和证据后再试。")
    with ZipFile(output, mode="w", compression=ZIP_DEFLATED, compresslevel=4) as archive:
        for resource in resources:
            entry = ZipInfo(resource.path, date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = ZIP_STORED if resource.media in {"image/png", "image/jpeg"} else ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, resource.body)
    payload = output.getvalue()
    if len(payload) > MAX_PACKET_BYTES:
        raise Limit("交付包最多 62 MiB。减少打包文件量后重试。")
    return payload


def make_packet(reel, request: PacketRequest, scan_index, takes, proofs, cues, recipe_map):
    exists_by_id = {cue["id"]: cue for cue in cues}
    clips, audio = expand_items(request.items, takes, exists_by_id, request.rate, request.continuous)
    if not clips and not proofs and not cues:
        raise BenchError("没有可纳入交付包的内容。")
    frozen_proofs, proof_files = proof_assets(proofs)
    picked_cues = cues if request.include_cues else []
    selected_scan_ids = dict.fromkeys(row["scan"] for row in [*clips, *frozen_proofs])
    for cue in picked_cues:
        selected_scan_ids[cue["scan_id"]] = None
    for proof in frozen_proofs:
        for related in proof["reading"].get("related_scans", []):
            selected_scan_ids[related] = None
    sources, source_files = asset_origins({scan_id: scan_index[scan_id] for scan_id in selected_scan_ids},
                                          recipe_map, request.sources)
    if request.sources and sum(len(file.body) for file in source_files) > MAX_PACKET_BYTES:
        raise Limit("原始扫描合计超过交付上限。可关闭‘纳入原始扫描’后再制作。")
    created = utc_text()
    resources = [*[Resource(*entry) for entry in audio], *proof_files, *source_files]
    if clips:
        resources.append(Resource("clip-index.csv", clip_document(clips, scan_index), "text/csv", "clip_table"))
    if request.include_cues:
        resources.append(Resource("review-cues.csv", cue_document(scan_index, picked_cues), "text/csv", "cue_table"))
    resources.append(Resource("review.html", print_document(reel, request.label, clips, frozen_proofs, picked_cues, created),
                              "text/html", "print_review"))
    manifest = {"schema": SCHEMA, "title": request.label, "reel": reel, "created": created, "scans": sources,
                "takes": [{k: v for k, v in take.items() if k != "wav"} for take in takes.values()],
                "recipes": list(recipe_map.values()), "clips": clips, "proofs": frozen_proofs,
                "cue_summary": cue_coverage(picked_cues), "members": [item.describe() for item in resources]}
    resources.insert(0, Resource(MANIFEST_NAME, json_bytes(manifest), "application/json", "manifest"))
    result = put_files(resources)
    roles = Counter(resource.role for resource in resources)
    summary = {"asset_count": len(resources), "clips": clips, "clip_count": len(clips), "images": len(frozen_proofs),
               "source_count": len(sources), "cue_count": len(picked_cues), "roles": dict(roles), "created": created,
               "bytes": len(result), "manifest_sha256": sha256(resources[0].body).hexdigest()}
    return Packet(result, f"{shipment_stem(request.label)}.zip", summary)


def path_allowed(name):
    if not isinstance(name, str) or len(name) > 160 or "\\" in name or name.startswith("/") or name.endswith("/"):
        return False
    parts = name.split("/")
    return not any(part in {"", ".", ".."} or ":" in part for part in parts) and PurePosixPath(name).as_posix() == name


def archive_index(archive):
    infos = archive.infolist()
    if len(infos) > MAX_MEMBERS or sum(info.file_size for info in infos) > MAX_EXPANDED_BYTES:
        raise Limit("提交的交付包超过验真容量。")
    index = {}
    for entry in infos:
        if not path_allowed(entry.filename) or entry.filename in index:
            raise BenchError("交付包含有不合规或重名条目。")
        if entry.compress_type not in {ZIP_STORED, ZIP_DEFLATED} or entry.flag_bits & 0x01:
            raise BenchError("交付包含有无法验真的文件类型或加密文件。")
        index[entry.filename] = entry
    if MANIFEST_NAME not in index or index[MANIFEST_NAME].file_size > 7 * 1024 * 1024:
        raise BenchError("交付包缺少可读取的 manifest.json。")
    return index


def member_claims(manifest):
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA or not isinstance(manifest.get("members"), list):
        raise BenchError("交付清单版本或内容无效。")
    records = {}
    for row in manifest["members"]:
        if not isinstance(row, dict):
            raise BenchError("交付清单的文件记录无效。")
        path, digest, length = row.get("path"), row.get("sha256"), row.get("bytes")
        if not path_allowed(path) or path in records or path == MANIFEST_NAME:
            raise BenchError("文件清单含有无效或重复路径。")
        if not isinstance(length, int) or length < 0 or not isinstance(digest, str) or not HEX64.fullmatch(digest):
            raise BenchError("文件清单缺少有效的大小或摘要。")
        records[path] = row
    return records


def verify_member(path, record, archive, actual):
    if actual is None:
        return {"path": path, "status": "missing"}
    if actual.file_size != record["bytes"]:
        return {"path": path, "status": "length_mismatch", "declared_bytes": record["bytes"], "found_bytes": actual.file_size}
    total = 0
    digest_maker = sha256()
    try:
        with closing(archive.open(actual)) as stream:
            while chunk := stream.read(128 * 1024):
                total += len(chunk)
                if total > record["bytes"]:
                    return {"path": path, "status": "length_mismatch", "found_bytes": total}
                digest_maker.update(chunk)
    except (BadZipFile, OSError, RuntimeError):
        return {"path": path, "status": "unreadable"}
    checksum = digest_maker.hexdigest()
    status = "verified" if checksum == record["sha256"] else "digest_mismatch"
    return {"path": path, "status": status, "bytes": total, "sha256": checksum}


def source_relationships(manifest, claims):
    findings = []
    sources = manifest.get("scans", [])
    if not isinstance(sources, list):
        return [{"kind": "invalid_sources"}]
    known_ids = set()
    for source in sources:
        if not isinstance(source, dict):
            findings.append({"kind": "invalid_source"})
            continue
        source_id = source.get("id")
        if not isinstance(source_id, str) or source_id in known_ids:
            findings.append({"kind": "invalid_source_id"})
            continue
        known_ids.add(source_id)
        stored = source.get("native")
        if stored is not None and (not isinstance(stored, str) or stored not in claims
                                   or claims[stored]["sha256"] != source.get("sha256") or claims[stored].get("role") != "original_scan"):
            findings.append({"kind": "source_digest_reference", "id": source_id})
    clip_list = manifest.get("clips", [])
    if not isinstance(clip_list, list):
        return [*findings, {"kind": "invalid_clips"}]
    for clip in clip_list:
        if not isinstance(clip, dict):
            findings.append({"kind": "invalid_clip"})
            continue
        src = clip.get("scan")
        audio = clip.get("file")
        if not isinstance(src, str) or src not in known_ids or not isinstance(audio, str) or audio not in claims:
            findings.append({"kind": "clip_reference", "index": str(clip.get("index"))[:25]})
    return findings


def verify_packet(data):
    if not data or len(data) > MAX_PACKET_BYTES:
        raise Limit("请选择大小不超过 62 MiB 的交付包。")
    try:
        with ZipFile(BytesIO(data), mode="r") as archive:
            actual = archive_index(archive)
            raw_manifest = archive.read(actual[MANIFEST_NAME])
            manifest = json.loads(raw_manifest)
            claims = member_claims(manifest)
            comparison = [verify_member(path, claim, archive, actual.get(path)) for path, claim in claims.items()]
            unexpected = sorted(set(actual) - set(claims) - {MANIFEST_NAME})
            comparison.extend({"path": name, "status": "undeclared"} for name in unexpected)
            findings = source_relationships(manifest, claims)
    except (BadZipFile, UnicodeDecodeError, json.JSONDecodeError, NotImplementedError, RuntimeError, RecursionError):
        raise BenchError("无法读取 ZIP 或清单。") from None
    counts = dict(Counter(item["status"] for item in comparison))
    clean = not findings and not any(result["status"] != "verified" for result in comparison)
    return {"verified": clean, "title": str(manifest.get("title", ""))[:120], "manifest_sha256": sha256(raw_manifest).hexdigest(),
            "check_counts": counts, "files": comparison, "relationships": findings, "original_count":
            sum(item.get("role") == "original_scan" for item in claims.values())}
