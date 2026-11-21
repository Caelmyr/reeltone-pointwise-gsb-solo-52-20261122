"""Reading-room gallery packets: redacted strip, monitor and row ranges only."""
from hashlib import sha256
from io import BytesIO
import json
from zipfile import BadZipFile, ZipFile

from .audition import wav_bytes
from .bench import load_material
from .contracts import mapping, shipment_stem, text
from .errors import BenchError, Limit
from .packet import MAX_PACKET_BYTES, archive_index, csv_bytes, json_bytes, put_files, Resource
from .redact import parse_feather, parse_spans, render_gallery
from .store import utc_text

SCHEMA = "reeltone-gallery-v1"


def _clock_rows(clock):
    return {"rows_per_second": round(clock.rows_per_second, 6), "reverse": clock.reverse,
            "frames_per_second": clock.frame_rate, "source": clock.source,
            "total_duration_seconds": round(clock.seconds, 7)}


def range_document(scan, spans, feather, clock):
    rows = [("first_native_row", "after_last_native_row", "label", "note", "feather_rows_each_side",
             "held_playback_start", "held_playback_stop")]
    for span in spans:
        start, stop = clock.extent((span.y0, span.y1))
        rows.append((span.y0, span.y1, span.label, span.note, feather, round(start, 7), round(stop, 7)))
    return csv_bytes(rows)


def build_gallery(catalog, scan_id, values):
    data = mapping(values, "阅览包请求")
    material = load_material(catalog, scan_id, data.get("recipe") or None)
    spans = parse_spans(data, material.height)
    feather = parse_feather(data)
    label = text(data.get("title", "阅览遮盖包"), "阅览包名称", maximum=100)
    wave, rate, image_png, safety, audition_result, envelope = render_gallery(material, spans, feather,
                                                                              data.get("monitor", {}))
    clock = audition_result.clock
    stem = shipment_stem(material.record["title"], "strip")
    image_name = f"redacted/{stem}-redacted.png"
    audio_name = f"redacted/{stem}-monitor.wav"
    ranges_name = "redacted-row-ranges.csv"
    wav_body = wav_bytes(wave, rate)
    created = utc_text()
    members = [Resource(image_name, image_png, "image/png", "redacted_strip"),
               Resource(audio_name, wav_body, "audio/wav", "redacted_monitor"),
               Resource(ranges_name, range_document(material.record, spans, feather, clock), "text/csv", "row_ranges")]
    manifest = {
        "schema": SCHEMA, "title": label, "created": created,
        "source": {"scan_id": material.record["id"], "title": material.record["title"],
                   "source_sha256": material.record["sha256"],
                   "width": material.width, "height": material.height,
                   "recipe_id": material.recipe_record["id"], "recipe_number": material.recipe_record["seq"]},
        "clock": _clock_rows(clock), "monitor": audition_result.options.as_dict(),
        "redaction": {"feather_rows": feather, "hard_held_rows": safety["held_rows"],
                      "controlled_rows": safety["controlled_rows"],
                      "spans": [{"rows": [span.y0, span.y1], "label": span.label, "note": span.note} for span in spans]},
        "audio_frames": safety["output_frames"], "safety": safety,
        "members": [item.describe() for item in members],
    }
    members.insert(0, Resource("manifest.json", json_bytes(manifest), "application/json", "manifest"))
    archive = put_files(members)
    if len(archive) > MAX_PACKET_BYTES:
        raise Limit("阅览包超过 62 MiB。")
    summary = {"schema": SCHEMA, "span_count": len(spans), "held_rows": safety["held_rows"],
               "asset_count": len(members), "bytes": len(archive),
               "manifest_sha256": sha256(members[0].body).hexdigest(), "created": created}
    return {"archive": archive, "filename": f"gallery-{shipment_stem(label)}.zip", "summary": summary, "manifest": manifest}


def save_gallery(catalog, reel_id, scan_id, values):
    built = build_gallery(catalog, scan_id, values)
    packet = catalog.save_packet(reel_id, built["manifest"]["title"], built["filename"], built["summary"],
                                 built["archive"], take_revisions={}, cue_revisions={})
    return packet, built["manifest"]


def verify_gallery(data):
    if not data or len(data) > MAX_PACKET_BYTES:
        raise Limit("请选择大小不超过 62 MiB 的阅览包。")
    try:
        with ZipFile(BytesIO(data), mode="r") as archive:
            index = archive_index(archive)
            raw_manifest = archive.read(index["manifest.json"])
            manifest = json.loads(raw_manifest)
            if manifest.get("schema") != SCHEMA:
                raise BenchError("这不是 ReelTone 阅览遮盖包（schema 不匹配）。")
            claims = {row["path"]: row for row in manifest["members"]}
            checks = []
            for path, claim in claims.items():
                entry = index.get(path)
                if entry is None:
                    checks.append({"path": path, "status": "missing"})
                    continue
                digest = sha256(archive.read(entry)).hexdigest()
                checks.append({"path": path, "bytes": entry.file_size,
                               "status": "verified" if digest == claim["sha256"] and entry.file_size == claim["bytes"]
                               else "digest_mismatch", "sha256": digest})
            undeclared = sorted(set(index) - set(claims) - {"manifest.json"})
            checks.extend({"path": name, "status": "undeclared"} for name in undeclared)
    except (BadZipFile, json.JSONDecodeError, KeyError, NotImplementedError, RuntimeError):
        raise BenchError("无法读取阅览包或清单。") from None
    roles = {item["role"] for item in manifest["members"]}
    required = {"redacted_strip", "redacted_monitor", "row_ranges"}
    clean = not undeclared and required <= roles and all(item["status"] == "verified" for item in checks)
    return {"schema": SCHEMA, "verified": clean, "title": manifest.get("title", ""),
            "manifest_sha256": sha256(raw_manifest).hexdigest(), "files": checks,
            "held_rows": manifest.get("redaction", {}).get("hard_held_rows"),
            "span_count": len(manifest.get("redaction", {}).get("spans", []))}
