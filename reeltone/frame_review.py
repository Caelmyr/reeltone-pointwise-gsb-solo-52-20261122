"""Service layer for frame-level evidence cross checks.

Resolves the retained reference (a crop proof's native companion pixels or a
native region of the current source scan), runs the frame check against an
uploaded image, and retains the result beside the existing evidence. The
archive source scan and the retained proof files are opened read-only and are
never rewritten.
"""
from hashlib import sha256

from .contracts import identifier, mapping, text
from .errors import BenchError, Conflict
from .framecheck import (
    VERDICT_LABELS, frame_check, read_returned,
)
from .geometry import Box
from .store import Catalog


def _decode_native_png(blob):
    from io import BytesIO
    from PIL import Image
    import numpy as np

    with Image.open(BytesIO(blob)) as image:
        if image.mode in {"I;16", "I;16L", "I;16B", "I;16N", "I"}:
            array = np.asarray(image)
            return np.asarray(array, dtype=np.float32) / 65535.0, 16
        return np.asarray(image.convert("L"), dtype=np.float32) / 255.0, 8


def reference_from_proof(catalog: Catalog, proof_id):
    proof = catalog.get_proof(proof_id, raw=True)
    evidence = proof["evidence"]
    # Crop proofs retain the un-turned, un-scaled native pixels as their
    # secondary companion; that is the only proof kind usable as a reference.
    if not evidence.get("comparison_preserves_native_pixels") or proof["secondary_file"] is None:
        raise BenchError("请选择「保留原像素证据」生成的裁片作为核对基准。")
    pixels, depth = _decode_native_png(proof["secondary_file"])
    raw_box = evidence["box"]
    # The companion PNG stores exactly the cropped native pixels; its box gives
    # the crop's position in un-turned source coordinates.
    box = Box(raw_box[0], raw_box[1], raw_box[0] + pixels.shape[1], raw_box[1] + pixels.shape[0])
    meta = {
        "kind": "proof_crop", "proof_id": proof["id"], "scan_id": proof["scan_id"],
        "title": proof["title"], "source_sha256": evidence["parent"]["source_sha256"],
        "bit_depth": depth,
    }
    return pixels, box, meta


def _region_request(values):
    """Normalize either a JSON ``box`` list or multipart ``box0..box3`` fields."""
    if values.get("box") is not None:
        return {"box": values["box"]}
    if all(values.get(f"box{i}") not in (None, "") for i in range(4)):
        try:
            return {"box": [int(values[f"box{i}"]) for i in range(4)]}
        except (TypeError, ValueError):
            raise BenchError("原片范围必须是整数列、行坐标。") from None
    return {}


def reference_from_region(material, request):
    box = material.region(_region_request(request),
                          fallback=Box(0, 0, material.width, min(560, material.height)))
    if box.area > 1_500_000:
        from .errors import Limit
        raise Limit("单张像素证据最多 150 万原生像素。")
    pixels = box.subimage(material.pixels)
    meta = {
        "kind": "source_region", "proof_id": None, "scan_id": material.record["id"],
        "title": material.record["title"], "source_sha256": material.record["sha256"],
        "bit_depth": material.raster.bit_depth,
    }
    return pixels, box, meta


def run_uploaded_check(catalog: Catalog, material, form, uploaded, filename):
    values = mapping(form, "核对请求") if isinstance(form, dict) else {}
    proof_id = values.get("proof_id")
    if proof_id:
        proof_id = identifier(proof_id, "裁片证据编号")
        if catalog.get_proof(proof_id)["scan_id"] != material.record["id"]:
            raise Conflict("裁片证据不属于当前扫描。")
        pixels, box, meta = reference_from_proof(catalog, proof_id)
    else:
        pixels, box, meta = reference_from_region(material, values)
    returned = read_returned(uploaded, filename)
    return frame_check(pixels, meta, box, returned)


def retain_check(catalog: Catalog, material, values, uploaded, filename):
    check = run_uploaded_check(catalog, material, values, uploaded, filename)
    title = text(values.get("title", ""), "核对名称", maximum=90)
    note = text(values.get("note", ""), "核对备注", maximum=650, required=False, multiline=True)
    proof_id = values.get("proof_id")
    if proof_id:
        proof_id = identifier(proof_id, "裁片证据编号")
    record = catalog.save_frame_check(material.record["id"], proof_id, title, note, check.reading(),
                                      check.primary_png, check.secondary_png,
                                      check.reading()["returned_image"]["sha256"])
    return record, check


def current_frame_check_properties(catalog: Catalog, check_id):
    check = catalog.get_frame_check(check_id, raw=True)
    source = catalog.get_scan(check["scan_id"])
    reading = check["reading"]
    origin = reading["reference"]
    checks = {
        "source": source["sha256"] == origin["source_sha256"],
        "primary_file": sha256(check.pop("primary_file")).hexdigest() == check["primary_sha256"],
        "secondary_file": sha256(check.pop("secondary_file")).hexdigest() == check["secondary_sha256"],
        "returned_recorded": reading["returned_image"]["sha256"] == check["returned_sha256"],
    }
    if origin.get("proof_id"):
        proof = catalog.get_proof(origin["proof_id"], raw=True)
        checks["reference_proof"] = proof["evidence"]["parent"]["source_sha256"] == origin["source_sha256"]
    reading["verdict_label"] = VERDICT_LABELS[reading["verdict"]]
    return {"check": check, "source": source, "integrity": checks, "verified": all(checks.values())}
