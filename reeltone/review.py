from hashlib import sha256

from .audition import read_wav, waveform_statistics
from .bench import load_material
from .contracts import identifier, mapping, text
from .cues import PacketRequest, parse_cue, validate_cue_anchor
from .errors import Conflict
from .packet import make_packet
from .store import Catalog


def current_take_properties(catalog, take_id):
    record = catalog.get_take(take_id, raw=True)
    source = catalog.get_scan(record["scan_id"])
    recipe = catalog.get_recipe(record["scan_id"], record["recipe_id"])
    related = record["payload"]["parent"]
    checks = {"original_scan": source["sha256"] == related["source_sha256"],
              "recipe": recipe["fingerprint"] == related["recipe_fingerprint"],
              "audio": sha256(record["wav"]).hexdigest() == record["wav_sha256"]}
    samples, rate = read_wav(record.pop("wav"))
    metadata = record["payload"]["audio"]
    checks["length"] = metadata["frames"] == len(samples)
    checks["rate"] = metadata["output_rate"] == rate
    return {"record": record, "source": source, "recipe": recipe, "verified": all(checks.values()),
            "integrity": checks, "retained_pcm_statistics": waveform_statistics(samples),
            "duration_seconds": round(len(samples) / rate, 8)}


def current_proof_properties(catalog, proof_id):
    proof = catalog.get_proof(proof_id, raw=True)
    source = catalog.get_scan(proof["scan_id"])
    recipe = catalog.get_recipe(proof["scan_id"], proof["recipe_id"])
    reading = proof["evidence"]
    origin = reading["parent"]
    checks = {"source": source["sha256"] == origin["source_sha256"],
              "recipe": recipe["fingerprint"] == origin["recipe_fingerprint"],
              "primary_file": sha256(proof.pop("primary_file")).hexdigest() == proof["primary_sha256"]}
    secondary = proof.pop("secondary_file")
    if secondary is not None:
        checks["secondary_file"] = sha256(secondary).hexdigest() == proof["secondary_sha256"]
    if reading.get("reference") is not None:
        other = catalog.get_scan(reading["reference"]["scan_id"])
        checks["related_source"] = other["sha256"] == reading["reference"]["source_sha256"]
    return {"proof": proof, "source": source, "recipe": recipe, "integrity": checks, "verified": all(checks.values())}


def save_cue(catalog, scan_id, values, cue_id=None):
    current = catalog.get_scan(scan_id)
    fields = parse_cue(values, current["metadata"]["height"])
    if fields["anchor"]:
        take = catalog.get_take(fields["anchor"]["take"])
        fields["anchor"] = validate_cue_anchor(fields, take, scan_id)
    return catalog.write_cue(scan_id, fields, cue_id)


def restore_view(catalog, scan_id, body):
    material = load_material(catalog, scan_id, body.get("recipe") or None)
    regions = catalog.list_markers(scan_id)
    marker_id = identifier(body.get("marker"), "定位框编号")
    choice = next((record for record in regions if record["id"] == marker_id), None)
    if choice is None:
        raise Conflict("该扫描不包含这个定位框。")
    if choice["display"]["source_sha256"] != material.record["sha256"]:
        raise Conflict("定位框记录的图像摘要与当前扫描不一致。")
    box = material.region({"box": choice["display"]["box"]})
    return {"marker": choice, "source_box": box.as_list(),
            "display_box": material.recipe.orientation.box(box, material.width, material.height).as_list(),
            "orientation": material.recipe.orientation.as_dict(), "current_recipe": material.recipe_record["id"]}


def build_packet(catalog: Catalog, reel_id, values):
    request = PacketRequest.parse(values)
    reel = catalog.get_reel(reel_id)
    scan_index = {scan["id"]: scan for scan in catalog.list_scans(reel_id)}
    if not scan_index:
        raise Conflict("档案卷还没有扫描源。")
    all_cues = catalog.list_cues(reel_id)
    takes = {}
    proofs = []
    recipes = {}
    source_bytes = 0
    to_load = set()
    for item in request.items:
        if item.take_id in takes:
            continue
        take = catalog.get_take(item.take_id, raw=True)
        if take["scan_id"] not in scan_index:
            raise Conflict("监听必须来自当前档案卷。")
        if sha256(take["wav"]).hexdigest() != take["wav_sha256"]:
            raise Conflict("监听内容与录入摘要不符，无法打包。")
        takes[item.take_id] = take
        to_load.add(take["scan_id"])
        recipes[take["recipe_id"]] = catalog.get_recipe(take["scan_id"], take["recipe_id"])
    for proof_id in request.proofs:
        proof = catalog.get_proof(proof_id, raw=True)
        if proof["scan_id"] not in scan_index:
            raise Conflict("证据图必须来自当前档案卷。")
        if sha256(proof["primary_file"]).hexdigest() != proof["primary_sha256"]:
            raise Conflict("证据图内容与录入摘要不符，无法打包。")
        if proof["secondary_file"] and sha256(proof["secondary_file"]).hexdigest() != proof["secondary_sha256"]:
            raise Conflict("证据图的配套文件与录入摘要不符，无法打包。")
        proofs.append(proof)
        recipes[proof["recipe_id"]] = catalog.get_recipe(proof["scan_id"], proof["recipe_id"])
        to_load.add(proof["scan_id"])
        for sid in proof["evidence"].get("related_scans", []):
            if sid not in scan_index:
                raise Conflict("关联证据的对照图不属于当前档案卷。")
            to_load.add(sid)
            reference = mapping(proof["evidence"].get("reference"))
            recipe = catalog.get_recipe(sid, reference["recipe_id"])
            recipes[recipe["id"]] = recipe
    if request.include_cues:
        for cue in all_cues:
            to_load.add(cue["scan_id"])
    if request.sources:
        for scanning in sorted(to_load):
            provisional = scan_index[scanning]
            source_bytes += provisional["metadata"]["file_size"]
            if source_bytes > 62 * 1024 * 1024:
                raise Conflict("选中的原始扫描大于交付上限。可关闭附带原始文件。")
            content = catalog.get_scan(scanning, raw=True)
            if sha256(content["raw"]).hexdigest() != content["sha256"]:
                raise Conflict("原始扫描与录入摘要不符，无法打包。")
            scan_index[scanning] = content
    archive = make_packet(reel, request, scan_index, takes, proofs, all_cues, recipes)
    named_cues = {item.cue_id for item in request.items if item.cue_id is not None}
    cue_revisions = {cue["id"]: cue["updated"] for cue in all_cues if request.include_cues or cue["id"] in named_cues}
    take_revisions = {take_id: take["updated"] for take_id, take in takes.items()}
    return catalog.save_packet(reel_id, request.label, archive.filename, archive.summary, archive.archive,
                               take_revisions=take_revisions, cue_revisions=cue_revisions)


def save_marker(catalog, material, raw):
    title = text(raw.get("title", ""), "定位框名称", maximum=85)
    kind = raw.get("category", "track")
    notes = text(raw.get("note", ""), "定位备注", maximum=600, required=False, multiline=True)
    region = material.region({"box": raw.get("box")})
    if raw.get("box") is None:
        raise Conflict("保存定位框前须选择原片范围。")
    nav = {"recipe": material.recipe_record["id"], "box": region.as_list(), "source_sha256": material.record["sha256"],
           "view_orientation": material.recipe.orientation.as_dict()}
    return catalog.add_marker(material.record["id"], title, kind, nav, notes)
