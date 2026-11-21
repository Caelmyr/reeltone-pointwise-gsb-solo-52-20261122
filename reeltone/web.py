from base64 import b64encode
from contextlib import contextmanager
from io import BytesIO
import os
from pathlib import Path
from threading import BoundedSemaphore

from flask import Flask, Response, jsonify, request, send_file, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge

from . import __version__
from .audition import MAX_MONITOR_SECONDS, RATES
from .bench import load_material, make_reading, render_audition, restore_settings, save_settings
from .contracts import flag, integer, mapping, option, safe_filename, shipment_stem, split_rows, text
from .contact import make_contact
from .cues import cue_coverage
from .demo import seed_library
from .errors import Absent, BenchError, Limit
from .instruments import TOOLS, run_tool
from .intake_batch import BATCH_BYTES, inspect_upload
from .listen_pair import retained_pair
from .packet import MAX_PACKET_BYTES, json_bytes, verify_packet
from .printscale import plan_physical, render_physical
from .raster import FILE_BYTES, MAX_HEIGHT, MAX_PIXELS, MAX_WIDTH, query_box, read_scan
from .recipe import Recipe
from .replay import retained_replay
from .review import build_packet, current_proof_properties, current_take_properties, restore_view, save_cue, save_marker
from .source_info import inspect_source
from .store import Catalog
from .transfer_review import export_review, from_catalog
from .visual import source_overview, source_tile


STATIC = Path(__file__).resolve().parents[1] / "static"


def json_body():
    return mapping(request.get_json(silent=True, cache=False), "提交的数据")


def want_recipe():
    return request.args.get("recipe") or None


def read_upload(field, ceiling):
    upload = request.files.get(field)
    if upload is None:
        raise BenchError("表单缺少文件。")
    content = upload.stream.read(ceiling + 1)
    if len(content) > ceiling:
        raise Limit(f"文件超过 {ceiling // (1024 * 1024)} MiB。")
    return content, upload.filename


def encode_data(content, media):
    if content is None:
        return None
    return f"data:{media};base64," + b64encode(content).decode("ascii")


def send_content(body, mime, name, *, attach=False, etag=None):
    response = send_file(BytesIO(body), mimetype=mime, as_attachment=attach, download_name=name,
                         conditional=True, etag=etag or False, max_age=0)
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


class ComputationGate:
    def __init__(self, slots=2):
        self.slots = BoundedSemaphore(slots)

    @contextmanager
    def work(self):
        held = self.slots.acquire(timeout=16)
        if not held:
            raise BenchError("有其他扫描正在处理。请稍后重试。", 503, "bench_busy")
        try:
            yield
        finally:
            self.slots.release()


def library_routes(app, catalog, gate):
    @app.get("/")
    def home():
        return send_from_directory(STATIC, "index.html")

    @app.get("/ui/<path:filename>")
    def static_ui(filename):
        return send_from_directory(STATIC, filename)

    @app.get("/api/health")
    def health():
        return jsonify(status="ready", version=__version__)

    @app.get("/api/library")
    def library():
        return jsonify(reels=catalog.reels(), configuration={"max_upload": FILE_BYTES, "max_width": MAX_WIDTH,
                       "max_height": MAX_HEIGHT, "max_pixels": MAX_PIXELS, "rates": RATES,
                       "max_monitor_seconds": MAX_MONITOR_SECONDS, "tools": list(TOOLS)})

    @app.post("/api/training")
    def training():
        with gate.work():
            result = seed_library(catalog)
            created = any(item["added"] for item in result["training"])
            return jsonify(result), 201 if created else 200

    @app.post("/api/reels")
    def create_reel():
        return jsonify(reel=catalog.create_reel(json_body())), 201

    @app.get("/api/reels/<reel_id>")
    def describe_reel(reel_id):
        reel = catalog.get_reel(reel_id)
        return jsonify(reel=reel, scans=catalog.list_scans(reel_id), packets=catalog.list_packets(reel_id))

    @app.get("/api/reels/<reel_id>/events")
    def reel_events(reel_id):
        limit = integer(request.args.get("limit", 24), "每页事件数", 1, 60)
        before = request.args.get("before")
        before = integer(before, "事件分页位置", 1, 2_147_483_647) if before is not None else None
        events = catalog.audit(reel_id, before, limit)
        return jsonify(events=events, next=events[-1]["id"] if len(events) == limit else None)

    @app.post("/api/reels/<reel_id>/scans")
    def import_scan(reel_id):
        reel = catalog.get_reel(reel_id)
        content, name = read_upload("scan", FILE_BYTES)
        fallback = safe_filename(name).rsplit(".", 1)[0]
        title = text(request.form.get("title") or fallback, "扫描名", maximum=110)
        note = text(request.form.get("note", ""), "接收说明", maximum=700, required=False, multiline=True)
        with gate.work():
            raster = read_scan(content, name)
            initial = Recipe.default(raster.width).as_dict()
            initial["clock"]["holes_per_frame"] = 4 if reel["gauge"] == "35 mm" else 1
            recipe = Recipe.parse(initial, raster.width)
            scan, fresh = catalog.add_scan(reel_id, raster, title, note, recipe)
        return jsonify(scan=scan, duplicate=not fresh), 201 if fresh else 200

    @app.post("/api/reels/<reel_id>/intake")
    def preflight(reel_id):
        reel = catalog.get_reel(reel_id)
        content, name = read_upload("scan", FILE_BYTES)
        with gate.work():
            raster = read_scan(content, name)
            known = next((scan for scan in catalog.list_scans(reel_id) if scan["sha256"] == raster.digest), None)
            recipe = Recipe.default(raster.width)
            preview = source_overview(raster.intensity, recipe.aperture, recipe.perforations, recipe.orientation, size=(220, 450))
        return jsonify(metadata=raster.metadata(), duplicate=known, preview=encode_data(preview, "image/png"),
                       suggested_gauge=reel["gauge"], draft_recipe=recipe.as_dict())

    @app.post("/api/reels/<reel_id>/scan-batch")
    def scan_batch(reel_id):
        content, _ = read_upload("batch", BATCH_BYTES)
        work = option(request.form.get("action", "preflight"), ("preflight", "commit"), "扫描批量操作")
        with gate.work():
            result = inspect_upload(catalog, reel_id, content, request.form.get("sha256"), making=work == "commit")
        for item in result["files"]:
            if item.get("preview") is not None:
                item["preview"] = encode_data(item["preview"], "image/png")
        return jsonify(result), 201 if result.get("added_count") else 200


def source_routes(app, catalog, gate):
    @app.get("/api/scans/<scan_id>")
    def scan(scan_id):
        scan = catalog.get_scan(scan_id)
        active = catalog.get_recipe(scan_id, want_recipe())
        most_recent = catalog.get_recipe(scan_id)
        return jsonify(scan=scan, recipe=active, is_latest=active["id"] == most_recent["id"], latest=most_recent["seq"],
                       markers=catalog.list_markers(scan_id))

    @app.get("/api/scans/<scan_id>/technical")
    def technical_receipt(scan_id):
        mode = option(request.args.get("format", "json"), ("json", "download"), "技术资料格式")
        with gate.work():
            material = load_material(catalog, scan_id, want_recipe())
            report = inspect_source(material)
        if mode == "json":
            return jsonify(report)
        basename = shipment_stem(material.record["title"], "scan")
        return send_content(json_bytes(report), "application/json", f"{basename}-receipt.json", attach=True)

    @app.get("/api/scans/<scan_id>/original")
    def original(scan_id):
        scan = catalog.get_scan(scan_id, raw=True)
        meta = scan["metadata"]
        return send_content(scan["raw"], meta["mime"], meta["filename"], attach=True, etag=scan["sha256"])

    @app.get("/api/scans/<scan_id>/tile.png")
    def tile(scan_id):
        with gate.work():
            material = load_material(catalog, scan_id, want_recipe())
            box = query_box(request.args, material.width, material.height, 460)
            if box.area > 2_200_000 or box.height > 3000:
                raise Limit("原片切片最多 2,200,000 像素且不超过 3000 行。")
            side = integer(request.args.get("view", 750), "预览尺寸", 140, 1200)
            style = option(request.args.get("tonal", "native"), ("native", "stretch"), "显示色调")
            tone = material.recipe.tone
            stretched = (tone.black, tone.white) if style == "stretch" else None
            image = source_tile(material.pixels, box, material.recipe.orientation, stretched=stretched, size=(side, side))
        response = send_content(image, "image/png", "scan-tile.png")
        response.headers["X-ReelTone-Source-Box"] = ",".join(str(value) for value in box.as_list())
        return response

    @app.get("/api/scans/<scan_id>/overview.png")
    def overview(scan_id):
        with gate.work():
            material = load_material(catalog, scan_id, want_recipe())
            recipe = material.recipe
            preview = source_overview(material.pixels, recipe.aperture, recipe.perforations, recipe.orientation)
        return send_content(preview, "image/png", "reel-overview.png")

    @app.get("/api/scans/<scan_id>/recipes")
    def history(scan_id):
        return jsonify(recipes=catalog.recipe_history(scan_id))

    @app.post("/api/scans/<scan_id>/recipe")
    def save_recipe(scan_id):
        saved, created = save_settings(catalog, scan_id, json_body())
        return jsonify(recipe=saved, created=created), 201 if created else 200

    @app.post("/api/scans/<scan_id>/recipes/<recipe_id>/restore")
    def restore_recipe(scan_id, recipe_id):
        saved, created = restore_settings(catalog, scan_id, recipe_id, json_body())
        return jsonify(recipe=saved, created=created), 201 if created else 200

    @app.post("/api/scans/<scan_id>/markers")
    def marker(scan_id):
        data = json_body()
        with gate.work():
            material = load_material(catalog, scan_id, data.get("recipe") or None)
            record = save_marker(catalog, material, data)
        return jsonify(marker=record), 201

    @app.post("/api/scans/<scan_id>/marker-view")
    def locate_marker(scan_id):
        with gate.work():
            return jsonify(restore_view(catalog, scan_id, json_body()))

    @app.post("/api/scans/<scan_id>/contact")
    def contact(scan_id):
        data = json_body()
        with gate.work():
            material = load_material(catalog, scan_id, data.get("recipe") or None)
            compiled = make_contact(material, data)
        return jsonify(reading=compiled.manifest, preview=encode_data(compiled.plate, "image/png"),
                       pdf=encode_data(compiled.pdf, "application/pdf"))


def print_routes(app, catalog, gate):
    @app.post("/api/reels/<reel_id>/physical-plan")
    def physical_plan(reel_id):
        values = json_body()
        with gate.work():
            manifest, previews = plan_physical(catalog, reel_id, values)
        return jsonify(reading=manifest, previews=[encode_data(item, "image/png") for item in previews])

    @app.post("/api/reels/<reel_id>/physical-print")
    def physical_print(reel_id):
        values = json_body()
        with gate.work():
            manifest, previews, pdf = render_physical(catalog, reel_id, values)
        if pdf is None:
            # Not a protocol error: the requested magnification does not fit and
            # the manifest carries the conflicts and the largest usable ratio.
            return jsonify(reading=manifest, previews=[], pdf=None)
        return jsonify(reading=manifest, previews=[encode_data(item, "image/png") for item in previews],
                       pdf=encode_data(pdf, "application/pdf"))


def analysis_routes(app, catalog, gate):
    @app.post("/api/scans/<scan_id>/audition")
    def listen(scan_id):
        data = json_body()
        retain = flag(data.get("retain", False), "保存监听")
        with gate.work():
            material = load_material(catalog, scan_id, data.get("recipe") or None)
            if retain:
                return jsonify(make_reading(catalog, material, data)), 201
            report, wave, trace, wav = render_audition(material, data)
            return jsonify(report=report, waveform=wave, trace=trace, audio=encode_data(wav, "audio/wav"))

    @app.get("/api/scans/<scan_id>/takes")
    def takes(scan_id):
        return jsonify(takes=catalog.list_takes(scan_id))

    @app.get("/api/takes/<take_id>")
    def take(take_id):
        with gate.work():
            return jsonify(current_take_properties(catalog, take_id))

    @app.get("/api/takes/<take_id>/wav")
    def take_wav(take_id):
        stored = catalog.get_take(take_id, raw=True)
        name = f"{shipment_stem(stored['title'])}.wav"
        return send_content(stored["wav"], "audio/wav", name, attach=request.args.get("download") == "1", etag=stored["wav_sha256"])

    @app.post("/api/takes/<take_id>/status")
    def take_state(take_id):
        return jsonify(take=catalog.set_take_status(take_id, json_body().get("status")))

    @app.post("/api/scans/<scan_id>/instrument/<name>")
    def instrument(scan_id, name):
        data = json_body()
        retain = flag(data.get("retain", False), "保存图像证据")
        with gate.work():
            material = load_material(catalog, scan_id, data.get("recipe") or None)
            result = run_tool(material, name, data, catalog)
            summary = result.summary()
            if not retain:
                return jsonify(**summary, preview=encode_data(result.preview, "image/png"),
                               companion=encode_data(result.companion, "image/png"), can_retain=result.category is not None)
            if result.category is None or result.preview is None:
                raise BenchError("该工具的临时结果可用于校准，不能保存为证据图。")
            title = text(data.get("title", ""), "证据名称", maximum=90)
            note = text(data.get("note", ""), "证据备注", maximum=650, required=False, multiline=True)
            record = catalog.save_proof(scan_id, material.recipe_record["id"], title, result.category, note,
                                        result.reading, result.preview, result.companion)
        return jsonify(result=summary, proof=record), 201

    @app.get("/api/scans/<scan_id>/proofs")
    def proofs(scan_id):
        return jsonify(proofs=catalog.list_proofs(scan_id))

    @app.get("/api/proofs/<proof_id>")
    def proof(proof_id):
        with gate.work():
            return jsonify(current_proof_properties(catalog, proof_id))

    @app.get("/api/proofs/<proof_id>/<part>.png")
    def proof_image(proof_id, part):
        if part not in ("primary", "secondary"):
            raise Absent("该证据图")
        proof = catalog.get_proof(proof_id, raw=True)
        body = proof[f"{part}_file"]
        if body is None:
            raise Absent("配套证据图")
        return send_content(body, "image/png", f"{shipment_stem(proof['title'])}-{part}.png",
                            attach=request.args.get("download") == "1", etag=proof[f"{part}_sha256"])

    @app.post("/api/scans/<scan_id>/ab")
    def ab_pair(scan_id):
        with gate.work():
            report, wav, figure = retained_pair(catalog, scan_id, json_body())
        return jsonify(report=report, audio=encode_data(wav, "audio/wav"), preview=encode_data(figure, "image/png"))

    @app.post("/api/takes/<take_id>/replay")
    def replay(take_id):
        with gate.work():
            result, wav, figure = retained_replay(catalog, take_id, json_body())
        return jsonify(result=result, audio=encode_data(wav, "audio/wav"), preview=encode_data(figure, "image/png"))


def review_routes(app, catalog, gate):
    @app.get("/api/reels/<reel_id>/cues")
    def cues(reel_id):
        scan_id = request.args.get("scan") or None
        if scan_id is not None and catalog.get_scan(scan_id)["reel_id"] != reel_id:
            raise Absent("档案卷下的扫描")
        cue_list = catalog.list_cues(reel_id, scan_id)
        rows = None
        if scan_id is not None and "from" in request.args and "to" in request.args:
            scan = catalog.get_scan(scan_id)
            rows = split_rows([request.args["from"], request.args["to"]], scan["metadata"]["height"])
        if rows:
            filtered = [item for item in cue_list if max(item["y0"], rows[0]) < min(item["y1"], rows[1])]
        else:
            filtered = cue_list
        return jsonify(cues=filtered, coverage=cue_coverage(filtered), rows=rows)

    @app.post("/api/scans/<scan_id>/cues")
    def add_cue(scan_id):
        return jsonify(cue=save_cue(catalog, scan_id, json_body())), 201

    @app.put("/api/scans/<scan_id>/cues/<cue_id>")
    def update_cue(scan_id, cue_id):
        return jsonify(cue=save_cue(catalog, scan_id, json_body(), cue_id))

    @app.post("/api/reels/<reel_id>/packets")
    def pack(reel_id):
        values = json_body()
        with gate.work():
            return jsonify(packet=build_packet(catalog, reel_id, values)), 201

    @app.get("/api/packets/<packet_id>/verify")
    def verify_retained_packet(packet_id):
        with gate.work():
            stored = catalog.get_packet(packet_id, raw=True)
            result = verify_packet(stored["zip_data"])
            result["receipt_matches"] = result["manifest_sha256"] == stored["summary"]["manifest_sha256"]
        return jsonify(result)

    @app.get("/api/packets/<packet_id>/download")
    def archive(packet_id):
        packet = catalog.get_packet(packet_id, raw=True)
        return send_content(packet["zip_data"], "application/zip", packet["filename"], attach=True, etag=packet["sha256"])

    @app.post("/api/packets/verify")
    def validate_packet():
        content, _ = read_upload("packet", MAX_PACKET_BYTES)
        with gate.work():
            return jsonify(verify_packet(content))

    @app.get("/api/reels/<reel_id>/readiness")
    def readiness(reel_id):
        with gate.work():
            return jsonify(from_catalog(catalog, reel_id))

    @app.get("/api/reels/<reel_id>/readiness/<kind>")
    def readiness_export(reel_id, kind):
        if kind not in {"png", "csv", "html"}:
            raise Absent("此格式的扫描审听报告")
        with gate.work():
            report = from_catalog(catalog, reel_id)
            data, media, filename, digest = export_review(report, kind)
        return send_content(data, media, filename, attach=kind == "csv", etag=digest)

    @app.get("/api/proofs/<proof_id>/reading.json")
    def proof_reading(proof_id):
        proof = catalog.get_proof(proof_id)
        filename = f"{shipment_stem(proof['title'])}-reading.json"
        return send_content(json_bytes(proof), "application/json", filename, attach=True)


def create_app(directory=None):
    app = Flask(__name__, static_folder=None)
    app.config.update(MAX_CONTENT_LENGTH=MAX_PACKET_BYTES + 384 * 1024, MAX_FORM_MEMORY_SIZE=450 * 1024,
                      MAX_FORM_PARTS=24, JSON_SORT_KEYS=False)
    catalog = Catalog(directory or os.environ.get("REELTONE_DATA", ".reeltone-data"))
    gate = ComputationGate()
    app.extensions["reeltone_catalog"] = catalog
    library_routes(app, catalog, gate)
    source_routes(app, catalog, gate)
    print_routes(app, catalog, gate)
    analysis_routes(app, catalog, gate)
    review_routes(app, catalog, gate)

    @app.errorhandler(BenchError)
    def user_error(error):
        return jsonify(error=error.message, code=error.code), error.status

    @app.errorhandler(RequestEntityTooLarge)
    def oversized(error):
        return jsonify(error="上传内容超过工作台的处理上限。", code="limit"), 413

    @app.errorhandler(404)
    def not_found(error):
        if request.path.startswith("/api/"):
            return jsonify(error="找不到这个工作台地址。", code="not_found"), 404
        return Response("ReelTone · Page not found", status=404, mimetype="text/plain")

    return app
