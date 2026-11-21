from dataclasses import dataclass
from hashlib import sha256

from .audition import read_wav
from .bench import Material, analyze_track, load_material
from .blemishes import MAX_QC_PIXEL_COUNT, mark_overlaps, scan_blemishes
from .comparison import Pair, compare_exposure, paired_line_profile
from .contracts import identifier, integer, option, real, text
from .errors import Absent, BenchError, Conflict, Limit
from .exposure import density_rolloff, exposure_report, reference_suggestion
from .geometry import Box
from .loupe import overlay_contour, pixel_grid, probe_at
from .optical import read_area, scan_trace, track_column_search
from .perforations import inspect_perfs
from .raster import encode_png
from .spectrum import narrowband_profile
from .sensitometry import assess_steps
from .splices import inspect_splices
from .slit import examine_slit
from .timing import clock_from_dict
from .pilot import assay
from .visual import (
    ACCENT, BLUE, TEAL, PlotSeries, blemish_heat, evidence_plate, exposure_heat,
    line_plot, luminance_triptych, spectrum_png,
)


@dataclass(frozen=True)
class Result:
    kind: str
    reading: dict
    preview: bytes | None = None
    companion: bytes | None = None
    category: str | None = None

    def summary(self):
        extra = {}
        if self.preview is not None:
            extra["preview_sha256"] = sha256(self.preview).hexdigest()
        if self.companion is not None:
            extra["companion_sha256"] = sha256(self.companion).hexdigest()
        return {"kind": self.kind, "reading": self.reading, **extra}


def crop(material, request, catalog):
    box = material.region(request, fallback=Box(0, 0, material.width, min(560, material.height)))
    if box.area > 1_500_000:
        raise Limit("单张像素证据最多 150 万原生像素。")
    view = material.recipe.orientation
    pixels = box.subimage(material.pixels)
    record = {"parent": material.identity(), "box": box.as_list(), "orientation": view.as_dict(),
              "archive_crop_orientation": "native", "sample_bit_depth": material.raster.bit_depth,
              "comparison_preserves_native_pixels": True}
    title = text(request.get("title", "Source crop"), "证据名称", maximum=90)
    labels = (("SOURCE", material.record["metadata"]["filename"]), ("SHA256", material.record["sha256"][:20]),
              ("RECIPE", f"v{material.recipe_record['seq']}"))
    preview, layout = evidence_plate(material.pixels, box, view, title, aperture=material.recipe.aperture,
                                     perfs=material.recipe.perforations, metadata=labels)
    record["plate_layout"] = layout
    native = encode_png(pixels, bits=material.raster.bit_depth)
    return Result("crop", record, preview, native, "image")


def exposure(material, request, catalog):
    box = material.region(request)
    report = exposure_report(material.pixels, box)
    chart = exposure_heat(report["tiles"])
    clip_chart = exposure_heat(report["tiles"], clipping=True)
    stats = report.pop("statistics")
    report["stats"] = stats
    report["cross_light"] = density_rolloff(box.subimage(material.pixels))
    report["parent"] = material.identity()
    return Result("exposure", report, chart, clip_chart, "exposure")


def references(material, request, catalog):
    a_box = Box.parse(request.get("dark"), material.width, material.height, minimum=(4, 6), name="暗参照")
    b_box = Box.parse(request.get("light"), material.width, material.height, minimum=(4, 6), name="亮参照")
    if a_box.area + b_box.area > 580_000:
        raise Limit("参照取样合计最多 58 万像素。")
    if a_box.intersection(b_box):
        raise BenchError("明暗参照框须相互分离。")
    output = reference_suggestion(a_box.subimage(material.pixels), b_box.subimage(material.pixels))
    return Result("references", {"parent": material.identity(), "dark": a_box.as_list(), "light": b_box.as_list(), **output})


def perforations(material, request, catalog):
    rows = material.rows(request, min_span=24)
    setting = material.recipe.clock
    pitch = real(request.get("pitch_hint", setting.pitch_guess), "候选孔距", 8, 5000)
    report = inspect_perfs(material.pixels, material.recipe.perforations, pitch, rows=rows)
    data = report.as_dict(models=True)
    data["parent"] = material.identity()
    data["guide"] = material.recipe.perforations.as_list()
    data["nominal_pitch"] = pitch
    records = data["spacing"]
    range_pitch = [item["pitch"] for item in records if item["trusted"]]
    limits = (min([pitch * 0.77, *range_pitch]), max([pitch * 1.24, *range_pitch]))
    tracks = [PlotSeries(TEAL, records, lambda item: item["pitch"] if item["trusted"] else None,
                         lambda item: item["first"], "Confirmed interval"),
              PlotSeries(ACCENT, records, lambda item: item["pitch"] if not item["trusted"] else None,
                         lambda item: item["first"], "Uncertain interval")]
    plot = line_plot("PERFORATION PITCH · NATIVE ROWS", tracks, x_range=rows, y_range=limits,
                     formatter=lambda a: f"row {a:.0f}", y_formatter=lambda a: f"{a:.1f} px")
    return Result("perforations", data, plot, category="perforations")


def locate(material, request, catalog):
    rows = material.rows(request, min_span=40)
    recipe = material.recipe
    guess = integer(request.get("width_hint", recipe.aperture.width), "预计读窗宽", 8, material.width)
    result = track_column_search(material.pixels, rows, recipe.perforations, recipe.aperture, guess)
    result["parent"] = material.identity()
    profile = list(enumerate(result["column_modulation"]))
    plot = line_plot("TRACK MODULATION BY SOURCE COLUMN", [PlotSeries(TEAL, profile, lambda row: row[1], lambda row: row[0], "Observed range")],
                     x_range=(0, material.width - 1), y_range=(0, max(0.1, max(result["column_modulation"]))),
                     formatter=lambda x: f"x {x:.0f}", y_formatter=lambda y: f"{y:.2f}")
    return Result("locate", result, plot)


def blemish(material, request, catalog):
    region = material.region(request, fallback=material.recipe.aperture.box(0, min(700, material.height)), minimum=(3, 5))
    if region.area > MAX_QC_PIXEL_COUNT:
        raise Limit("瑕疵巡检每次最多读取 250 万原生像素。")
    method = integer(request.get("filter_width", 7), "邻域尺寸", 3, 15)
    if method % 2 == 0:
        raise BenchError("邻域尺寸须为奇数。")
    limit = real(request.get("contrast", 0.19), "局部对比阈值", 0.03, 0.92)
    pixels = integer(request.get("min_pixels", 6), "最小候选像素数", 2, 500)
    result = scan_blemishes(material.pixels, region, threshold=limit, support=method, min_pixels=pixels)
    data = result.summary()
    data.update({"parent": material.identity(), "neighborhood": method, "minimum_pixels": pixels,
                 "overlap_with_guides": mark_overlaps([material.recipe.aperture.box(0, material.height),
                                                        material.recipe.perforations.box(0, material.height)], region)})
    return Result("blemish", data, blemish_heat(material.pixels, result, material.recipe.orientation), category="blemish")


def compare(material, request, catalog):
    other_id = identifier(request.get("other_scan"), "另一张扫描")
    if other_id == material.record["id"]:
        raise BenchError("请选择同卷的另一张源扫描。")
    other = load_material(catalog, other_id)
    if other.record["reel_id"] != material.record["reel_id"]:
        raise Conflict("对照扫描须属于同一个档案卷。")
    a_default = Box(0, 0, material.width, min(660, material.height))
    a = material.region(request, fallback=a_default, minimum=(4, 6))
    b = Box.parse(request.get("other_box"), other.width, other.height, minimum=(4, 6), name="对照片范围")
    if a.area + b.area > 3_600_000:
        raise Limit("对照选区合计最多 360 万像素。")
    selection = Pair(a, b, material.record["id"], other.record["id"])
    comparison = compare_exposure(material.pixels, other.pixels, selection)
    report = comparison.summary()
    report.update({"parent": material.identity(), "reference": other.identity(), "related_scans": [other.record["id"]],
                   "row_profiles": paired_line_profile(comparison)})
    return Result("compare", report, luminance_triptych(comparison), category="comparison")


def sample(material, request, catalog):
    x = integer(request.get("x"), "原片列", 0, material.width - 1)
    y = integer(request.get("y"), "原片行", 0, material.height - 1)
    radius = integer(request.get("radius", 5), "像素近邻半径", 2, 16)
    data, probe = probe_at(material.pixels, material.recipe, material.raster.bit_depth, x, y, radius,
                           multichannel=material.raster.channel_count != 1)
    data["parent"] = material.identity()
    group = list(enumerate(data["row_profile"]))
    chart = line_plot("CROSS-STRIP PIXEL PROFILE", [PlotSeries(BLUE, group, lambda row: row[1], lambda row: row[0], f"Source row {y}")],
                      x_range=(0, material.width - 1), y_range=(0, 1), formatter=lambda a: f"x {a:.0f}", y_formatter=lambda a: f"{a:.2f}")
    return Result("sample", data, pixel_grid(probe, material.recipe.orientation), chart, "sample")


def contour(material, request, catalog):
    rows = material.rows(request, min_span=8)
    aperture, tone = material.recipe.aperture, material.recipe.tone
    reading = read_area(material.pixels, aperture, tone, rows)
    traits = reading.summary()
    bins = scan_trace(reading, size=360)
    traces = [PlotSeries(BLUE, bins, lambda item: item["min"], lambda item: item["rows"][0], "Minimum aperture"),
              PlotSeries(ACCENT, bins, lambda item: item["max"], lambda item: item["rows"][0], "Maximum aperture"),
              PlotSeries(TEAL, bins, lambda item: item["mean"], lambda item: item["rows"][0], "Mean aperture")]
    chart = line_plot("OPTICAL OPENING IN EACH SOURCE-ROW BIN", traces, x_range=rows, y_range=(0, 1),
                      formatter=lambda a: f"{a:.0f}", y_formatter=lambda a: f"{a:.0%}")
    preview = overlay_contour(material.pixels, reading, aperture, material.recipe.orientation, top=rows[0], stop=min(rows[0] + 720, rows[1]))
    return Result("contour", {"parent": material.identity(), "summary": traits, "trace": bins,
                              "guide": aperture.as_list(), "detail_rows": [rows[0], min(rows[0] + 720, rows[1])]}, chart, preview, "scope")


def scope(material: Material, request, catalog):
    resolution = option(request.get("resolution", "standard"), ("compact", "standard", "detail"), "频谱窗口")
    take_id = request.get("take") or None
    if take_id:
        stored = catalog.get_take(identifier(take_id, "监听编号"), raw=True)
        if stored["scan_id"] != material.record["id"]:
            raise Conflict("监听快照不属于该扫描。")
        if stored["recipe_id"] != material.recipe_record["id"]:
            raise Conflict("请先打开该监听使用的读片配方。")
        waveform, rate = read_wav(stored["wav"])
        clock = clock_from_dict(stored["payload"]["clock"])
        source_rows = stored["payload"]["audio"]["source_rows"]
        related = {"kind": "retained", "take_id": stored["id"], "take_sha256": stored["wav_sha256"]}
    else:
        source_rows, _, _, recorded, _ = analyze_track(material, request)
        waveform, rate, clock = recorded.wave, recorded.render_rate, recorded.clock
        related = {"kind": "live"}
    spectrum, result = narrowband_profile(waveform, rate, clock, resolution, source_rows)
    time0, _ = clock.extent(source_rows)
    plot = spectrum_png(spectrum, time_origin=time0)
    data = {"parent": material.identity(), "source_rows": list(source_rows), "signal": related, "window": resolution,
            "clock": clock.as_dict(source_rows), **result}
    return Result("scope", data, plot, category="scope")


def exposure_wedge(material, request, catalog):
    baseline = option(request.get("baseline", "fit"), ("fit", "fixed"), "黑点模型")
    readings, plate = assess_steps(material.pixels, request.get("steps"), baseline)
    readings["parent"] = material.identity()
    return Result("wedge", readings, plate, category="exposure")


def splice_search(material, request, catalog):
    readings, plate, plot = inspect_splices(material, request)
    return Result("splices", readings, plate, plot, "image")


def slit_geometry(material, request, catalog):
    readings, chart = examine_slit(material, request)
    return Result("slit", readings, chart, category="scope")


def test_tone(material, request, catalog):
    readings, chart, admission = assay(material, request)
    return Result("pilot", readings, chart, admission, "scope")


TOOLS = {"crop": crop, "exposure": exposure, "references": references, "perforations": perforations, "locate": locate,
         "blemish": blemish, "compare": compare, "sample": sample, "contour": contour, "scope": scope,
         "wedge": exposure_wedge, "splices": splice_search, "slit": slit_geometry, "pilot": test_tone}


def run_tool(material, name, request, catalog):
    if name not in TOOLS:
        raise Absent("该巡检工具")
    return TOOLS[name](material, request, catalog)
