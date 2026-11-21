from dataclasses import dataclass

from .audition import Monitor, audition, wav_bytes, waveform_envelope
from .contracts import integer, mapping, split_rows, text
from .errors import Conflict, Limit
from .geometry import Box
from .optical import read_area, scan_trace
from .perforations import inspect_perfs
from .raster import read_scan
from .recipe import Recipe
from .timing import clock_table, establish_clock


@dataclass(frozen=True)
class Material:
    record: dict
    raster: object
    recipe_record: dict
    recipe: Recipe

    @property
    def pixels(self):
        return self.raster.intensity

    @property
    def width(self):
        return self.raster.width

    @property
    def height(self):
        return self.raster.height

    @property
    def bounds(self):
        return Box(0, 0, self.width, self.height)

    def identity(self):
        return {"scan_id": self.record["id"], "source_sha256": self.record["sha256"], "width": self.width,
                "height": self.height, "recipe_id": self.recipe_record["id"], "recipe_number": self.recipe_record["seq"],
                "recipe_fingerprint": self.recipe_record["fingerprint"]}

    def clock(self):
        setting = self.recipe.clock
        report = None
        if setting.model in ("perfs", "local"):
            report = inspect_perfs(self.pixels, self.recipe.perforations, setting.pitch_guess)
        return establish_clock(setting, self.height, self.recipe.orientation.reverse_y, report), report

    def rows(self, data, min_span=1, max_span=None):
        proposed = data.get("rows")
        default = (0, self.height if max_span is None else min(max_span, self.height))
        rows = split_rows(proposed, self.height, min_span=min_span, default=default)
        if max_span is not None and rows[1] - rows[0] > max_span:
            raise Limit(f"该工具一次最多读取 {max_span} 行。缩小原片范围再试。")
        return rows

    def region(self, data, fallback=None, minimum=(1, 1)):
        if data.get("box") is None:
            if fallback is not None:
                return fallback
            return self.bounds
        return Box.parse(data["box"], self.width, self.height, minimum=minimum)


def load_material(catalog, scan_id, recipe_id=None):
    record = catalog.get_scan(scan_id, raw=True)
    recipe_record = catalog.get_recipe(scan_id, recipe_id)
    raster = read_scan(record.pop("raw"), record["metadata"]["filename"])
    if raster.digest != record["sha256"]:
        raise Conflict("原始扫描内容与保存的摘要不一致。")
    recipe = Recipe.parse(recipe_record["settings"], raster.width)
    if recipe.fingerprint() != recipe_record["fingerprint"]:
        raise Conflict("读片配方内容与保存的摘要不一致。")
    return Material(record, raster, recipe_record, recipe)


def save_settings(catalog, scan_id, values):
    values = mapping(values, "读片设置请求")
    meta = catalog.get_scan(scan_id)["metadata"]
    recipe = Recipe.parse(values.get("settings"), meta["width"])
    label = text(values.get("label", "手工读片校准"), "配方名", maximum=100)
    expected = integer(values.get("expected"), "现行配方版本", 1, 200)
    return catalog.save_recipe(scan_id, recipe, label, "manual", expected)


def restore_settings(catalog, scan_id, recipe_id, values):
    current = catalog.get_scan(scan_id)
    expected = integer(values.get("expected"), "现行配方版本", 1, 200)
    source = catalog.get_recipe(scan_id, recipe_id)
    recipe = Recipe.parse(source["settings"], current["metadata"]["width"])
    return catalog.save_recipe(scan_id, recipe, f"复用 v{source['seq']} · {source['label'][:75]}", "restored", expected)


def analyze_track(material, data):
    rows = material.rows(data, min_span=8)
    recipe = material.recipe
    clock, perfs = material.clock()
    settings = Monitor.parse(data.get("monitor", {}))
    reading = read_area(material.pixels, recipe.aperture, recipe.tone, rows)
    result = audition(reading, clock, settings)
    return rows, settings, reading, result, perfs


def render_audition(material, data):
    rows, settings, reading, result, perfs = analyze_track(material, data)
    report = {"parent": material.identity(), "audio": result.summary(), "read": reading.summary(),
              "clock": result.clock.as_dict(rows), "monitor": settings.as_dict(), "time_marks": clock_table(result.clock, rows),
              "perforations": perfs.as_dict(maximum=30) if perfs else None,
              "aperture": material.recipe.aperture.as_list(), "tone": material.recipe.tone.as_dict()}
    if result.clock.local:
        report["clock_rows"] = local_clock_rows(result.clock, rows)
    waves = waveform_envelope(result)
    curves = scan_trace(reading, size=340)
    return report, waves, curves, wav_bytes(result.wave, result.render_rate)


def local_clock_rows(clock, rows):
    top, bottom = rows
    marks = []
    for entry in clock.segments:
        if entry.native_rows is None:
            continue
        low, high = entry.native_rows
        if high < top or low > bottom:
            continue
        marks.append({"native_rows": [max(low, top), min(high, bottom)], "rate": round(entry.rate, 6),
                      "trusted": entry.trusted})
    marks.sort(key=lambda entry: entry["native_rows"][0])
    return marks


def make_reading(catalog, material, data):
    title = text(data.get("title", "声轨试听"), "试听名称", maximum=90)
    report, waves, curves, wave = render_audition(material, data)
    record = catalog.save_take(material.record["id"], material.recipe_record["id"], title, report["monitor"], report, wave)
    return {"take": record, "waveform": waves, "trace": curves}
