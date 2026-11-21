from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .blemishes import BlemishMap
from .comparison import WITHHELD, Comparison
from .geometry import Box, Orientation
from .raster import byte_preview, encode_image, fitted_preview
from .spectrum import Spectrum


PAGE = "#111923"
PANEL = "#18232e"
GRID = "#364856"
TEXT = "#f2f5eb"
MUTED = "#a5b3bc"
ACCENT = "#e9b76b"
TEAL = "#77ded1"
RED = "#f48680"
BLUE = "#83a9eb"


@lru_cache(maxsize=16)
def available_font(size):
    for face in ("DejaVuSans.ttf", "LiberationSans-Regular.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(face, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def print_text(draw, xy, content, size=15, fill=TEXT, *, anchor=None):
    draw.text(xy, str(content), font=available_font(size), fill=fill, anchor=anchor)


def interpolate(value, minimum, maximum, left, right):
    return left + (value - minimum) / (maximum - minimum) * (right - left)


def wrapped_ascii(value, width):
    value = value.encode("ascii", "replace").decode("ascii") if value else ""
    words = value.split()
    lines = []
    active = ""
    for word in words:
        pieces = [word[a:a + width] for a in range(0, len(word), width)]
        for piece in pieces:
            if active and len(active) + 1 + len(piece) > width:
                lines.append(active)
                active = piece
            else:
                active = f"{active} {piece}".strip()
    if active:
        lines.append(active)
    return lines


@dataclass(frozen=True)
class Window:
    native_box: Box
    canvas_box: Box
    orientation: Orientation
    width: int
    height: int

    def point(self, x, y):
        native_x = self.width - x if self.orientation.mirror_x else x
        native_y = self.height - y if self.orientation.reverse_y else y
        display = self.orientation.box(self.native_box, self.width, self.height)
        shown = self.canvas_box
        return (round(interpolate(native_x, display.x0, display.x1, shown.x0, shown.x1)),
                round(interpolate(native_y, display.y0, display.y1, shown.y0, shown.y1)))

    def rect(self, native):
        a, b = self.point(native.x0, native.y0)
        c, d = self.point(native.x1, native.y1)
        return min(a, c), min(b, d), max(a, c), max(b, d)

    def scale(self):
        return {"native_box": self.native_box.as_list(), "image_box": self.canvas_box.as_list(),
                "orientation": self.orientation.as_dict()}


def source_tile(values, native_box, orientation, *, stretched=None, size=(1100, 1000)):
    clip = native_box.subimage(values)
    oriented = orientation.image(clip)
    lo, hi = stretched if stretched is not None else (0.0, 1.0)
    preview = fitted_preview(byte_preview(oriented, lo=lo, hi=hi), *size)
    return encode_image(preview, "PNG")


def source_overview(values, aperture, perfs, orientation, size=(340, 620)):
    preview = byte_preview(orientation.image(values))
    height, width = values.shape
    width_px = min(size[0], width)
    height_px = min(size[1], height)
    image = preview.resize((width_px, height_px), Image.Resampling.BOX).convert("RGB")
    draw = ImageDraw.Draw(image)
    for guide, color in ((aperture, TEAL), (perfs, ACCENT)):
        view = orientation.guide(guide, width)
        for edge in (view.x0, view.x1):
            x = min(width_px - 1, max(0, round(edge * width_px / width)))
            draw.line((x, 0, x, height_px - 1), fill=color, width=2)
    return encode_image(image, "PNG")


def place_film(page, values, native_box, orientation, area, *, tone=(0.0, 1.0), stretch=False):
    tile = orientation.image(native_box.subimage(values))
    source_image = byte_preview(tile, lo=tone[0], hi=tone[1])
    if stretch:
        bitmap = source_image.resize((area.width, area.height), Image.Resampling.BOX)
    else:
        bitmap = fitted_preview(source_image, area.width, area.height)
    origin = (area.x0 + (area.width - bitmap.width) // 2, area.y0 + (area.height - bitmap.height) // 2)
    page.paste(bitmap.convert("RGB"), origin)
    canvas = Box(origin[0], origin[1], origin[0] + bitmap.width, origin[1] + bitmap.height)
    return Window(native_box, canvas, orientation, values.shape[1], values.shape[0])


def native_marks(draw, window, rectangles):
    for native, color, line_width in rectangles:
        intersection = native.intersection(window.native_box)
        if intersection is None:
            continue
        left, top, right, bottom = window.rect(intersection)
        if top == bottom or left == right:
            continue
        draw.rectangle((left, top, right - 1, bottom - 1), outline=color, width=line_width)


def film_rulers(draw, geometry, desired=6):
    tile = geometry.canvas_box
    box = geometry.native_box
    for row in np.rint(np.linspace(box.y0, box.y1, desired)).astype(int):
        _, y = geometry.point(box.x0, row)
        draw.line((tile.x0 - 5, y, tile.x0 - 1, y), fill=MUTED)
        print_text(draw, (tile.x0 - 8, y), row, 11, MUTED, anchor="rm")
    for col in np.rint(np.linspace(box.x0, box.x1, 5)).astype(int):
        x, _ = geometry.point(col, box.y0)
        draw.line((x, tile.y1 + 1, x, tile.y1 + 5), fill=MUTED)
        print_text(draw, (x, tile.y1 + 9), col, 11, MUTED, anchor="mt")


def evidence_plate(image, box, orientation, title, *, aperture=None, perfs=None, metadata=()):
    paper = Image.new("RGB", (860, 710), PAGE)
    draw = ImageDraw.Draw(paper)
    print_text(draw, (27, 27), "REELTONE  /  SOURCE EVIDENCE", 18, TEAL)
    short_title = " ".join(wrapped_ascii(title, 65))[:65]
    print_text(draw, (27, 57), short_title, 21)
    area = Box(94, 118, 553, 650)
    window = place_film(paper, image, box, orientation, area)
    film_rulers(draw, window)
    rules = []
    if aperture:
        rules.append((aperture.box(box.y0, box.y1), TEAL, 2))
    if perfs:
        rules.append((perfs.box(box.y0, box.y1), ACCENT, 2))
    native_marks(draw, window, rules)
    draw.rectangle((570, 118, 830, 650), fill=PANEL)
    sidebar = [("NATIVE PIXELS", f"{box.width} × {box.height}"),
               ("X RANGE", f"{box.x0} .. {box.x1} exclusive"),
               ("Y RANGE", f"{box.y0} .. {box.y1} exclusive"),
               ("FILM TRAVEL", "Reverse" if orientation.reverse_y else "Forward"), *metadata]
    y = 138
    for label, value in sidebar:
        print_text(draw, (590, y), str(label), 12, MUTED)
        lines = wrapped_ascii(str(value), 28)
        y += 20
        for line in lines[:2]:
            print_text(draw, (590, y), line, 14)
            y += 19
        y += 18
        if y > 620:
            break
    return encode_image(paper, "PNG"), window.scale()


def exposure_heat(matrix, *, clipping=False):
    rows, cols = len(matrix), len(matrix[0])
    cell = 26
    canvas = Image.new("RGB", (cols * cell + 66, rows * cell + 70), PAGE)
    draw = ImageDraw.Draw(canvas)
    print_text(draw, (14, 11), "EXPOSURE  /  SOURCE GRID", 16, TEAL)
    for y, row in enumerate(matrix):
        for x, reading in enumerate(row):
            value = reading["mean"]
            if clipping:
                low, high = reading["low_fraction"], reading["high_fraction"]
                color = (round(30 + high * 210), round(38 + min(low, high) * 105), round(53 + low * 185))
            else:
                color = (round(13 + 192 * value), round(36 + 194 * value), round(42 + 176 * value))
            box = (51 + x * cell, 38 + y * cell, 50 + (x + 1) * cell - 1, 37 + (y + 1) * cell - 1)
            draw.rectangle(box, fill=color)
        if y % max(1, rows // 6) == 0:
            print_text(draw, (47, 41 + y * cell), row[0]["bounds"][1], 10, MUTED, anchor="rm")
    caption = "Blue: near black. Red: near white." if clipping else "Cell brightness: mean scan code."
    print_text(draw, (14, canvas.height - 20), caption, 11, MUTED)
    return encode_image(canvas, "PNG")


def blemish_heat(image, result: BlemishMap, orientation):
    box = result.region
    values = box.subimage(image)
    grayscale = np.rint(values * 168).astype(np.uint8)
    canvas = np.repeat(grayscale[:, :, None], 3, axis=2)
    canvas[result.bright] = [250, 178, 95]
    canvas[result.dark] = [81, 176, 234]
    transformed = orientation.image(canvas)
    return encode_image(fitted_preview(Image.fromarray(transformed), 900, 1000), "PNG")


def _diverging(delta, *, blank=None):
    half = np.clip(np.nan_to_num(delta, nan=0.0) * 2.75, -1, 1)
    red = 60 + np.maximum(half, 0) * 185
    cyan = 60 + np.maximum(-half, 0) * 178
    green = 58 + np.abs(half) * 93
    palette = np.rint(np.stack((red, green, cyan), axis=-1)).astype(np.uint8)
    if blank is not None:
        palette[blank] = (42, 50, 59)
    return Image.fromarray(palette)


def _native_rect(box, selection):
    box = box if isinstance(box, Box) else Box(*box)
    left = (box.x0 - selection.x0) / max(1, selection.width)
    right = (box.x1 - selection.x0) / max(1, selection.width)
    top = (box.y0 - selection.y0) / max(1, selection.height)
    bottom = (box.y1 - selection.y0) / max(1, selection.height)
    return min(left, right), min(top, bottom), max(left, right), max(top, bottom)


def _reference_marks(draw, panel_box, selection, references, colors):
    for (dark, light), color in zip(references, colors):
        for box in (dark, light):
            left, top, right, bottom = _native_rect(box, selection)
            left, right = min(1, max(0, left)), min(1, max(0, right))
            top, bottom = min(1, max(0, top)), min(1, max(0, bottom))
            if left == right or top == bottom:
                continue
            area = (panel_box[0] + left * (panel_box[2] - panel_box[0]),
                    panel_box[1] + top * (panel_box[3] - panel_box[1]),
                    panel_box[0] + right * (panel_box[2] - panel_box[0]),
                    panel_box[1] + bottom * (panel_box[3] - panel_box[1]))
            draw.rectangle(area, outline=color, width=2)


def _band_strip(draw, comparison, x, y, width):
    states = comparison.band_status
    count = len(states)
    for index, state in enumerate(states):
        left = x + round(index / count * width)
        right = x + round((index + 1) / count * width)
        draw.rectangle((left, y, max(left, right - 1), y + 13), fill={0: TEAL, 1: ACCENT, 2: RED}[int(state)])


def luminance_triptych(comparison: Comparison):
    height, width = comparison.delta.shape
    corrected = comparison.corrected_delta is not None
    width_cap = max(84, min(240 if corrected else 300, width))
    height_cap = max(120, min(420, height))
    tags = ("FIRST SCAN", "SECOND SCAN", "SECOND minus FIRST")
    columns = 4 if corrected else 3
    raster = Image.new("RGB", (columns * (width_cap + 16) + 20, height_cap + (150 if corrected else 92)), PAGE)
    draw = ImageDraw.Draw(raster)
    panels = [byte_preview(comparison.left), byte_preview(comparison.right), _diverging(comparison.delta)]
    if corrected:
        panels.append(_diverging(comparison.corrected_delta, blank=comparison.band_status == WITHHELD))
        tags += ("CORRECTED · EDGE REFERENCES",)
    for i, (tag, panel) in enumerate(zip(tags, panels)):
        x = 16 + i * (width_cap + 16)
        print_text(draw, (x, 21), tag, 12 if corrected else 13, TEAL if i < 2 else ACCENT)
        raster.paste(panel.resize((width_cap, height_cap), Image.Resampling.BILINEAR).convert("RGB"), (x, 49))
    if corrected:
        report = comparison.illumination
        references = report["references"]
        _reference_marks(draw, (16, 49, 16 + width_cap, 49 + height_cap), comparison.selection.left,
                         ((references["first"]["dark"], references["first"]["light"]),), (BLUE,))
        _reference_marks(draw, (32 + width_cap, 49, 32 + 2 * width_cap, 49 + height_cap), comparison.selection.right,
                         ((references["second"]["dark"], references["second"]["light"]),), (BLUE,))
        strip_x = 16 + 3 * (width_cap + 16)
        _band_strip(draw, comparison, strip_x, 49 + height_cap + 18, width_cap)
        print_text(draw, (strip_x, 49 + height_cap + 36), "measured", 10, TEAL)
        print_text(draw, (strip_x + 60, 49 + height_cap + 36), "estimated", 10, ACCENT)
        print_text(draw, (strip_x + 126, 49 + height_cap + 36), "withheld", 10, RED)
        print_text(draw, (16, raster.height - 26),
                   "Reference boxes outlined on each source. Grey rows lack usable references; their corrected difference is left blank.",
                   11, MUTED)
        print_text(draw, (16, raster.height - 11),
                   "Left panels and raw difference retain original scan codes. Corrected panel removes edge-referenced brightness and contrast.",
                   11, MUTED)
    else:
        print_text(draw, (16, raster.height - 25), "Relative coordinates within each selected rectangle.", 12, MUTED)
    return encode_image(raster, "PNG")


def reference_plate(report, first_image, second_image):
    references = report["references"]
    margin, cell, gap, strip = 20, 190, 16, 60
    width = 2 * (cell + gap) + strip + margin * 2
    canvas = Image.new("RGB", (max(width, 760), 2 * 214 + 104), PAGE)
    draw = ImageDraw.Draw(canvas)
    print_text(draw, (margin, 15), "EDGE REFERENCE PATCHES · NATIVE CROPS", 16, TEAL)
    for row, (key, image, title) in enumerate((("first", first_image, "FIRST SCAN"),
                                               ("second", second_image, "SECOND SCAN"))):
        reference = references[key]
        top = 52 + row * 214
        print_text(draw, (margin, top), title, 13, TEAL)
        for column, (side, tone) in enumerate((("dark", "DARK"), ("light", "LIGHT"))):
            box = Box(*reference[side])
            patch = byte_preview(box.subimage(image))
            fitted = fitted_preview(patch, cell, 132, filter=Image.Resampling.BOX)
            fitted = fitted.convert("RGB").resize((cell, 132), Image.Resampling.BILINEAR)
            x = margin + column * (cell + gap)
            canvas.paste(fitted, (x, top + 20))
            draw.rectangle((x, top + 20, x + cell - 1, top + 151), outline=GRID)
            print_text(draw, (x, top + 156), f"{tone} · {box.as_list()}", 10, MUTED)
            quality = reference["quality"][side]
            print_text(draw, (x, top + 170),
                       f"median {fmt_value(reference[f'{side}_median'])} · usable {quality['usable']}/{report['coverage']['bands']}",
                       10, MUTED)
            print_text(draw, (x, top + 184),
                       f"scratch {quality['scratch_or_obstruction']} · saturated {quality['saturated']} · "
                       f"uneven {quality['uneven']}", 10, MUTED)
        strip_x = margin + 2 * (cell + gap)
        strip_width = strip - 16
        for index, state in enumerate(np.asarray(report["band_states"])):
            y = top + 20 + round(index / max(1, len(report["band_states"])) * 132)
            y2 = top + 20 + round((index + 1) / max(1, len(report["band_states"])) * 132)
            draw.rectangle((strip_x, y, strip_x + strip_width, max(y, y2 - 1)),
                           fill={0: TEAL, 1: ACCENT, 2: RED}[int(state)])
        print_text(draw, (strip_x, top + 156), "per-band quality", 10, MUTED)
    legend_y = canvas.height - 46
    for offset, (color, label) in enumerate(((TEAL, "measured"), (ACCENT, "estimated"), (RED, "withheld"))):
        x = margin + offset * 110
        draw.rectangle((x, legend_y + 2, x + 12, legend_y + 12), fill=color)
        print_text(draw, (x + 18, legend_y), label, 10, MUTED)
    print_text(draw, (margin, canvas.height - 18),
               "Native crops from each own source at its retained recipe; bands without usable references are withheld.",
               10, MUTED)
    return encode_image(canvas, "PNG")


def fmt_value(value, digits=5):
    return "—" if value is None else f"{value:.{digits}f}"


def spectrum_png(result: Spectrum, time_origin=0.0):
    width, height = 960, 338
    left, top, right, bottom = 63, 42, width - 22, height - 42
    page = Image.new("RGB", (width, height), PAGE)
    draw = ImageDraw.Draw(page)
    view = np.flipud(result.power)
    db = 10.0 * np.log10(np.maximum(view, 1e-12))
    upper = max(-65, np.percentile(db, 98))
    relative = np.clip((db - (upper - 62)) / 62, 0.0, 1.0)
    palette = np.rint(np.stack((31 + relative * 200, 39 + relative * 149, 70 + relative * 27), axis=-1)).astype(np.uint8)
    page.paste(Image.fromarray(palette).resize((right - left, bottom - top), Image.Resampling.BILINEAR), (left, top))
    print_text(draw, (17, 12), "LISTENING SPECTROGRAM", 16, TEAL)
    print_text(draw, (width - 25, 14), f"{result.window} samples · {result.bin_size:.1f} Hz/bin", 12, MUTED, anchor="ra")
    high = float(result.hertz[-1])
    for hz in np.linspace(0, high, 6):
        y = bottom if high == 0 else round(bottom - hz / high * (bottom - top))
        draw.line((left - 4, y, left, y), fill=GRID)
        print_text(draw, (left - 7, y), f"{hz:.0f}", 11, MUTED, anchor="rm")
    a, b = float(result.seconds[0]), float(result.seconds[-1])
    for part in np.linspace(0, 1, 7):
        x = round(left + part * (right - left))
        draw.line((x, bottom, x, bottom + 5), fill=GRID)
        print_text(draw, (x, bottom + 8), f"{time_origin + a + part * (b - a):.3f}s", 10, MUTED, anchor="mt")
    return encode_image(page, "PNG")


@dataclass(frozen=True)
class PlotSeries:
    color: str
    records: list
    get_y: Callable
    get_x: Callable
    label: str


def line_plot(title, series, *, x_range, y_range, formatter=str, y_formatter=str, size=(940, 330)):
    canvas = Image.new("RGB", size, PAGE)
    draw = ImageDraw.Draw(canvas)
    print_text(draw, (19, 13), title, 16, TEAL)
    left, top, right, bottom = 69, 52, size[0] - 28, size[1] - 48
    x0, x1 = x_range
    y0, y1 = y_range
    if x1 <= x0:
        x1 = x0 + 1
    if y1 <= y0:
        y1 = y0 + 1
    def px(value):
        return round(interpolate(value, x0, x1, left, right))
    def py(value):
        return round(interpolate(value, y0, y1, bottom, top))
    for x in np.linspace(x0, x1, 7):
        pos = px(x)
        draw.line((pos, top, pos, bottom), fill=GRID)
        print_text(draw, (pos, bottom + 9), formatter(x), 10, MUTED, anchor="mt")
    for y in np.linspace(y0, y1, 5):
        pos = py(y)
        draw.line((left, pos, right, pos), fill=GRID)
        print_text(draw, (left - 9, pos), y_formatter(y), 10, MUTED, anchor="rm")
    legend_left = left
    for line in series:
        previous = None
        for record in line.records:
            y = line.get_y(record)
            if y is None or not np.isfinite(y):
                previous = None
                continue
            x = line.get_x(record)
            point = (px(np.clip(x, x0, x1)), py(np.clip(y, y0, y1)))
            if previous is not None:
                draw.line((*previous, *point), fill=line.color, width=2)
            else:
                draw.point(point, fill=line.color)
            previous = point
        draw.line((legend_left, size[1] - 15, legend_left + 12, size[1] - 15), fill=line.color, width=2)
        print_text(draw, (legend_left + 17, size[1] - 21), line.label, 11, MUTED)
        legend_left += 28 + len(line.label) * 7
    return encode_image(canvas, "PNG")
