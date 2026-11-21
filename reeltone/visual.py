from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .blemishes import BlemishMap
from .comparison import Comparison
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


def luminance_triptych(comparison: Comparison):
    height, width = comparison.delta.shape
    width_cap = max(104, min(300, width))
    height_cap = max(120, min(420, height))
    tags = ("FIRST SCAN", "SECOND SCAN", "SECOND minus FIRST")
    raster = Image.new("RGB", (3 * (width_cap + 16) + 20, height_cap + 92), PAGE)
    draw = ImageDraw.Draw(raster)
    half = np.clip(comparison.delta * 2.75, -1, 1)
    red = 60 + np.maximum(half, 0) * 185
    cyan = 60 + np.maximum(-half, 0) * 178
    green = 58 + np.abs(half) * 93
    palette = np.rint(np.stack((red, green, cyan), axis=-1)).astype(np.uint8)
    panels = (byte_preview(comparison.left), byte_preview(comparison.right), Image.fromarray(palette))
    for i, (tag, panel) in enumerate(zip(tags, panels)):
        x = 16 + i * (width_cap + 16)
        print_text(draw, (x, 21), tag, 13, TEAL if i < 2 else ACCENT)
        raster.paste(panel.resize((width_cap, height_cap), Image.Resampling.BILINEAR).convert("RGB"), (x, 49))
    print_text(draw, (16, raster.height - 25), "Relative coordinates within each selected rectangle.", 12, MUTED)
    return encode_image(raster, "PNG")


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
