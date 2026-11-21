"""Physical-scale proof sheets.

The ordinary contact sheet fits every crop into its own frame, so two scans
made at different resolutions only *look* alike.  A checker comparing edge
bands with a ruler needs paper millimetres to mean film millimetres: every
crop on every page is placed at one shared paper-to-film magnification.

The operator supplies pixels per millimetre for each scan; embedded file DPI
is treated as a declaration and is never used for measurement.  Page rulers
are drawn in paper millimetres from the page edges, and the preview PNG and
the PDF are the same rasters, so a ruler on the print agrees with the
placement recorded in the manifest.
"""

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import math

from PIL import Image, ImageDraw

from .bench import load_material
from .contracts import identifier, mapping, option, real, sequence, text
from .errors import BenchError, Conflict, Limit
from .geometry import Box
from .raster import byte_preview
from .visual import available_font, print_text, wrapped_ascii


PAPER_MM = (210.0, 297.0)          # A4 only; page rulers are printed for this stock.
PRINT_DPI = 240
PREVIEW_LIMIT = (1100, 1500)
MARGIN_LEFT = 14.0
MARGIN_RIGHT = 14.0
MARGIN_TOP = 31.0
MARGIN_BOTTOM = 34.0
SIDE_GAP = 9.0
STACK_GAP = 4.0
CAPTION_MM = 19.5                   # title, source range, measurements, bar; sit clear of the rulers
MIN_SCALE = 0.01
MAX_SCALE = 20.0
SCALE_STEP = 0.001
GROUP_PIXEL_LIMIT = 4_000_000
TOTAL_PIXEL_LIMIT = 8_000_000
MAX_GROUPS = 2

WHITE = 250
INK = 26
MID = 82
SOFT = 132
PALE = 200


@dataclass(frozen=True)
class Panel:
    label: str
    material: object
    box: Box
    pixels_per_mm: float
    tonal: str

    @property
    def film_mm(self):
        """Physical size of the crop, in millimetres, as scanned."""
        return self.box.width / self.pixels_per_mm, self.box.height / self.pixels_per_mm

    def record(self):
        return {"label": self.label, "scan_id": self.material.record["id"],
                "scan_title": self.material.record["title"],
                "source_sha256": self.material.record["sha256"],
                "recipe_id": self.material.recipe_record["id"],
                "recipe_number": self.material.recipe_record["seq"],
                "native_box": self.box.as_list(), "pixels_per_mm": self.pixels_per_mm,
                "film_mm": [round(value, 4) for value in self.film_mm], "tonal": self.tonal,
                "orientation": self.material.recipe.orientation.as_dict()}


@dataclass(frozen=True)
class Group:
    label: str
    layout: str
    arrange: str
    panels: tuple

    def record(self):
        return {"label": self.label, "paper": "A4", "orientation": self.layout, "arrange": self.arrange,
                "panels": [panel.record() for panel in self.panels]}


@dataclass(frozen=True)
class Page:
    width_mm: float
    height_mm: float
    width_px: int
    height_px: int
    usable_width_mm: float
    usable_height_mm: float
    scale_x: float
    scale_y: float


def page_for(layout):
    width_mm, height_mm = PAPER_MM if layout == "portrait" else PAPER_MM[::-1]
    width_px, height_px = round(width_mm / 25.4 * PRINT_DPI), round(height_mm / 25.4 * PRINT_DPI)
    return Page(width_mm, height_mm, width_px, height_px,
                width_mm - MARGIN_LEFT - MARGIN_RIGHT, height_mm - MARGIN_TOP - MARGIN_BOTTOM,
                width_px / width_mm, height_px / height_mm)


def _plain(value, width=90):
    return " ".join(wrapped_ascii(value, width))[:width]


def _label(value, fallback, width=90):
    """ASCII-safe print label; Chinese labels stay whole in the JSON manifest."""
    rendered = _plain(value, width)
    if value and sum(character.isascii() and character.isalnum() for character in rendered) < 2:
        return _plain(fallback, width)
    return rendered


def _fit_text(content, size, maximum_px):
    """Trim a one-line label so it cannot run under a page ruler."""
    content = str(content)
    if maximum_px <= 0:
        return ""
    font = available_font(size)
    while content and font.getlength(content) > maximum_px:
        content = content[:-1]
    return content.rstrip(" ,;:-")


def _trim(value, digits=3):
    rendered = f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return rendered or "0"


def ratio_text(scale):
    if abs(scale - 1.0) < 1e-12:
        return "1:1"
    if scale > 1:
        return f"{_trim(scale, 4)}:1"
    return f"1:{_trim(1.0 / scale, 4)}"


CAPTION_COLUMN_MM = 44.0     # legible caption under one crop; sets the minimum column


def group_extent(group, scale):
    """Paper footprint of one group at *scale*.

    Side-by-side crops each get a caption column at least
    ``CAPTION_COLUMN_MM`` wide, so a small crop cannot push its caption under
    its neighbour.  Stacked crops stack crops and captions in one column.
    Fit checks and drawing both read this geometry.
    """
    widths = [panel.film_mm[0] * scale for panel in group.panels]
    heights = [panel.film_mm[1] * scale for panel in group.panels]
    if group.arrange == "side":
        columns = [max(width, CAPTION_COLUMN_MM) for width in widths]
        return {"width": sum(columns) + SIDE_GAP, "height": max(heights) + CAPTION_MM,
                "widths": widths, "heights": heights, "columns": columns}
    columns = [max(max(widths), CAPTION_COLUMN_MM)] * len(widths)
    return {"width": columns[0], "height": sum(heights) + CAPTION_MM * len(widths)
            + STACK_GAP * max(0, len(widths) - 1),
            "widths": widths, "heights": heights, "columns": columns}


def group_fit(group, scale):
    page = page_for(group.layout)
    extent = group_extent(group, scale)
    overflow_x = max(0.0, extent["width"] - page.usable_width_mm)
    overflow_y = max(0.0, extent["height"] - page.usable_height_mm)
    return {"label": group.label, "orientation": group.layout, "arrange": group.arrange,
            "fits": overflow_x <= 1e-6 and overflow_y <= 1e-6,
            "required_mm": [round(extent["width"], 4), round(extent["height"], 4)],
            "available_mm": [round(page.usable_width_mm, 4), round(page.usable_height_mm, 4)],
            "over_mm": {"width": round(overflow_x, 4), "height": round(overflow_y, 4)},
            "panels": [{"label": panel.label, "film_mm": [round(value, 4) for value in panel.film_mm],
                        "paper_mm": [round(width, 4), round(height, 4)]}
                       for panel, width, height in zip(group.panels, extent["widths"], extent["heights"])]}


def _shared_max_scale(groups):
    """Largest shared magnification that still fits every group, to 0.001."""
    if all(group_fit(group, MAX_SCALE)["fits"] for group in groups):
        return MAX_SCALE
    low, high = 0.0, MAX_SCALE
    for _ in range(48):
        middle = (low + high) / 2
        if all(group_fit(group, middle)["fits"] for group in groups):
            low = middle
        else:
            high = middle
    snapped = math.floor(low / SCALE_STEP) * SCALE_STEP
    while snapped > 0 and not all(group_fit(group, snapped)["fits"] for group in groups):
        snapped = round(snapped - SCALE_STEP, 6)
    return round(snapped, 6) if snapped >= MIN_SCALE else 0.0


def _material(cache, catalog, reel_id, scan_id, recipe_id):
    key = (scan_id, recipe_id)
    if key not in cache:
        source = catalog.get_scan(scan_id)
        if source["reel_id"] != reel_id:
            raise BenchError("所选扫描不属于这个档案卷。")
        cache[key] = load_material(catalog, scan_id, recipe_id)
    return cache[key]


def parse_request(catalog, reel_id, values):
    values = mapping(values, "物理倍率联排请求")
    catalog.get_reel(reel_id)
    title = text(values.get("title", "Physical scale proof"), "联排标题", maximum=85)
    note = text(values.get("note", ""), "联排说明", maximum=240, required=False)
    scale = real(values.get("scale", 1.0), "出图倍率", MIN_SCALE, MAX_SCALE)
    rows = sequence(values.get("groups"), "物理联排分组", minimum=1, maximum=MAX_GROUPS)
    cache, groups, total = {}, [], 0
    resolution_by_scan = {}
    for number, raw in enumerate(rows, 1):
        raw = mapping(raw, "物理联排分组")
        label = text(raw.get("label", f"Group {number}"), "分组名称", maximum=65)
        layout = option(raw.get("layout", "portrait"), ("portrait", "landscape"), "纸张方向")
        arrange = option(raw.get("arrange", "side"), ("side", "stack"), "裁片摆放")
        panels, group_pixels = [], 0
        for index, item in enumerate(sequence(raw.get("panels"), "分组裁片", minimum=2, maximum=2), 1):
            item = mapping(item, "裁片")
            scan_id = identifier(item.get("scan"), "裁片扫描")
            recipe_id = identifier(item["recipe"], "裁片配方") if item.get("recipe") else None
            material = _material(cache, catalog, reel_id, scan_id, recipe_id)
            box = Box.parse(item.get("box"), material.width, material.height, minimum=(4, 8), name="裁片原片框")
            if item.get("pixels_per_mm") in (None, ""):
                raise BenchError("请为每张扫描填写每毫米像素数；这里不使用图像内嵌 DPI。")
            pixels_per_mm = real(item.get("pixels_per_mm"), "裁片每毫米扫描像素数", 0.1, 25_000)
            # One scan has one resolution, however many crops it appears in.
            known = resolution_by_scan.setdefault(scan_id, pixels_per_mm)
            if abs(known - pixels_per_mm) > 1e-9:
                raise Conflict("同一张扫描填写了两个不同的每毫米像素数；请统一后再排。")
            tonal = option(item.get("tonal", "native"), ("native", "stretch"), "裁片色调")
            label_text = text(item.get("label", f"Panel {index}"), "裁片标签", maximum=65)
            group_pixels += box.area
            panels.append(Panel(label_text, material, box, pixels_per_mm, tonal))
        if group_pixels > GROUP_PIXEL_LIMIT:
            raise Limit("每组裁片最多读取 400 万原生像素。")
        total += group_pixels
        groups.append(Group(label, layout, arrange, tuple(panels)))
    if total > TOTAL_PIXEL_LIMIT:
        raise Limit("一次物理倍率联排最多读取 800 万原生像素。")
    return title, note, scale, groups


def plan_digest(title, note, scale, groups):
    document = {"title": title, "note": note, "scale": scale, "paper": "A4", "dpi": PRINT_DPI,
                "groups": [group.record() for group in groups]}
    encoded = json.dumps(document, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")
    return sha256(encoded).hexdigest()


def _conflict_manifest(title, note, scale, groups, maximum, digest):
    fits = [group_fit(group, scale) for group in groups]
    broken = [item for item in fits if not item["fits"]]
    manifest = {"status": "conflict" if maximum > 0 else "impossible", "title": title, "note": note,
                "paper": "A4", "requested_scale": scale, "requested_ratio": ratio_text(scale),
                "shared_scale": True, "max_scale": maximum, "max_ratio": ratio_text(maximum) if maximum else None,
                "print_dpi": PRINT_DPI, "plan_sha256": digest, "groups": fits,
                "conflicts": broken}
    if maximum > 0 and scale > maximum + 1e-9:
        manifest["message"] = (f"按 {ratio_text(scale)} 出图时 {len(broken)} 个分组超出 A4 可用区域；"
                               f"同一排版最多可到 {ratio_text(maximum)}，即两页共用的最大等比倍率。"
                               f"确认等比缩小后可预览并下载。")
        manifest["suggested_scale"] = maximum
        manifest["suggested_ratio"] = ratio_text(maximum)
        manifest["remedies"] = ["采用建议倍率重新预览", "改横竖纸或改成上下叠放", "缩小原片框"]
    else:
        manifest["status"] = "impossible"
        manifest["message"] = "即使缩到最小支持倍率，仍有裁片无法完整放入 A4；请分开出图或缩小原片框。"
        manifest["suggested_scale"] = None
        manifest["suggested_ratio"] = None
    for group in broken:
        group["limiting"] = [name for name in ("width", "height") if group["over_mm"][name] > 1e-6]
    return manifest


def _placements(group, page, scale):
    """Top-left paper position for each crop plus its caption column."""
    extent = group_extent(group, scale)
    left = MARGIN_LEFT + max(0.0, page.usable_width_mm - extent["width"]) / 2
    top = MARGIN_TOP + max(0.0, page.usable_height_mm - extent["height"]) / 2
    result = []
    if group.arrange == "side":
        cursor = left
        bottom = top + max(extent["heights"])
        for panel, width, height, column in zip(group.panels, extent["widths"], extent["heights"], extent["columns"]):
            result.append((panel, cursor + (column - width) / 2, top, width, height, bottom + 2.2))
            cursor += column + SIDE_GAP
    else:
        cursor = top
        column = extent["columns"][0]
        for panel, width, height in zip(group.panels, extent["widths"], extent["heights"]):
            result.append((panel, left + (column - width) / 2, cursor, width, height, cursor + height + 2.2))
            cursor += height + CAPTION_MM + STACK_GAP
    return result


def _paper_box(x_mm, y_mm, width_mm, height_mm, page):
    x0, y0 = round(x_mm * page.scale_x), round(y_mm * page.scale_y)
    x1 = max(x0 + 1, round((x_mm + width_mm) * page.scale_x))
    y1 = max(y0 + 1, round((y_mm + height_mm) * page.scale_y))
    return Box(x0, y0, x1, y1)


def _millimetre_rulers(draw, page):
    width, height = page.width_mm, page.height_mm
    axis = height - 9.0
    draw.line((0, round(axis * page.scale_y), round(width * page.scale_x), round(axis * page.scale_y)),
              fill=INK, width=1)
    for tick in range(int(width) + 1):
        length = 3.0 if tick % 10 == 0 else 2.0 if tick % 5 == 0 else 1.2
        x = round(tick * page.scale_x)
        draw.line((x, round(axis * page.scale_y), x, round((axis - length) * page.scale_y)), fill=INK, width=1)
        if tick % 10 == 0:
            print_text(draw, (x, round((axis - 4.0) * page.scale_y)), str(tick), 17, MID, anchor="mb")
    print_text(draw, (round(2.0 * page.scale_x), round((axis + 1.2) * page.scale_y)),
               "PAPER mm", 15, MID, anchor="lt")
    edge = width - 5.2
    draw.line((round(edge * page.scale_x), 0, round(edge * page.scale_x), round(height * page.scale_y)),
              fill=INK, width=1)
    for tick in range(int(height) + 1):
        length = 2.4 if tick % 10 == 0 else 1.2
        y = round(tick * page.scale_y)
        draw.line((round(edge * page.scale_x), y, round((edge - length) * page.scale_x), y), fill=INK, width=1)
        if tick % 20 == 0 and 24 <= tick < height - 26:
            print_text(draw, (round((edge - 3.2) * page.scale_x), y), str(tick), 17, MID, anchor="rm")


def _header(draw, page, title, note, scale, group, number, total):
    left = round(MARGIN_LEFT * page.scale_x)
    right = round((page.width_mm - MARGIN_RIGHT) * page.scale_x)
    print_text(draw, (left, round(8.5 * page.scale_y)), "REELTONE  /  PHYSICAL SCALE PROOF", 24, MID)
    print_text(draw, (left, round(14.2 * page.scale_y)),
               _fit_text(_label(title, "Physical scale proof", 60), 40,
                         round(page.scale_x * (page.width_mm - MARGIN_RIGHT - 7.0)) - left), 40, INK)
    name = _label(group.label, "", 28)
    if name.strip().lower() in {"", f"group {number}", f"group {number:02d}"}:
        name = ""
    run = (f"GROUP {number}" + (f"   {name}" if name else "") +
           f"   PAPER:FILM {ratio_text(scale)}   A4 {page.width_mm:.0f} x {page.height_mm:.0f} mm"
           f"   {PRINT_DPI} dpi   {'side by side' if group.arrange == 'side' else 'stacked'}")
    print_text(draw, (left, round(23.6 * page.scale_y)),
               _fit_text(run, 20, round(page.scale_x * (page.width_mm - MARGIN_RIGHT - 7.0)) - left), 20, MID)
    draw.line((left, round(27.6 * page.scale_y), right, round(27.6 * page.scale_y)), fill=PALE, width=1)
    available = round(page.scale_x * (page.width_mm - MARGIN_RIGHT - 7.0)) - left
    footer = (_label(note, "", 160) or
              "All crops share one paper-to-film magnification. Millimetre rulers run from the page edges.")
    print_text(draw, (left, round((page.height_mm - 22.6) * page.scale_y)),
               _fit_text(footer, 18, available), 18, MID)
    print_text(draw, (left, round((page.height_mm - 18.2) * page.scale_y)),
               _fit_text(f"ratio {ratio_text(scale)}   ticks 1/5/10 mm   page {number} of {total}"
                         f"   print at 100% actual size", 17, available), 17, SOFT)


def _tick_fractions(span_px):
    """Label 5, 3 or 2 reference points so numbers never run together."""
    if span_px >= 5 * 24:
        return (0.0, 0.25, 0.5, 0.75, 1.0)
    if span_px >= 3 * 24:
        return (0.0, 0.5, 1.0)
    return (0.0, 1.0)


def _native_ticks(draw, canvas, panel, page):
    """Native column/row numbers at the crop edges, kept outside the image."""
    box = panel.box
    orientation = panel.material.recipe.orientation
    for fraction in _tick_fractions(canvas.width):
        x = round(canvas.x0 + fraction * (canvas.width - 1))
        native_x = box.x1 - fraction * box.width if orientation.mirror_x else box.x0 + fraction * box.width
        draw.line((x, canvas.y0 - round(1.4 * page.scale_y), x, canvas.y0), fill=INK, width=1)
        print_text(draw, (x, canvas.y0 - round(2.0 * page.scale_y)), str(int(round(native_x))), 14, MID, anchor="mb")
    for fraction in _tick_fractions(canvas.height):
        y = round(canvas.y0 + fraction * (canvas.height - 1))
        native_y = box.y1 - fraction * box.height if orientation.reverse_y else box.y0 + fraction * box.height
        draw.line((canvas.x0 - round(1.4 * page.scale_x), y, canvas.x0, y), fill=INK, width=1)
        print_text(draw, (canvas.x0 - round(1.9 * page.scale_x), y), str(int(round(native_y))), 14, MID, anchor="rm")


def _draw_panel(plate, draw, panel, x_mm, y_mm, width_mm, height_mm, caption_y, page):
    tile = panel.material.recipe.orientation.image(panel.box.subimage(panel.material.pixels))
    tone = panel.material.recipe.tone
    low, high = (tone.black, tone.white) if panel.tonal == "stretch" else (0.0, 1.0)
    source = byte_preview(tile, lo=low, hi=high)
    canvas = _paper_box(x_mm, y_mm, width_mm, height_mm, page)
    # Resize to the same box that is recorded and outlined, so the printed
    # crop, the manifest and the frame cannot disagree by a rounding pixel.
    target = (canvas.width, canvas.height)
    kernel = Image.Resampling.LANCZOS if target[0] < source.width else Image.Resampling.BICUBIC
    plate.paste(source.resize(target, kernel), (canvas.x0, canvas.y0))
    draw.rectangle((canvas.x0, canvas.y0, canvas.x1 - 1, canvas.y1 - 1), outline=INK, width=1)
    _native_ticks(draw, canvas, panel, page)

    film_width, film_height = panel.film_mm
    left = round(x_mm * page.scale_x)
    print_text(draw, (left, round(caption_y * page.scale_y)), _label(panel.label, "Crop", 34), 22, INK)
    source_line = (f"{_label(panel.material.record['title'], 'Scan ' + panel.material.record['sha256'][:6], 26)}"
                   f"   x {panel.box.x0}:{panel.box.x1}   row {panel.box.y0}:{panel.box.y1}")
    print_text(draw, (left, round((caption_y + 3.8) * page.scale_y)), source_line[:84], 16, MID)
    measurements = (f"{_trim(panel.pixels_per_mm)} px/mm   film {_trim(film_width)} x {_trim(film_height)} mm")
    print_text(draw, (left, round((caption_y + 6.6) * page.scale_y)), measurements, 16, MID)
    drawn = (f"paper {_trim(width_mm)} x {_trim(height_mm)} mm"
             f"   paper:film {ratio_text(width_mm / film_width if film_width else 0)}")
    print_text(draw, (left, round((caption_y + 9.4) * page.scale_y)), drawn, 16, MID)
    bar_y = round((caption_y + 13.4) * page.scale_y)
    bar_end = left + round(10.0 * page.scale_x)
    draw.line((left, bar_y, bar_end, bar_y), fill=INK, width=2)
    for x in (left, bar_end):
        draw.line((x, bar_y - round(1.2 * page.scale_y), x, bar_y + round(1.2 * page.scale_y)), fill=INK, width=1)
    print_text(draw, (bar_end + round(1.8 * page.scale_x), bar_y), "10 mm paper", 15, MID, anchor="lm")
    orientation = panel.material.recipe.orientation
    print_text(draw, (left, round((caption_y + 16.8) * page.scale_y)),
               f"{'reversed' if orientation.reverse_y else 'forward'} rows"
               f"   {'mirrored' if orientation.mirror_x else 'as scanned'}"
               f"   {panel.material.record['sha256'][:10]}", 15, SOFT)
    return canvas


def _render_page(group, scale, title, note, number, total):
    page = page_for(group.layout)
    plate = Image.new("L", (page.width_px, page.height_px), WHITE)
    draw = ImageDraw.Draw(plate)
    _header(draw, page, title, note, scale, group, number, total)
    _millimetre_rulers(draw, page)
    placements = []
    for panel, x_mm, y_mm, width_mm, height_mm, caption_y in _placements(group, page, scale):
        canvas = _draw_panel(plate, draw, panel, x_mm, y_mm, width_mm, height_mm, caption_y, page)
        record = panel.record()
        record.update({"page_mm": [round(x_mm, 4), round(y_mm, 4)],
                       "paper_mm": [round(width_mm, 4), round(height_mm, 4)],
                       "page_pixel_box": canvas.as_list(),
                       "measured_paper_mm": [round(canvas.width / page.scale_x, 4),
                                             round(canvas.height / page.scale_y, 4)],
                       "measured_ratio": [round(canvas.width / page.scale_x / panel.film_mm[0], 6),
                                          round(canvas.height / page.scale_y / panel.film_mm[1], 6)]})
        placements.append(record)
    return plate, page, placements


def _preview(plate):
    thumb = plate.copy()
    thumb.thumbnail(PREVIEW_LIMIT, Image.Resampling.LANCZOS)
    buffer = BytesIO()
    thumb.save(buffer, "PNG", compress_level=4)
    return buffer.getvalue(), [thumb.width, thumb.height]


def jpeg_size(stream):
    with Image.open(BytesIO(stream)) as image:
        return image.width, image.height


def _pdf_bytes(pages, title, note):
    """Write a minimal multi-page PDF; each page is one DeviceGray DCT image.

    The page raster is stretched to the exact A4 MediaBox, so a page pixel is
    a fixed fraction of the paper and the millimetre ruler cannot drift
    between the preview and the print.
    """
    encoded = []
    for plate, width_mm, height_mm in pages:
        stream = BytesIO()
        plate.convert("L").save(stream, "JPEG", quality=90, optimize=True)
        encoded.append((stream.getvalue(), width_mm, height_mm))
    objects = {1: b"<< /Type /Catalog /Pages 2 0 R >>", 3: b""}
    kids = []
    for index, (stream, width_mm, height_mm) in enumerate(encoded):
        content_id, image_id, page_id = 4 + index * 3, 5 + index * 3, 6 + index * 3
        kids.append(page_id)
        width_pt = round(width_mm * 72.0 / 25.4, 4)
        height_pt = round(height_mm * 72.0 / 25.4, 4)
        content = f"q {width_pt} 0 0 {height_pt} 0 0 cm /Im0 Do Q\n".encode("ascii")
        objects[content_id] = (f"<< /Length {len(content)} >>\nstream\n".encode("ascii") + content + b"endstream")
        image_width, image_height = jpeg_size(stream)
        objects[image_id] = (f"<< /Type /XObject /Subtype /Image /Width {image_width} /Height {image_height}"
                             f" /ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /DCTDecode"
                             f" /Length {len(stream)} >>\nstream\n").encode("ascii") + stream + b"\nendstream"
        # The image CTM below is the page box itself, so paper millimetres in the
        # raster and on a ruler read off the print are the same quantity.
        objects[page_id] = (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width_pt} {height_pt}]"
                            f" /Resources << /ProcSet [ /PDF /ImageB ] /XObject << /Im0 {image_id} 0 R >> >>"
                            f" /Contents {content_id} 0 R >>").encode("ascii")
    objects[2] = f"<< /Type /Pages /Count {len(encoded)} /Kids [ {' '.join(f'{item} 0 R' for item in kids)} ] >>".encode("ascii")
    objects[3] = (f"<< /Title ({_pdf_text(title)}) /Subject ({_pdf_text(note)}) /Producer (ReelTone) >>").encode("ascii")
    output = BytesIO()
    output.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = {}
    for number in sorted(objects):
        offsets[number] = output.tell()
        output.write(f"{number} 0 obj\n".encode("ascii") + objects[number] + b"\nendobj\n")
    start = output.tell()
    size = max(objects) + 1
    output.write(f"xref\n0 {size}\n0000000000 65535 f \n".encode("ascii"))
    for number in range(1, size):
        output.write(f"{offsets[number]:010d} 00000 n \n".encode("ascii"))
    output.write(f"trailer\n<< /Size {size} /Root 1 0 R /Info 3 0 R >>\nstartxref\n{start}\n%%EOF".encode("ascii"))
    return output.getvalue()


def _pdf_text(value):
    return _plain(value, 150).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _pages(groups, scale, title, note):
    rendered = [_render_page(group, scale, title, note, number, len(groups))
                for number, group in enumerate(groups, 1)]
    records, plates = [], []
    for plate, page, placements in rendered:
        preview, preview_size = _preview(plate)
        records.append({"orientation": "portrait" if page.height_px > page.width_px else "landscape",
                        "size_mm": [round(page.width_mm, 3), round(page.height_mm, 3)],
                        "size_px": [page.width_px, page.height_px], "preview_px": preview_size,
                        "preview_mm_per_px": round(page.width_mm / preview_size[0], 6),
                        "preview_sha256": sha256(preview).hexdigest(), "placements": placements})
        plates.append((plate, preview))
    return records, plates


def _measurement_manifest(title, note, scale, groups, maximum, digest, records):
    return {"status": "fits", "title": title, "note": note, "paper": "A4", "requested_scale": scale,
            "requested_ratio": ratio_text(scale), "shared_scale": True, "max_scale": maximum,
            "max_ratio": ratio_text(maximum), "print_dpi": PRINT_DPI, "plan_sha256": digest,
            "notes": ["倍率由操作者填写的每毫米像素数换算；不使用图像内嵌 DPI。",
                      "各分组共用同一纸面倍率；裁片完整显示，每页只放一个分组，最多两页。",
                      f"标尺零点在页面左下与右中纸边，刻度为 1 mm、5 mm、10 mm；预览 PNG 与 PDF 使用同一 {PRINT_DPI} dpi 栅格。",
                      "题注给出原片行号与列号、每毫米像素数、原片毫米与纸面毫米，可带回原片位置复核。"],
            "pages": records, "groups": [group_fit(group, scale) for group in groups]}


def plan_physical(catalog, reel_id, values):
    title, note, scale, groups = parse_request(catalog, reel_id, values)
    digest = plan_digest(title, note, scale, groups)
    maximum = _shared_max_scale(groups)
    if not all(group_fit(group, scale)["fits"] for group in groups):
        return _conflict_manifest(title, note, scale, groups, maximum, digest), []
    records, plates = _pages(groups, scale, title, note)
    for index, group in enumerate(groups):
        records[index]["label"] = group.label
        records[index]["group_index"] = index + 1
    return _measurement_manifest(title, note, scale, groups, maximum, digest, records), [preview for _, preview in plates]


def render_physical(catalog, reel_id, values):
    values = mapping(values, "物理倍率联排请求")
    title, note, scale, groups = parse_request(catalog, reel_id, values)
    digest = plan_digest(title, note, scale, groups)
    maximum = _shared_max_scale(groups)
    if not all(group_fit(group, scale)["fits"] for group in groups):
        # Report the layout conflict itself; the operator must pick a fitting
        # magnification and preview it before this endpoint will produce a PDF.
        return _conflict_manifest(title, note, scale, groups, maximum, digest), [], None
    if values.get("plan_sha256") != digest:
        raise Conflict("排版与已预览的方案不一致。请重新预览后再下载。")
    records, plates = _pages(groups, scale, title, note)
    for index, group in enumerate(groups):
        records[index]["label"] = group.label
        records[index]["group_index"] = index + 1
    pdf = _pdf_bytes([(plate, record["size_mm"][0], record["size_mm"][1])
                      for (plate, _), record in zip(plates, records)], title, note)
    manifest = _measurement_manifest(title, note, scale, groups, maximum, digest, records)
    manifest["pdf_sha256"] = sha256(pdf).hexdigest()
    return manifest, [preview for _, preview in plates], pdf
