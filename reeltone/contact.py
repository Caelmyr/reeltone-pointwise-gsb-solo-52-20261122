from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO

from PIL import Image, ImageDraw

from .contracts import mapping, option, sequence, text
from .errors import BenchError, Limit
from .geometry import Box
from .raster import encode_image
from .visual import ACCENT, GRID, MUTED, PAGE, PANEL, TEAL, TEXT, film_rulers, native_marks, place_film, print_text, wrapped_ascii


PAPER_SIZES = {"A4": (210, 297), "Letter": (216, 279)}
PRINT_DPI = 120


@dataclass(frozen=True)
class Panel:
    label: str
    area: Box
    mode: str
    with_guides: bool

    def contents(self, scale):
        return {"label": self.label, "native_box": self.area.as_list(), "tonal": self.mode,
                "mark_guides": self.with_guides, "layout": scale}


@dataclass(frozen=True)
class Contact:
    plate: bytes
    pdf: bytes
    manifest: dict


def arrange_panels(data, material):
    rows = sequence(data.get("panels"), "证据联排", minimum=1, maximum=6)
    result = []
    total = 0
    for n, raw in enumerate(rows, 1):
        raw = mapping(raw, "联排框")
        name = text(raw.get("title", f"Panel {n}"), "联排条目名称", maximum=65)
        box = Box.parse(raw.get("box"), material.width, material.height, minimum=(4, 8), name="联排原片框")
        total += box.area
        if total > 4_000_000:
            raise Limit("一次联排最多读取 400 万原生像素。")
        mode = option(raw.get("tonal", "native"), ("native", "stretch"), "联排色调")
        guides = raw.get("with_guides", True)
        if not isinstance(guides, bool):
            raise BenchError("联排读窗显示须为布尔值。")
        result.append(Panel(name, box, mode, guides))
    return result


def output_dimensions(paper, turn):
    width_mm, height_mm = PAPER_SIZES[paper]
    if turn == "landscape":
        width_mm, height_mm = height_mm, width_mm
    pixels = tuple(round(value / 25.4 * PRINT_DPI) for value in (width_mm, height_mm))
    return pixels, (width_mm, height_mm)


def page_design(size, count, turn):
    width, height = size
    columns = min(count, 3 if turn == "landscape" else 2)
    rows = (count + columns - 1) // columns
    left, top, right, bottom = (44, 126, width - 44, height - 62)
    col_gap, row_gap = 18, 21
    panel_width = (right - left - (columns - 1) * col_gap) // columns
    panel_height = (bottom - top - (rows - 1) * row_gap) // rows
    result = []
    for i in range(count):
        x = left + (i % columns) * (panel_width + col_gap)
        y = top + (i // columns) * (panel_height + row_gap)
        result.append(Box(x, y, x + panel_width, y + panel_height))
    return result


def panel_annotation(draw, rect, panel, info, item, ink):
    headline = " ".join(wrapped_ascii(panel.label, 38))[:38]
    print_text(draw, (rect.x0 + 12, rect.y0 + 12), f"{item:02d}  {headline}", 14, ink["text"])
    left, top, right, bottom = panel.area.as_list()
    position = f"x {left}:{right}  |  row {top}:{bottom}"
    print_text(draw, (rect.x0 + 12, rect.y1 - 43), position, 11, ink["muted"])
    native_size = f"{panel.area.width} × {panel.area.height} native pixels"
    print_text(draw, (rect.x0 + 12, rect.y1 - 25), native_size, 11, ink["muted"])
    img = info.canvas_box
    ratio = min(img.width / panel.area.width, img.height / panel.area.height)
    return round(ratio, 6)


def contact_sheet(material, values, *, light=False):
    title = text(values.get("title", "ReelTone contact"), "联排标题", maximum=85)
    page_size = option(values.get("paper", "A4"), PAPER_SIZES, "纸张")
    layout = option(values.get("layout", "portrait"), ("portrait", "landscape"), "纸张方向")
    notes = text(values.get("note", ""), "联排说明", maximum=240, required=False)
    panels = arrange_panels(values, material)
    size_px, size_mm = output_dimensions(page_size, layout)
    rectangles = page_design(size_px, len(panels), layout)
    if light:
        ink = {"background": "#fffdfa", "panel": "#f0f3f1", "text": "#18222a", "muted": "#4a5b60", "accent": "#155f56", "border": "#ccd7d4"}
    else:
        ink = {"background": PAGE, "panel": PANEL, "text": TEXT, "muted": MUTED, "accent": TEAL, "border": GRID}
    image = Image.new("RGB", size_px, ink["background"])
    draw = ImageDraw.Draw(image)
    print_text(draw, (44, 27), "REELTONE  /  FILM CONTACT SHEET", 17, ink["accent"])
    print_text(draw, (44, 56), " ".join(wrapped_ascii(title, 68))[:68], 25, ink["text"])
    metadata = f"{material.raster.filename}   v{material.recipe_record['seq']}   {page_size}   {layout}"
    print_text(draw, (44, 95), " ".join(wrapped_ascii(metadata, 95))[:95], 12, ink["muted"])
    tone = material.recipe.tone
    panel_entries = []
    for i, (panel, rect) in enumerate(zip(panels, rectangles), 1):
        draw.rounded_rectangle(tuple(rect.as_list()), radius=7, fill=ink["panel"], outline=ink["border"], width=1)
        inner = Box(rect.x0 + 38, rect.y0 + 42, rect.x1 - 17, rect.y1 - 57)
        stretch = (tone.black, tone.white) if panel.mode == "stretch" else (0, 1)
        info = place_film(image, material.pixels, panel.area, material.recipe.orientation, inner, tone=stretch)
        if panel.with_guides:
            crop_span = (panel.area.y0, panel.area.y1)
            selections = [(material.recipe.aperture.box(*crop_span), TEAL, 2),
                          (material.recipe.perforations.box(*crop_span), ACCENT, 2)]
            native_marks(draw, info, selections)
        if not light:
            film_rulers(draw, info, desired=4)
        render_scale = panel_annotation(draw, rect, panel, info, i, ink)
        panel_entries.append({"title": panel.label, "item": i, "plate_box": rect.as_list(), "image_scale": render_scale, **panel.contents(info.scale())})
    note = notes or "Image evidence. Labels reference native scan columns and rows."
    print_text(draw, (44, size_px[1] - 44), " ".join(wrapped_ascii(note, 120))[:120], 11, ink["muted"])
    digest = material.record["sha256"][:18]
    print_text(draw, (size_px[0] - 45, size_px[1] - 29), f"SHA256 {digest}", 11, ink["muted"], anchor="ra")
    return image, {"title": title, "parent": material.identity(), "size_pixels": list(size_px), "size_mm": list(size_mm),
                   "paper": page_size, "orientation": layout, "dpi": PRINT_DPI, "notes": notes, "panels": panel_entries}


def make_contact(material, values):
    dark, reading = contact_sheet(material, values)
    light, _ = contact_sheet(material, values, light=True)
    pdf = BytesIO()
    light.save(pdf, format="PDF", resolution=float(PRINT_DPI), title="ReelTone film contact sheet", author="ReelTone")
    printable = pdf.getvalue()
    preview = encode_image(dark, "PNG")
    reading.update({"png_sha256": sha256(preview).hexdigest(), "pdf_sha256": sha256(printable).hexdigest()})
    return Contact(preview, printable, reading)
