from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from .geometry import Box, full_box
from .optical import tonal_plane
from .raster import byte_preview, encode_image
from .visual import ACCENT, BLUE, GRID, MUTED, PAGE, TEAL, TEXT, print_text


@dataclass(frozen=True)
class Probe:
    point: tuple[int, int]
    box: Box
    values: np.ndarray
    tone: np.ndarray
    original_code: int
    region: str
    line: np.ndarray
    passage: np.ndarray
    threshold: float
    radius: int

    def summary(self):
        x, y = self.point
        center = float(self.values[y - self.box.y0, x - self.box.x0])
        center_tone = float(self.tone[y - self.box.y0, x - self.box.x0])
        return {"x": x, "y": y, "window": self.box.as_list(), "source_intensity": round(center, 7),
                "original_code": self.original_code, "decoded_intensity": round(center_tone, 7),
                "region": self.region, "above_threshold": center_tone >= self.threshold,
                "threshold": self.threshold, "patch_mean": round(float(self.values.mean()), 7),
                "patch_median": round(float(np.median(self.values)), 7), "patch_minimum": round(float(self.values.min()), 7),
                "patch_maximum": round(float(self.values.max()), 7),
                "row_profile": [round(float(value), 6) for value in self.line],
                "column_profile": [round(float(value), 6) for value in self.passage]}


def probe_at(image, recipe, depth, x, y, radius, multichannel=False):
    box = Box(x, y, x + 1, y + 1).expand(radius, radius, full_box(image))
    values = box.subimage(image)
    plane = tonal_plane(values, recipe.tone)
    scale = (1 << depth) - 1
    code = round(float(image[y, x]) * scale)
    if recipe.aperture.x0 <= x < recipe.aperture.x1:
        region = "track"
    elif recipe.perforations.x0 <= x < recipe.perforations.x1:
        region = "perforations"
    else:
        region = "surround"
    report = Probe((x, y), box, values, plane, code, region, image[y, :], image[box.y0:box.y1, x], recipe.tone.threshold, radius)
    summary = report.summary()
    summary["code_description"] = "加权灰度代码" if multichannel else "原灰度代码"
    return summary, report


def pixel_grid(probe, orientation):
    box = probe.box
    cell = max(15, min(43, 520 // max(box.width, box.height)))
    content = orientation.image(probe.values)
    tonal = orientation.image(probe.tone)
    active_x = probe.point[0] - box.x0
    active_y = probe.point[1] - box.y0
    view_x, view_y = orientation.index(active_x, active_y, box.width, box.height)
    top, left = 43, 55
    rows, cols = content.shape
    width, height = left + cols * cell + 20, top + rows * cell + 47
    page = Image.new("RGB", (width, height), PAGE)
    draw = ImageDraw.Draw(page)
    print_text(draw, (15, 11), "NATIVE PIXEL LOUPE", 15, TEAL)
    for display_y, row in enumerate(content):
        for display_x, brightness in enumerate(row):
            box_on_page = (left + display_x * cell, top + display_y * cell,
                           left + (display_x + 1) * cell - 1, top + (display_y + 1) * cell - 1)
            shade = round(float(brightness) * 255)
            draw.rectangle(box_on_page, fill=(shade, shade, shade), outline=GRID)
            if display_x == view_x and display_y == view_y:
                draw.rectangle(box_on_page, outline=ACCENT, width=3)
            elif tonal[display_y, display_x] >= probe.threshold and cell >= 26:
                draw.ellipse((box_on_page[2] - 6, box_on_page[3] - 6, box_on_page[2] - 3, box_on_page[3] - 3), fill=TEAL)
            if cell >= 35:
                color = PAGE if shade > 140 else TEXT
                print_text(draw, (box_on_page[0] + 6, box_on_page[1] + 8), f"{shade:03}", 11, color)
        if rows <= 20 or display_y % 2 == 0:
            _, ny = orientation.index(0, display_y, box.width, box.height)
            print_text(draw, (left - 6, top + display_y * cell + cell // 2), ny + box.y0, 11, MUTED, anchor="rm")
    for display_x in range(cols):
        if cols > 20 and display_x % 2:
            continue
        nx, _ = orientation.index(display_x, 0, box.width, box.height)
        print_text(draw, (left + display_x * cell + cell // 2, top + rows * cell + 6), nx + box.x0, 10, MUTED, anchor="mt")
    caption = f"Probe on x {probe.point[0]}, row {probe.point[1]}"
    print_text(draw, (13, height - 18), caption, 11, MUTED)
    return encode_image(page, "PNG")


def overlay_contour(image, read, aperture, orientation, *, top, stop, width_cap=500, height_cap=700):
    selection = image[top:stop, aperture.x0:aperture.x1]
    window = orientation.image(selection)
    width, height = min(width_cap, max(190, selection.shape[1] * 4)), min(height_cap, max(110, selection.shape[0]))
    bitmap = byte_preview(window).resize((width, height), Image.Resampling.BOX).convert("RGB")
    draw = ImageDraw.Draw(bitmap)
    start_index = top - read.source_rows[0]
    stop_index = stop - read.source_rows[0]
    left = read.edge_low[start_index:stop_index]
    right = read.edge_high[start_index:stop_index]
    if orientation.reverse_y:
        left, right = left[::-1], right[::-1]
    if orientation.mirror_x:
        left, right = aperture.x0 + aperture.x1 - right, aperture.x0 + aperture.x1 - left
    for values, color in ((left, TEAL), (right, ACCENT)):
        previous = None
        for i, native in enumerate(values):
            if not np.isfinite(native):
                previous = None
                continue
            x = min(width - 1, max(0, round((native - aperture.x0) / aperture.width * width)))
            y = round((i + 0.5) * height / len(values))
            if previous is not None:
                draw.line((*previous, x, y), fill=color, width=1)
            previous = (x, y)
    draw.line((0, height - 1, width, height - 1), fill=BLUE)
    return encode_image(bitmap, "PNG")
