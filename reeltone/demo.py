from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .raster import encode_png, read_scan
from .recipe import Recipe


@dataclass(frozen=True)
class FilmSample:
    name: str
    filename: str
    note: str
    width: int
    height: int
    row_rate: int
    pitch: int
    gauge: str
    seed: int
    variant: str
    reverse: bool = False
    mirror: bool = False
    negative: bool = False


INK_CART = (
    FilmSample("码头口述 · 首次扫描", "harbour-primary.png", "合成透光宽度声轨，齿孔间隔均匀。两个节奏段可用于区分试听起止。", 244, 7680, 7200, 300, "16 mm", 502, "primary"),
    FilmSample("码头口述 · 再扫样片", "harbour-recapture.png", "同一合成片段的第二次曝光，走片方向相反并有边缘阴影。", 244, 7680, 7200, 300, "16 mm", 502, "second", reverse=True, mirror=True),
    FilmSample("码头口述 · 磨损拷贝", "harbour-wear.png", "合成磨损图，加入短光线遮挡、胶片边缘的纵向划痕和画面区横向接片。", 244, 7680, 7200, 300, "16 mm", 502, "wear"),
)

NIGHT_FERRY = (
    FilmSample("渡轮引片 · 反差参照", "ferry-step.png", "合成反相宽度声轨，片边有均匀方块可选取明暗参考，右边缘另有五级已知透射阶梯。", 276, 9216, 9600, 100, "35 mm", 891, "wedge", negative=True),
    FilmSample("渡轮引片 · 曝光偏斜", "ferry-exposure.png", "同一个信号的合成不均匀曝光，保留较亮及较暗的边缘参照片。", 276, 9216, 9600, 100, "35 mm", 891, "slant", negative=True),
    FilmSample("渡轮引片 · 走片变速", "ferry-perfs.png", "合成穿孔节奏有缓慢波动，并模拟一个位置看不到齿孔。", 276, 9216, 9600, 100, "35 mm", 891, "speed", negative=True),
)

DEMO_REELS = (
    ({"title": "码头口述（合成）", "accession": "RT-LAB-016-A", "gauge": "16 mm",
      "note": "实验室合成声轨。三张图可用于听前后段、检查磨损、比较两次曝光。声音为程序产生的音色。", "demo_key": "harbour-v1"}, INK_CART),
    ({"title": "夜间渡轮（合成）", "accession": "RT-LAB-035-B", "gauge": "35 mm",
      "note": "实验室合成声轨。包含参照色阶、曝光不均和走片波动，适合核对极性、齿孔时间和质量证据。", "demo_key": "ferry-v1"}, NIGHT_FERRY),
)


def attack_envelope(t, onset, release, rise, fall):
    opening = np.clip((t - onset) / rise, 0, 1)
    closing = np.clip((release - t) / fall, 0, 1)
    return (np.sin(opening * np.pi / 2) * np.sin(closing * np.pi / 2)) ** 2


def simple_tones(t, sample):
    if sample.gauge == "16 mm":
        carrier = np.sin(2 * np.pi * (288 * t + 51 * t**2))
        harmonics = 0.21 * np.sin(2 * np.pi * t * 647 + 0.55)
        middle = attack_envelope(t, 0.035, 0.51, 0.028, 0.08)
        low = attack_envelope(t, 0.57, 1.045, 0.045, 0.05) * (0.72 * np.sin(2 * np.pi * 455 * t))
        return np.clip((0.68 * carrier + harmonics) * middle + low, -0.92, 0.92)
    notes = [(0.03, 0.27, 379), (0.28, 0.54, 512), (0.60, 0.94, 266)]
    values = np.zeros_like(t)
    for start, finish, hz in notes:
        active = attack_envelope(t, start, finish, 0.022, 0.068)
        carrier = 0.64 * np.sin(2 * np.pi * hz * t)
        partial = 0.13 * np.cos(2 * np.pi * hz * 2.01 * t + 0.25)
        values += (carrier + partial) * active
    return values


def geometry(sample):
    w = sample.width
    return (6, round(w * 0.145)), (round(w * 0.215), round(w * 0.595)), round(w * 0.63), round(w * 0.71)


def add_picture_border(image, sample, start, rng):
    rows, cols = image.shape
    frame_step = sample.row_rate / 24
    r = np.arange(rows)[:, None]
    c = np.arange(cols - start)[None, :]
    count = int(np.ceil(rows / frame_step)) + 2
    exposure = rng.uniform(0.34, 0.57, count)
    frame = np.floor(r / frame_step).astype(int)
    band = np.exp(-((r % frame_step - frame_step * 0.52) / (frame_step * 0.40)) ** 2)
    scene_edges = 0.11 * np.sin(c / 9.0 + frame * 0.18) + 0.06 * np.cos(c / 3.8 - r / 102.0)
    base = 0.21 + exposure[frame] * band + scene_edges + 0.018 * rng.normal(size=(rows, cols - start))
    gutter = (r % frame_step) < max(4, frame_step * 0.05)
    image[:, start:] = np.where(gutter, 0.17 + 0.06 * scene_edges, base)


def perforated_centers(sample):
    base = np.arange(sample.pitch * 0.47, sample.height - 8, sample.pitch)
    if sample.variant == "speed":
        base = base + 21 * np.sin(np.arange(len(base)) / 8.3)
        base = np.delete(base, len(base) // 3)
    return base


def perforated_edge(image, sample, band, rng):
    start, stop = band
    c = np.arange(start, stop)[None, :]
    center = c.mean() + 0.32
    semi_width = (stop - start) * 0.43
    semi_height = sample.pitch * (0.19 if sample.gauge == "16 mm" else 0.24)
    for hole in perforated_centers(sample):
        a, b = max(0, int(hole - semi_height - 2)), min(sample.height, int(hole + semi_height + 3))
        r = np.arange(a, b)[:, None]
        shape = (((r - hole) / semi_height) ** 4 + ((c - center) / semi_width) ** 4) ** 0.25
        inner = np.clip((1.02 - shape) * 12, 0, 1)
        local = image[a:b, start:stop]
        transmission = 0.964 + 0.007 * rng.normal(size=shape.shape)
        image[a:b, start:stop] = local + inner * (transmission - local)


def add_sound(image, sample, interval, rng):
    r = np.arange(sample.height, dtype=np.float64)
    x = np.arange(sample.width, dtype=np.float64)
    wave = simple_tones(r / sample.row_rate, sample)
    x0, x1 = interval
    track_midpoint = 0.5 * (x0 + x1)
    center = track_midpoint + 0.55 * np.sin(r / 400) + 0.35 * np.cos(r / 268)
    semi_width = (x1 - x0) * (0.234 + 0.145 * wave)
    strength = np.clip((semi_width[:, None] + 0.72 - np.abs(x[None, :] - center[:, None])) / 1.44, 0, 1)
    base = 0.135 + 0.008 * rng.normal(size=(sample.height, 1))
    level = 0.85 + 0.013 * np.cos(r / 680)
    signal = base + strength * (level[:, None] - base)
    if sample.negative:
        signal = 0.96 - signal
    image[:, x0:x1] = signal[:, x0:x1]


def add_reference_blocks(image, start, stop, sample):
    span = (stop - start - 2) // 2
    if span < 2:
        return
    image[:, start:start + span] = 0.12
    image[:, start + span + 1:stop - 1] = 0.86
    row = slice(int(sample.height * 0.68), int(sample.height * 0.76))
    image[row, start:start + span] += 0.025
    image[row, start + span + 1:stop - 1] -= 0.025


def print_wedge(image):
    height, width = image.shape
    start, stop = width - 23, width - 5
    for index, level in enumerate((0.08, 0.27, 0.48, 0.68, 0.87)):
        row0 = round(height * index / 5) + 18
        row1 = round(height * (index + 1) / 5) - 18
        surface = image[row0:row1, start:stop]
        curve = 0.073 + 0.84 * level ** 1.24
        surface[:] = curve + (surface - np.median(surface)) * 0.018


def add_wear(image, sample, interval, rng):
    rows, cols = image.shape
    center = round(sum(interval) / 2)
    r = np.arange(rows)[:, None]
    c = np.arange(cols)[None, :]
    start, stop = round(rows * 0.22), round(rows * 0.62)
    y_mask = np.clip((r - start) / 12, 0, 1) * np.clip((stop - r) / 20, 0, 1)
    scrape = np.exp(-((c - (center + 5.5 + np.sin(r / 170) * 1.3)) / 1.32) ** 2) * y_mask
    image[:] = image * (1 - 0.88 * scrape) + (0.07 + 0.06 * scrape) * scrape
    for _ in range(27):
        row = int(rng.integers(250, sample.height - 250))
        col = int(rng.integers(5, int(cols * 0.64)))
        radius = int(rng.integers(1, 5))
        y = np.arange(row - radius, row + radius + 1)[:, None]
        x = np.arange(col - radius, col + radius + 1)[None, :]
        mask = np.maximum(0.0, 1 - np.hypot(y - row, x - col) / (radius + 1))
        view = image[row - radius:row + radius + 1, col - radius:col + radius + 1]
        view[:] = view * (1 - mask) + rng.choice([0.07, 0.97]) * mask


def sample_image(sample):
    rng = np.random.default_rng(sample.seed)
    perf, track, reference, picture = geometry(sample)
    rows, cols = sample.height, sample.width
    y = np.arange(rows)[:, None]
    x = np.arange(cols)[None, :]
    base = 0.255 + 0.029 * np.cos(x / 14) + 0.021 * np.sin(y / 764)
    image = base + 0.012 * rng.normal(size=(rows, cols))
    add_sound(image, sample, track, rng)
    perforated_edge(image, sample, perf, rng)
    add_picture_border(image, sample, picture, rng)
    add_reference_blocks(image, reference, picture, sample)
    if sample.variant == "wedge":
        print_wedge(image)
    if sample.variant == "wear":
        add_wear(image, sample, track, rng)
        first = round(sample.height * 0.43)
        band = image[first:first + 16, picture:]
        band[:] = 0.23 * band + 0.59
    elif sample.variant in {"second", "slant"}:
        axis = (x - 0.5 * cols) / cols
        image = (0.95 * image + 0.039) * (0.98 + 0.24 * axis) + 0.10 * np.sin(y / rows * 4.42) * (sample.variant == "slant")
        image = ndimage.gaussian_filter(image, (0.33, 0.46))
    if sample.reverse:
        image = image[::-1]
    if sample.mirror:
        image = image[:, ::-1]
    return np.clip(image, 0, 1).astype(np.float32)


def sample_recipe(sample):
    perf, track, _, _ = geometry(sample)
    if sample.mirror:
        perf = (sample.width - perf[1], sample.width - perf[0])
        track = (sample.width - track[1], sample.width - track[0])
    settings = {"aperture": track, "perforations": perf,
                "orientation": {"mirror_x": sample.mirror, "reverse_y": sample.reverse},
                "tone": {"black": 0.12, "white": 0.84 if sample.negative else 0.86, "polarity": "dark" if sample.negative else "clear"},
                "clock": {"model": "perfs", "pitch_guess": sample.pitch, "row_rate": sample.row_rate,
                          "frames_per_second": 24, "holes_per_frame": 4 if sample.gauge == "35 mm" else 1}}
    return Recipe.parse(settings, sample.width)


def seed_library(catalog):
    results = []
    for reel, samples in DEMO_REELS:
        previous = catalog.demo_reel(reel["demo_key"])
        if previous:
            results.append({"reel": catalog.get_reel(previous), "added": False, "scans": catalog.list_scans(previous)})
            continue
        drafts = []
        for sample in samples:
            png = encode_png(sample_image(sample), bits=16)
            drafts.append({"raster": read_scan(png, sample.filename), "title": sample.name, "note": sample.note,
                           "recipe": sample_recipe(sample)})
        record, fresh = catalog.seed_reel(reel, drafts)
        results.append({"reel": record, "added": fresh, "scans": catalog.list_scans(record["id"])})
    return {"training": results, "scan_count": sum(len(row["scans"]) for row in results), "synthetic": True}
