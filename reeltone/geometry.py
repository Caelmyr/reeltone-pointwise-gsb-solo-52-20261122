from dataclasses import asdict, dataclass

import numpy as np

from .contracts import flag, integer, mapping, sequence
from .errors import BenchError


@dataclass(frozen=True)
class Box:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self):
        return self.x1 - self.x0

    @property
    def height(self):
        return self.y1 - self.y0

    @property
    def area(self):
        return self.width * self.height

    def as_list(self):
        return [self.x0, self.y0, self.x1, self.y1]

    def slices(self):
        return slice(self.y0, self.y1), slice(self.x0, self.x1)

    def subimage(self, img):
        return img[self.slices()]

    def intersection(self, other):
        x0 = max(self.x0, other.x0)
        y0 = max(self.y0, other.y0)
        x1 = min(self.x1, other.x1)
        y1 = min(self.y1, other.y1)
        if x0 >= x1 or y0 >= y1:
            return None
        return Box(x0, y0, x1, y1)

    def expand(self, x_pad, y_pad, bounds):
        asked = Box(self.x0 - x_pad, self.y0 - y_pad, self.x1 + x_pad, self.y1 + y_pad)
        return asked.intersection(bounds)

    @classmethod
    def parse(cls, raw, width, height, *, minimum=(1, 1), name="选区"):
        c = sequence(raw, name, minimum=4, maximum=4)
        x0 = integer(c[0], f"{name}左列", 0, width - 1)
        y0 = integer(c[1], f"{name}首行", 0, height - 1)
        x1 = integer(c[2], f"{name}右边界", x0 + 1, width)
        y1 = integer(c[3], f"{name}末行之后", y0 + 1, height)
        box = cls(x0, y0, x1, y1)
        if box.width < minimum[0] or box.height < minimum[1]:
            raise BenchError(f"{name}至少为 {minimum[0]} 列 × {minimum[1]} 行。")
        return box


@dataclass(frozen=True)
class Guide:
    x0: int
    x1: int

    @property
    def width(self):
        return self.x1 - self.x0

    def as_list(self):
        return [self.x0, self.x1]

    def box(self, top, bottom):
        return Box(self.x0, top, self.x1, bottom)

    @classmethod
    def parse(cls, raw, width, name, minimum=4):
        pair = sequence(raw, name, minimum=2, maximum=2)
        left = integer(pair[0], f"{name}左列", 0, width - minimum)
        right = integer(pair[1], f"{name}右边界", left + minimum, width)
        return cls(left, right)


@dataclass(frozen=True)
class Orientation:
    mirror_x: bool = False
    reverse_y: bool = False

    def as_dict(self):
        return asdict(self)

    def image(self, data):
        rows = slice(None, None, -1) if self.reverse_y else slice(None)
        columns = slice(None, None, -1) if self.mirror_x else slice(None)
        return data[rows, columns]

    def box(self, box, width, height):
        x0, x1 = (width - box.x1, width - box.x0) if self.mirror_x else (box.x0, box.x1)
        y0, y1 = (height - box.y1, height - box.y0) if self.reverse_y else (box.y0, box.y1)
        return Box(x0, y0, x1, y1)

    def index(self, x, y, width, height):
        return (width - 1 - x if self.mirror_x else x, height - 1 - y if self.reverse_y else y)

    def guide(self, guide, width):
        return Guide(width - guide.x1, width - guide.x0) if self.mirror_x else guide

    @classmethod
    def parse(cls, raw):
        raw = mapping(raw, "走片方向")
        return cls(flag(raw.get("mirror_x", False), "左右镜像"), flag(raw.get("reverse_y", False), "反向走片"))


def full_box(image):
    return Box(0, 0, image.shape[1], image.shape[0])


def runs(mask, min_count=1):
    values = np.asarray(mask, dtype=bool)
    edges = np.diff(values, prepend=False, append=False)
    breaks = np.flatnonzero(edges)
    return [(int(a), int(b)) for a, b in zip(breaks[::2], breaks[1::2]) if b - a >= min_count]
