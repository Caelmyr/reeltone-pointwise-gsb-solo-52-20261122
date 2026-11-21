from dataclasses import dataclass
import hashlib
import json

from .contracts import integer, mapping, option, real
from .errors import BenchError
from .geometry import Guide, Orientation


@dataclass(frozen=True)
class Tone:
    black: float
    white: float
    threshold: float
    polarity: str

    @classmethod
    def parse(cls, raw):
        values = mapping(raw, "透射设定")
        black = real(values.get("black", 0.10), "暗参考", 0, 1)
        white = real(values.get("white", 0.90), "亮参考", 0, 1)
        if white - black < 0.025:
            raise BenchError("亮参考与暗参考至少相差 0.025。")
        threshold = real(values.get("threshold", 0.5), "轮廓阈值", 0.08, 0.92)
        polarity = option(values.get("polarity", "clear"), ("clear", "dark"), "信号极性")
        return cls(black, white, threshold, polarity)

    def as_dict(self):
        return {"black": self.black, "white": self.white, "threshold": self.threshold, "polarity": self.polarity}


@dataclass(frozen=True)
class ClockSetting:
    model: str
    row_rate: float
    frames_per_second: float
    holes_per_frame: int
    pitch_guess: float

    @classmethod
    def parse(cls, raw):
        values = mapping(raw, "时间基准")
        model = option(values.get("model", "manual"), ("manual", "perfs", "local"), "时间基准")
        rate = real(values.get("row_rate", 6_000), "每秒扫描行", 100, 96_000)
        fps = real(values.get("frames_per_second", 24), "每秒画格", 8, 60)
        holes = integer(values.get("holes_per_frame", 1), "每格齿孔", 1, 12)
        pitch = real(values.get("pitch_guess", 250), "预估齿孔间距", 8, 5_000)
        if model in ("perfs", "local") and not 100 <= pitch * fps * holes <= 96_000:
            raise BenchError("齿孔间距和画格速率的组合须对应 100–96,000 行/秒。")
        return cls(model, rate, fps, holes, pitch)

    def as_dict(self):
        return {"model": self.model, "row_rate": self.row_rate, "frames_per_second": self.frames_per_second,
                "holes_per_frame": self.holes_per_frame, "pitch_guess": self.pitch_guess}


@dataclass(frozen=True)
class Recipe:
    aperture: Guide
    perforations: Guide
    orientation: Orientation
    tone: Tone
    clock: ClockSetting

    @classmethod
    def parse(cls, raw, width):
        values = mapping(raw, "读片配方")
        aperture = Guide.parse(values.get("aperture"), width, "声轨读窗", minimum=8)
        perfs = Guide.parse(values.get("perforations"), width, "齿孔带", minimum=4)
        if min(aperture.x1, perfs.x1) > max(aperture.x0, perfs.x0):
            raise BenchError("声轨读窗和齿孔带不能重叠。")
        orientation = Orientation.parse(values.get("orientation", {}))
        tone = Tone.parse(values.get("tone", {}))
        clock = ClockSetting.parse(values.get("clock", {}))
        return cls(aperture, perfs, orientation, tone, clock)

    @classmethod
    def default(cls, width):
        perf_width = max(4, int(width * 0.12))
        track_left = max(perf_width + 1, int(width * 0.24))
        track_right = min(width, max(track_left + 8, int(width * 0.57)))
        return cls.parse({"aperture": [track_left, track_right], "perforations": [1, perf_width + 1]}, width)

    def as_dict(self):
        return {"aperture": self.aperture.as_list(), "perforations": self.perforations.as_list(),
                "orientation": self.orientation.as_dict(), "tone": self.tone.as_dict(), "clock": self.clock.as_dict()}

    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"),
                                         ensure_ascii=True).encode("ascii")).hexdigest()
