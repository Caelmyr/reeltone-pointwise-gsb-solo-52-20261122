from dataclasses import dataclass

import numpy as np

from .errors import BenchError
from .perforations import PerforationReport
from .recipe import ClockSetting


@dataclass(frozen=True)
class RowClock:
    rows_per_second: float
    source: str
    native_height: int
    reverse: bool
    frame_rate: float
    holes_per_frame: int
    measured_pitch: float | None
    warnings: tuple[str, ...]

    @property
    def seconds(self):
        return self.native_height / self.rows_per_second

    def edge_to_time(self, native_edge):
        forward_edge = self.native_height - native_edge if self.reverse else native_edge
        return forward_edge / self.rows_per_second

    def native_center_at_time(self, time):
        forward_center = np.asarray(time) * self.rows_per_second
        return self.native_height - forward_center if self.reverse else forward_center

    def extent(self, rows):
        first, last = [self.edge_to_time(edge) for edge in rows]
        return min(first, last), max(first, last)

    def clip_frames(self, rows):
        start, stop = self.extent(rows)
        return start * self.frame_rate, stop * self.frame_rate

    def as_dict(self, rows=None):
        info = {"rows_per_second": round(self.rows_per_second, 6), "source": self.source,
                "native_height": self.native_height, "reverse": self.reverse,
                "frames_per_second": self.frame_rate, "holes_per_frame": self.holes_per_frame,
                "measured_pitch": self.measured_pitch, "seconds": round(self.seconds, 7),
                "warnings": list(self.warnings)}
        if rows is not None:
            start, stop = self.extent(rows)
            frame0, frame1 = self.clip_frames(rows)
            info["source_rows"] = list(rows)
            info["start_seconds"] = round(start, 7)
            info["stop_seconds"] = round(stop, 7)
            info["duration"] = round(stop - start, 7)
            info["frame_span"] = [round(frame0, 4), round(frame1, 4)]
        return info


def establish_clock(setting: ClockSetting, height, reverse, report: PerforationReport | None = None):
    warnings = []
    pitch = None
    if setting.model == "perfs":
        if report is None or report.estimate is None:
            raise BenchError("未检测到可靠齿孔间距。请校正齿孔带，或明确选择手动行速。", code="untrusted_clock")
        pitch = report.estimate
        rate = pitch * setting.frames_per_second * setting.holes_per_frame
        if not 100 <= rate <= 96_000:
            raise BenchError("实际孔距对应的行速超出监听范围。")
        if report.warning:
            warnings.append(report.warning)
        if report.missing_estimate:
            warnings.append(f"检测到可能缺失的齿孔约 {report.missing_estimate} 个。采用全片代表速度，未填补缺失影像。")
    else:
        rate = setting.row_rate
    return RowClock(rate, setting.model, height, reverse, setting.frames_per_second,
                    setting.holes_per_frame, pitch, tuple(warnings))


def format_side_time(seconds):
    milliseconds = round(seconds * 1000)
    minutes, remainder = divmod(milliseconds, 60_000)
    whole, fractions = divmod(remainder, 1000)
    return f"{minutes:02}:{whole:02}.{fractions:03}"


def clock_table(clock, rows, ticks=9):
    a, b = rows
    mark_count = min(b - a + 1, ticks)
    entries = []
    for edge in np.rint(np.linspace(a, b, mark_count)).astype(int):
        time = clock.edge_to_time(int(edge))
        entries.append({"native_edge": int(edge), "seconds": round(time, 7), "timecode": format_side_time(time),
                        "frame_position": round(time * clock.frame_rate, 4)})
    if clock.reverse:
        entries.reverse()
    return entries
