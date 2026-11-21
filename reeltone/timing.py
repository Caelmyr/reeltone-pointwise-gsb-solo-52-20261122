from dataclasses import dataclass
from functools import cached_property

import numpy as np

from .errors import BenchError
from .perforations import PerforationReport
from .recipe import ClockSetting


@dataclass(frozen=True)
class ClockSegment:
    """A stretch of scan between two adjacent confirmed perforations."""

    forward_start: float | None
    forward_stop: float | None
    rate: float
    trusted: bool
    native_rows: tuple | None = None
    steps: int | None = None

    def as_dict(self):
        return {"native_rows": list(self.native_rows) if self.native_rows else None,
                "steps": self.steps, "rate": round(self.rate, 6), "trusted": self.trusted}


@dataclass(frozen=True)
class EstimatedSpan:
    """A stretch where the adjacent hole spacing could not be confirmed.

    Listening stays possible there by carrying the nearest confirmed cadence,
    but the span is disclosed so downstream results can say which times are
    estimates.  Confirmed intervals on either side keep their own measured rate.
    """

    native_rows: tuple
    rate: float
    reason: str = "无法确认孔数或间距的区段，按相邻已确认节奏估算"

    def as_dict(self):
        return {"native_rows": list(self.native_rows), "rate": round(self.rate, 6), "reason": self.reason}


@dataclass(frozen=True)
class RowClock:
    """A scan-row-to-time mapping, uniform or varying along the scan.

    ``edge_to_time`` and ``native_center_at_time`` are the only conversions
    callers should use: with a local perforation clock the rows-per-second value
    varies, so multiplying rows by a single rate is no longer valid.
    """

    rows_per_second: float
    source: str
    native_height: int
    reverse: bool
    frame_rate: float
    holes_per_frame: int
    measured_pitch: float | None
    warnings: tuple[str, ...]
    segments: tuple = ()
    estimated_spans: tuple = ()

    @property
    def local(self):
        return bool(self.segments)

    @property
    def seconds(self):
        if not self.segments:
            return self.native_height / self.rows_per_second
        return float(self._piecewise[1][-1])

    def _forward(self, native):
        if isinstance(native, np.ndarray):
            return self.native_height - native if self.reverse else native.astype(float)
        return float(self.native_height - native) if self.reverse else float(native)

    def _native(self, forward):
        return self.native_height - forward if self.reverse else forward

    @cached_property
    def _piecewise(self):
        """Knot positions, cumulative time and rate for each uniform piece."""
        if not self.segments:
            return np.array([0.0, float(self.native_height)]), np.array([0.0, self.seconds]), \
                np.array([self.rows_per_second, self.rows_per_second], dtype=float)
        segments = sorted(self.segments, key=lambda item: item.forward_start)
        limits = {0.0, float(self.native_height)}
        for segment in segments:
            limits.add(float(np.clip(segment.forward_start, 0, self.native_height)))
            limits.add(float(np.clip(segment.forward_stop, 0, self.native_height)))
        knots = np.array(sorted(limits), dtype=float)
        rates = np.empty(max(1, len(knots) - 1), dtype=float)
        for index, (start, stop) in enumerate(zip(knots[:-1], knots[1:])):
            middle = 0.5 * (start + stop)
            rate = next((segment.rate for segment in segments if segment.forward_start <= middle <= segment.forward_stop), None)
            if rate is None:
                before = [segment for segment in segments if segment.forward_stop <= middle]
                after = [segment for segment in segments if segment.forward_start >= middle]
                rate = before[-1].rate if before else after[0].rate if after else self.rows_per_second
            rates[index] = rate
        steps = np.diff(knots) / rates
        cumulative = np.concatenate(([0.0], np.cumsum(steps)))
        return knots, cumulative, rates

    def duration(self, rows):
        start, stop = self.extent(rows)
        return stop - start

    def rate_at(self, native_row):
        if not self.segments:
            return self.rows_per_second
        knots, _, rates = self._piecewise
        forward = float(np.clip(self._forward(native_row), 0, self.native_height))
        index = min(len(rates) - 1, max(0, int(np.searchsorted(knots, forward, side="right") - 1)))
        return float(rates[index])

    def rate_range(self, rows):
        if not self.segments:
            return (self.rows_per_second, self.rows_per_second)
        first, second = sorted(self._forward(edge) for edge in rows)
        sampled = [self.rate_at(self._native(value)) for value in np.linspace(first, second, 33)]
        return (min(sampled), max(sampled))

    def edge_to_time(self, native_edge):
        array = np.asarray(native_edge, dtype=float)
        forward = (self.native_height - array if self.reverse else array) / self.rows_per_second
        if not self.segments:
            return forward if np.ndim(native_edge) else float(forward)
        forward = np.clip(self.native_height - array if self.reverse else array, 0.0, self.native_height)
        knots, cumulative, rates = self._piecewise
        indexes = np.clip(np.searchsorted(knots, forward, side="right") - 1, 0, len(rates) - 1)
        times = cumulative[indexes] + (forward - knots[indexes]) / rates[indexes]
        return times if np.ndim(native_edge) else float(times)

    def forward_at_time(self, time):
        times = np.clip(np.asarray(time, dtype=float), 0.0, self.seconds)
        if not self.segments:
            return times * self.rows_per_second
        knots, cumulative, _ = self._piecewise
        forward = np.interp(times, cumulative, knots)
        return forward if np.ndim(time) else float(forward)

    def native_center_at_time(self, time):
        forward = self.forward_at_time(time)
        native = self.native_height - forward if self.reverse else forward
        return native if np.ndim(time) else float(native)

    def span_at_time(self, seconds):
        return int(round(self.native_center_at_time(seconds)))

    def extent(self, rows):
        first, last = [self.edge_to_time(edge) for edge in rows]
        return min(first, last), max(first, last)

    def clip_frames(self, rows):
        start, stop = self.extent(rows)
        return start * self.frame_rate, stop * self.frame_rate

    def estimated_spans_between(self, first, second, *, base=0.0):
        """Estimated stretches overlapping the rows, with times relative to base."""
        low, high = min(first, second), max(first, second)
        result = []
        for span in self.estimated_spans:
            a, b = span.native_rows
            start, stop = max(a, low), min(b, high)
            if start >= stop:
                continue
            times = sorted((self.edge_to_time(start), self.edge_to_time(stop)))
            result.append({"native_rows": [start, stop], "rate": round(span.rate, 6),
                           "seconds": [round(times[0] - base, 7), round(times[1] - base, 7)],
                           "timecodes": [format_side_time(times[0] - base), format_side_time(times[1] - base)],
                           "reason": span.reason})
        return result

    def as_dict(self, rows=None):
        info = {"rows_per_second": round(self.rows_per_second, 6), "source": self.source,
                "native_height": self.native_height, "reverse": self.reverse,
                "frames_per_second": self.frame_rate, "holes_per_frame": self.holes_per_frame,
                "measured_pitch": self.measured_pitch, "seconds": round(self.seconds, 7),
                "mapping": "local_perforation" if self.local else "uniform", "warnings": list(self.warnings)}
        if self.local:
            info["local_segments"] = [segment.as_dict() for segment in self.segments]
            rate_low, rate_high = self.rate_range(rows or (0, self.native_height))
            info["local_rate_range"] = [round(rate_low, 6), round(rate_high, 6)]
            info["estimated_spans"] = [span.as_dict() for span in self.estimated_spans]
        if rows is not None:
            start, stop = self.extent(rows)
            frame0, frame1 = self.clip_frames(rows)
            info["source_rows"] = list(rows)
            info["start_seconds"] = round(start, 7)
            info["stop_seconds"] = round(stop, 7)
            info["duration"] = round(stop - start, 7)
            info["frame_span"] = [round(frame0, 4), round(frame1, 4)]
            estimated = self.estimated_spans_between(*rows, base=start)
            info["estimated_spans_in_range"] = estimated
            info["estimated_seconds"] = round(sum(item["seconds"][1] - item["seconds"][0] for item in estimated), 7)
        return info


def local_perforation_mapping(report: PerforationReport, setting, height, reverse, representative):
    """Turn directly observed adjacent-hole intervals into a row-varying time base.

    Only one-step intervals between two directly detected holes anchor the
    clock.  Stretches where a hole was hidden (several pitch lengths between
    detections, possibly taped over) keep listening possible by carrying the
    nearest confirmed cadence, but they are returned as explicit estimated
    spans.  Confirmed intervals on either side keep their own measured rate.
    """
    rate_of = lambda pitch: pitch * setting.frames_per_second * setting.holes_per_frame
    forward = lambda native: height - native if reverse else native
    heights = float(height)

    trusted = []
    for interval in report.spacing:
        if not interval.trusted or interval.steps != 1:
            continue
        rate = rate_of(interval.pitch)
        if not 100 <= rate <= 96_000:
            continue
        a, b = sorted((forward(interval.first), forward(interval.second)))
        start, stop = max(0.0, a), min(heights, b)
        if stop - start < 1e-6:
            continue
        native = tuple(sorted((interval.first, interval.second)))
        trusted.append((start, stop, rate, native))
    if not trusted:
        return (), ()

    trusted.sort()
    pieces = []
    estimated = []
    cursor = 0.0
    previous_rate = trusted[0][2]
    for start, stop, rate, native in trusted:
        start = max(start, cursor)
        if start > cursor + 1e-6:
            span_native = tuple(sorted((height - start if reverse else start,
                                        height - cursor if reverse else cursor)))
            reason = gap_reason(report, span_native, at_edge=False)
            pieces.append(ClockSegment(cursor, start, previous_rate, False, span_native))
            estimated.append(EstimatedSpan(span_native, previous_rate, reason))
        if stop > start + 1e-6:
            pieces.append(ClockSegment(start, stop, rate, True, native, 1))
        cursor = max(cursor, stop)
        previous_rate = rate
    if cursor < heights - 1e-6:
        span_native = tuple(sorted((height - cursor if reverse else cursor, 0 if reverse else height)))
        pieces.append(ClockSegment(cursor, heights, previous_rate, False, span_native))
        estimated.append(EstimatedSpan(span_native, previous_rate, "区间首尾没有可确认的相邻孔距，按相邻节奏估算"))
    return tuple(pieces), tuple(estimated)


def gap_reason(report, native_rows, *, at_edge):
    if at_edge:
        return "区间首尾没有可确认的相邻孔距，按相邻节奏估算"
    low, high = sorted(native_rows)
    for interval in report.spacing:
        a, b = sorted((interval.first, interval.second))
        if a <= low + 0.5 and b >= high - 0.5:
            if interval.trusted and interval.steps and interval.steps > 1:
                return "跨过多个齿孔，孔数无法确认，按相邻节奏估算"
            return "相邻孔距无法确认，按相邻节奏估算"
    return "未检测到可靠齿孔，按相邻节奏估算"



def clock_from_dict(info, fallback_rate=None):
    """Rebuild the saved row-to-time mapping, so cue cuts and replays use it."""
    height = int(info["native_height"])
    reverse = bool(info["reverse"])
    representative = float(info.get("rows_per_second") or fallback_rate or 1.0)
    segments = []
    for entry in info.get("local_segments") or ():
        native = entry.get("native_rows")
        if not native:
            continue
        rate = float(entry.get("rate") or representative)
        a, b = sorted((height - native[0] if reverse else native[0], height - native[1] if reverse else native[1]))
        segments.append(ClockSegment(a, b, rate, bool(entry.get("trusted")), tuple(sorted(native)), entry.get("steps")))
    estimated = []
    for entry in info.get("estimated_spans") or ():
        native = tuple(sorted(entry["native_rows"]))
        estimated.append(EstimatedSpan(native, float(entry.get("rate") or representative),
                                       entry.get("reason", "无法确认孔数或间距的区段，按相邻已确认节奏估算")))
    if not segments:
        segments, estimated = (), ()
    return RowClock(representative, info.get("source", "perfs"), height, reverse,
                    float(info.get("frames_per_second") or 24), int(info.get("holes_per_frame") or 1),
                    info.get("measured_pitch"), tuple(info.get("warnings") or ()),
                    tuple(segments), tuple(estimated))


def establish_clock(setting: ClockSetting, height, reverse, report: PerforationReport | None = None):
    warnings = []
    pitch = None
    if setting.model in ("perfs", "local"):
        if report is None or report.estimate is None:
            raise BenchError("未检测到可靠齿孔间距。请校正齿孔带，或明确选择手动行速。", code="untrusted_clock")
        pitch = report.estimate
        rate = pitch * setting.frames_per_second * setting.holes_per_frame
        if not 100 <= rate <= 96_000:
            raise BenchError("实际孔距对应的行速超出监听范围。")
        if report.warning:
            warnings.append(report.warning)
        if report.missing_estimate:
            warnings.append(f"检测到可能缺失的齿孔约 {report.missing_estimate} 个。")
        segments, estimated = local_perforation_mapping(report, setting, height, reverse, rate)
        local = setting.model == "local"
        if local and not segments:
            raise BenchError("所选区间没有可确认的相邻孔距，无法建立局部齿孔节奏。可改用手动行速。",
                             code="untrusted_clock")
        if local:
            trusted_count = sum(segment.trusted for segment in segments)
            warnings.append(f"按局部齿孔节奏定时：{trusted_count} 段按实测相邻孔距各行速独立计时。")
            if estimated:
                warnings.append(f"有 {len(estimated)} 段齿孔间距无法确认，该处监听按相邻已确认节奏延续，"
                                f"结果中已标出估算时间；其后能确认的区段仍按各自观测节奏计时。")
        elif report.missing_estimate:
            warnings.append("采用全片代表速度，未填补缺失影像。")
        return RowClock(rate, setting.model, height, reverse, setting.frames_per_second,
                        setting.holes_per_frame, pitch, tuple(warnings),
                        segments if local else (), estimated if local else ())
    rate = setting.row_rate
    return RowClock(rate, setting.model, height, reverse, setting.frames_per_second,
                    setting.holes_per_frame, pitch, tuple(warnings))


def format_side_time(seconds):
    sign = "-" if seconds < 0 else ""
    milliseconds = round(abs(seconds) * 1000)
    minutes, remainder = divmod(milliseconds, 60_000)
    whole, fractions = divmod(remainder, 1000)
    return f"{sign}{minutes:02}:{whole:02}.{fractions:03}"


def clock_table(clock, rows, ticks=9):
    a, b = rows
    span = max(1, round((b - a) / max(1, ticks - 1)))
    mark_count = min(b - a + 1, ticks)
    entries = []
    for edge in np.rint(np.linspace(a, b, mark_count)).astype(int):
        time = clock.edge_to_time(int(edge))
        entries.append({"native_edge": int(edge), "seconds": round(time, 7), "timecode": format_side_time(time),
                        "frame_position": round(time * clock.frame_rate, 4),
                        "estimated": bool(clock.estimated_spans_between(int(edge), int(edge) + span))})
    if clock.reverse:
        entries.reverse()
    return entries
