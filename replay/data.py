"""Replay data -- parses .osr files and provides interpolated cursor queries."""

from __future__ import annotations

import bisect
import threading
from dataclasses import dataclass, field

import numpy as np
from osupyparser.osr.osr_parser import ReplayFile
from osu_map.mods import HR, normalize_mods
from speedups import interpolate_cursor_query, keys_index_at

OSU_H = 384
KEY_M1 = 1
KEY_M2 = 2
KEY_K1 = 4
KEY_K2 = 8
KEY_ANY = KEY_M1 | KEY_M2 | KEY_K1 | KEY_K2

_REPLAYFILE_PARSE_LOCK = threading.Lock()


@dataclass(slots=True)
class _ReplayFileSnapshot:
    player_name: str
    mods: int
    score: int
    max_combo: int
    n300: int
    n100: int
    n50: int
    nmiss: int
    map_md5: str
    frames: list


@dataclass
class RFrame:
    """Single replay frame with absolute timestamp."""
    time_ms: float
    x: float
    y: float
    keys: int


@dataclass
class KeyPress:
    """A newly pressed key event extracted from replay frames."""
    time_ms: float
    keys: int


@dataclass(slots=True)
class ReplaySummary:
    """Lightweight metadata used before full replay parsing."""

    path: str
    player_name: str
    mods: int
    score: int
    max_combo: int
    n300: int
    n100: int
    n50: int
    nmiss: int
    map_md5: str


class ReplayData:
    """Parsed replay with fast cursor/key queries."""

    def __init__(self, frames: list[RFrame], player_name: str = "",
                 mods: int = 0, score: int = 0, max_combo: int = 0,
                 n300: int = 0, n100: int = 0, n50: int = 0, nmiss: int = 0,
                 map_md5: str = ""):
        self.frames = frames
        self.player_name = player_name
        self.mods = mods
        self.score = score
        self.max_combo = max_combo
        self.n300 = n300
        self.n100 = n100
        self.n50 = n50
        self.nmiss = nmiss
        self.map_md5 = map_md5

        self._times = np.array([f.time_ms for f in frames], dtype=np.float64)
        self._xs = np.array([f.x for f in frames], dtype=np.float64)
        self._ys = np.array([f.y for f in frames], dtype=np.float64)
        self._keys = np.array([f.keys for f in frames], dtype=np.int32)

        self._presses: list[KeyPress] = []
        self._build_presses()
        self._key_intervals = self._build_key_intervals()
        self._key_interval_start_times = np.array(
            [start_ms for _key, start_ms, _end_ms in self._key_intervals],
            dtype=np.float64,
        )

        self._press_times = np.array(
            [p.time_ms for p in self._presses], dtype=np.float64
        )
        self._last_pos_idx = 0
        self._last_keys_idx = 0

    def _locate_time_index(self, time_ms: float, *, anchor: int) -> int:
        n = len(self._times)
        if n == 0:
            return 0
        anchor = max(0, min(anchor, n))
        if anchor > 0 and time_ms < self._times[anchor - 1]:
            anchor = bisect.bisect_right(self._times, time_ms, 0, anchor)
        elif anchor < n and time_ms >= self._times[anchor]:
            anchor = bisect.bisect_right(self._times, time_ms, anchor, n)
        return anchor

    @staticmethod
    def _read_replay_file(path: str):
        # osupyparser's ReplayFile parser mutates shared class state internally,
        # and even returns the class object itself. Snapshot the parsed values
        # while holding the lock so later parses cannot mutate prior results.
        with _REPLAYFILE_PARSE_LOCK:
            rp = ReplayFile.from_file(path)
            return _ReplayFileSnapshot(
                player_name=rp.player_name or "",
                mods=int(rp.mods),
                score=int(rp.score),
                max_combo=int(rp.max_combo),
                n300=int(rp.n300),
                n100=int(rp.n100),
                n50=int(rp.n50),
                nmiss=int(rp.nmiss),
                map_md5=rp.map_md5 or "",
                frames=list(rp.frames),
            )

    @staticmethod
    def peek_mods(path: str) -> int:
        """Read only the mods bitmask from an .osr file without full parsing."""
        rp = ReplayData._read_replay_file(path)
        return rp.mods

    @staticmethod
    def peek_summary(path: str) -> ReplaySummary:
        """Read replay metadata without building interpolated frame arrays."""
        rp = ReplayData._read_replay_file(path)
        return ReplaySummary(
            path=path,
            player_name=rp.player_name,
            mods=rp.mods,
            score=rp.score,
            max_combo=rp.max_combo,
            n300=rp.n300,
            n100=rp.n100,
            n50=rp.n50,
            nmiss=rp.nmiss,
            map_md5=rp.map_md5,
        )

    @classmethod
    def from_file(cls, path: str) -> ReplayData:
        rp = cls._read_replay_file(path)
        frames: list[RFrame] = []
        abs_time = 0.0
        for f in rp.frames:
            abs_time += f.delta
            frames.append(RFrame(
                time_ms=abs_time,
                x=float(f.x),
                y=float(f.y),
                keys=int(f.keys),
            ))
        return cls(
            frames=frames,
            player_name=rp.player_name,
            mods=rp.mods,
            score=rp.score,
            max_combo=rp.max_combo,
            n300=rp.n300,
            n100=rp.n100,
            n50=rp.n50,
            nmiss=rp.nmiss,
            map_md5=rp.map_md5,
        )

    def _build_presses(self) -> None:
        """Extract new key-press events (key was up, now down)."""
        prev_keys = 0
        for f in self.frames:
            new_keys = f.keys & ~prev_keys & KEY_ANY
            if new_keys:
                self._presses.append(KeyPress(f.time_ms, new_keys))
            prev_keys = f.keys

    def _build_key_intervals(self) -> list[tuple[int, float, float]]:
        """Build K1/K2 hold intervals for HUD overlay rebuilds."""
        intervals: list[tuple[int, float, float]] = []
        prev_k1 = False
        prev_k2 = False
        press_start_1 = -1.0
        press_start_2 = -1.0

        for f in self.frames:
            k1_now = bool(f.keys & (KEY_M1 | KEY_K1))
            k2_now = bool(f.keys & (KEY_M2 | KEY_K2))

            if k1_now and not prev_k1:
                press_start_1 = float(f.time_ms)
            elif prev_k1 and not k1_now and press_start_1 >= 0.0:
                intervals.append((1, press_start_1, float(f.time_ms)))
                press_start_1 = -1.0

            if k2_now and not prev_k2:
                press_start_2 = float(f.time_ms)
            elif prev_k2 and not k2_now and press_start_2 >= 0.0:
                intervals.append((2, press_start_2, float(f.time_ms)))
                press_start_2 = -1.0

            prev_k1 = k1_now
            prev_k2 = k2_now

        if prev_k1 and press_start_1 >= 0.0:
            intervals.append((1, press_start_1, float("inf")))
        if prev_k2 and press_start_2 >= 0.0:
            intervals.append((2, press_start_2, float("inf")))

        return intervals

    def position_at(self, time_ms: float) -> tuple[float, float]:
        """Interpolated cursor position at the given time (osu! coords, Y-down)."""
        n = len(self._times)
        if n == 0:
            return 256.0, 192.0
        fast_result = interpolate_cursor_query(self._times, self._xs, self._ys, time_ms, self._last_pos_idx)
        if fast_result is not None:
            idx, x, y = fast_result
            self._last_pos_idx = int(idx)
            return float(x), float(y)
        idx = self._locate_time_index(time_ms, anchor=self._last_pos_idx)
        self._last_pos_idx = idx
        if idx <= 0:
            return float(self._xs[0]), float(self._ys[0])
        if idx >= n:
            return float(self._xs[-1]), float(self._ys[-1])
        t0, t1 = self._times[idx - 1], self._times[idx]
        dt = t1 - t0
        if dt < 0.01:
            return float(self._xs[idx]), float(self._ys[idx])
        frac = (time_ms - t0) / dt
        x = self._xs[idx - 1] + (self._xs[idx] - self._xs[idx - 1]) * frac
        y = self._ys[idx - 1] + (self._ys[idx] - self._ys[idx - 1]) * frac
        return float(x), float(y)

    def keys_at(self, time_ms: float) -> int:
        """Key state at the given time."""
        n = len(self._times)
        if n == 0:
            return 0
        fast_idx = keys_index_at(self._times, time_ms, self._last_keys_idx)
        if fast_idx is not None:
            idx = int(fast_idx)
            self._last_keys_idx = max(0, idx)
            if idx < 0:
                return 0
            return int(self._keys[min(idx, n - 1)])
        idx = self._locate_time_index(time_ms, anchor=self._last_keys_idx) - 1
        self._last_keys_idx = max(0, idx)
        if idx < 0:
            return 0
        return int(self._keys[min(idx, n - 1)])

    def new_presses_between(self, t0: float, t1: float) -> list[KeyPress]:
        """All new key presses in the half-open interval (t0, t1]."""
        if not self._presses:
            return []
        lo = bisect.bisect_right(self._press_times, t0)
        hi = bisect.bisect_right(self._press_times, t1)
        return self._presses[lo:hi]

    def with_target_mods(self, target_mods: int) -> "ReplayData":
        """Return a replay remapped to match the target mod space.

        Only HR currently changes replay cursor coordinates directly, so
        toggling HR is implemented as a vertical mirror of the replay frames.
        Timing conversion for DT/HT is handled by gameplay clock-rate logic.
        """
        target_mods = normalize_mods(target_mods)
        source_hr = bool(normalize_mods(self.mods) & HR)
        target_hr = bool(target_mods & HR)
        if source_hr == target_hr:
            return self

        flipped_frames = [
            RFrame(
                time_ms=frame.time_ms,
                x=frame.x,
                y=OSU_H - frame.y,
                keys=frame.keys,
            )
            for frame in self.frames
        ]
        return ReplayData(
            frames=flipped_frames,
            player_name=self.player_name,
            mods=target_mods,
            score=self.score,
            max_combo=self.max_combo,
            n300=self.n300,
            n100=self.n100,
            n50=self.n50,
            nmiss=self.nmiss,
            map_md5=self.map_md5,
        )
