"""Lightweight opt-in runtime profiling helpers."""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class _Stat:
    total_ms: float = 0.0
    count: int = 0
    min_ms: float = float("inf")
    max_ms: float = 0.0
    samples_ms: list[float] | None = None


class RuntimeProfiler:
    """Aggregates timings and counters and periodically prints summaries."""

    def __init__(self) -> None:
        self.enabled = _env_flag("OSU_PROFILE")
        self.interval_s = max(0.25, float(os.environ.get("OSU_PROFILE_INTERVAL", "2.0") or 2.0))
        self.sample_limit = max(0, int(os.environ.get("OSU_PROFILE_SAMPLE_LIMIT", "4096") or 4096))
        self._lock = threading.Lock()
        self._stats: dict[str, _Stat] = {}
        self._frame_name: str | None = None
        self._frame_start: float = 0.0
        self._last_report = time.perf_counter()

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def begin_frame(self, name: str) -> None:
        if not self.enabled:
            return
        self._frame_name = name
        self._frame_start = time.perf_counter()

    def end_frame(self) -> None:
        if not self.enabled or self._frame_name is None:
            return
        elapsed_ms = (time.perf_counter() - self._frame_start) * 1000.0
        self.record(f"{self._frame_name}.frame", elapsed_ms)
        self._frame_name = None
        self._maybe_report()

    def record(self, name: str, elapsed_ms: float) -> None:
        if not self.enabled:
            return
        with self._lock:
            stat = self._stats.setdefault(
                name,
                _Stat(samples_ms=[] if self.sample_limit > 0 else None),
            )
            stat.total_ms += elapsed_ms
            stat.count += 1
            stat.min_ms = min(stat.min_ms, elapsed_ms)
            stat.max_ms = max(stat.max_ms, elapsed_ms)
            if stat.samples_ms is not None and len(stat.samples_ms) < self.sample_limit:
                stat.samples_ms.append(elapsed_ms)

    def count(self, name: str, amount: int = 1) -> None:
        if not self.enabled:
            return
        with self._lock:
            stat = self._stats.setdefault(
                name,
                _Stat(samples_ms=[] if self.sample_limit > 0 else None),
            )
            stat.count += amount

    @contextmanager
    def timer(self, name: str):
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(name, (time.perf_counter() - start) * 1000.0)

    def _maybe_report(self) -> None:
        now = time.perf_counter()
        if now - self._last_report < self.interval_s:
            return
        self.report_now()

    def snapshot(self, *, reset: bool = False) -> dict[str, dict[str, float | int]]:
        with self._lock:
            snapshot = self._stats
            if reset:
                self._stats = {}
        return {
            name: self._summarize_stat(stat)
            for name, stat in snapshot.items()
        }

    def report_now(self, *, reset: bool = True) -> dict[str, dict[str, float | int]]:
        snapshot = self.snapshot(reset=reset)
        self._last_report = time.perf_counter()
        if not snapshot:
            return {}

        lines = ["[profile]"]
        for name, stat in sorted(
            snapshot.items(),
            key=lambda item: (float(item[1].get("total_ms", 0.0)), int(item[1].get("count", 0))),
            reverse=True,
        ):
            total_ms = float(stat.get("total_ms", 0.0))
            if total_ms > 0.0:
                lines.append(
                    "  "
                    f"{name}: total={total_ms:.2f}ms "
                    f"avg={float(stat.get('avg_ms', 0.0)):.3f}ms "
                    f"min={float(stat.get('min_ms', 0.0)):.3f}ms "
                    f"p95={float(stat.get('p95_ms', 0.0)):.3f}ms "
                    f"max={float(stat.get('max_ms', 0.0)):.3f}ms "
                    f"count={int(stat.get('count', 0))}"
                )
            else:
                lines.append(f"  {name}: count={int(stat.get('count', 0))}")
        print("\n".join(lines))
        return snapshot

    @staticmethod
    def _percentile(samples: list[float], percentile: float) -> float:
        if not samples:
            return 0.0
        ordered = sorted(samples)
        if len(ordered) == 1:
            return ordered[0]
        rank = max(0.0, min(1.0, percentile / 100.0)) * (len(ordered) - 1)
        lo = int(rank)
        hi = min(lo + 1, len(ordered) - 1)
        frac = rank - lo
        return ordered[lo] * (1.0 - frac) + ordered[hi] * frac

    def _summarize_stat(self, stat: _Stat) -> dict[str, float | int]:
        avg_ms = stat.total_ms / stat.count if stat.count else 0.0
        p95_ms = self._percentile(stat.samples_ms or [], 95.0)
        return {
            "total_ms": stat.total_ms,
            "avg_ms": avg_ms,
            "min_ms": 0.0 if stat.count == 0 else stat.min_ms,
            "p95_ms": p95_ms,
            "max_ms": stat.max_ms,
            "count": stat.count,
        }


profiler = RuntimeProfiler()
