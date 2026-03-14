"""Beatmap scanner -- discovers and indexes mapsets under a root directory."""

from __future__ import annotations

import hashlib
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from osupyparser import OsuFile

from profiling import profiler


@dataclass
class BeatmapInfo:
    path: str
    beatmap_md5: str
    directory: str
    title: str
    title_unicode: str
    artist: str
    artist_unicode: str
    creator: str
    version: str
    preview_time: int
    background_file: str
    audio_filename: str
    audio_path: str
    cs: float
    ar: float
    od: float
    hp: float


@dataclass
class BeatmapSet:
    directory: str
    maps: list[BeatmapInfo] = field(default_factory=list)

    @property
    def display_title(self) -> str:
        m = self.maps[0] if self.maps else None
        if m is None:
            return "Unknown"
        return m.title_unicode or m.title or "Unknown"

    @property
    def display_artist(self) -> str:
        m = self.maps[0] if self.maps else None
        if m is None:
            return ""
        return m.artist_unicode or m.artist or ""

    @property
    def creator(self) -> str:
        return self.maps[0].creator if self.maps else ""

    @property
    def preview_time(self) -> int:
        return self.maps[0].preview_time if self.maps else 0

    @property
    def background_path(self) -> str | None:
        if not self.maps:
            return None
        m = self.maps[0]
        if not m.background_file:
            return None
        p = Path(m.directory) / m.background_file
        return str(p) if p.is_file() else None

    @property
    def audio_path(self) -> str | None:
        if not self.maps:
            return None
        return self.maps[0].audio_path


class BeatmapScanner:
    """Scans a directory tree for osu! mapsets."""

    def __init__(self, maps_root: str, scan_immediately: bool = True):
        self._root = Path(maps_root)
        self._sets: list[BeatmapSet] = []
        self._lock = threading.Lock()
        self._scan_complete: bool = False
        if scan_immediately:
            self.scan()

    @property
    def sets(self) -> list[BeatmapSet]:
        with self._lock:
            return self._sets

    @property
    def scan_complete(self) -> bool:
        with self._lock:
            return self._scan_complete

    def scan(self) -> None:
        new_sets: list[BeatmapSet] = []
        if self._root.is_dir():
            subdirs = [sub for sub in sorted(self._root.iterdir()) if sub.is_dir()]
            max_workers = max(1, min(len(subdirs), int(os.environ.get("OSU_SCAN_WORKERS", os.cpu_count() or 4))))
            with profiler.timer("scanner.scan"):
                if max_workers <= 1:
                    for sub in subdirs:
                        bset = self._scan_mapset(sub)
                        if bset is not None:
                            new_sets.append(bset)
                else:
                    ordered_results: dict[int, BeatmapSet] = {}
                    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="beatmap-scan") as executor:
                        futures = {
                            executor.submit(self._scan_mapset, sub): idx
                            for idx, sub in enumerate(subdirs)
                        }
                        for future in as_completed(futures):
                            idx = futures[future]
                            bset = future.result()
                            if bset is not None:
                                ordered_results[idx] = bset
                    for idx in sorted(ordered_results):
                        new_sets.append(ordered_results[idx])

        with self._lock:
            self._sets = new_sets
            self._scan_complete = True

    def _scan_mapset(self, sub: Path) -> BeatmapSet | None:
        osu_files = sorted(sub.glob("*.osu"))
        if not osu_files:
            return None

        bset = BeatmapSet(directory=str(sub))
        for osu_path in osu_files:
            info = self._parse_meta(osu_path, sub)
            if info is not None:
                bset.maps.append(info)
        return bset if bset.maps else None

    @staticmethod
    def _parse_meta(osu_path: Path, directory: Path) -> BeatmapInfo | None:
        try:
            osu = OsuFile(str(osu_path))
            osu.parse_file()
        except Exception:
            return None

        audio_path = str(directory / osu.audio_filename) if osu.audio_filename else ""

        return BeatmapInfo(
            path=str(osu_path),
            beatmap_md5=hashlib.md5(osu_path.read_bytes()).hexdigest(),
            directory=str(directory),
            title=osu.title or "",
            title_unicode=getattr(osu, "title_unicode", "") or "",
            artist=osu.artist or "",
            artist_unicode=getattr(osu, "artist_unicode", "") or "",
            creator=getattr(osu, "creator", "") or "",
            version=getattr(osu, "version", "") or "",
            preview_time=getattr(osu, "preview_time", 0) or 0,
            background_file=getattr(osu, "background_file", "") or "",
            audio_filename=osu.audio_filename or "",
            audio_path=audio_path,
            cs=osu.cs,
            ar=osu.ar,
            od=osu.od,
            hp=osu.hp,
        )
