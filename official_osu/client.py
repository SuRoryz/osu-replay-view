from __future__ import annotations

import queue
import re
import tempfile
import threading
import time
import webbrowser
import zipfile
from pathlib import Path
from typing import Callable

import requests

from official_osu.models import (
    OsuAccount,
    OsuAuthStatus,
    OsuBeatmapLookup,
    OsuBeatmapLookupState,
    OsuBeatmapsetSearchItem,
    OsuMapSearchState,
    OsuOfficialScore,
    OsuOfficialScoreState,
)
from official_osu.storage import OfficialOsuLocalState


def _sanitize_filename(value: str, fallback: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]+', "_", value).strip().strip(".")
    return cleaned or fallback


class OfficialOsuClient:
    def __init__(self, *, base_url: str, player_uuid: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.player_uuid = player_uuid
        self.local_state = OfficialOsuLocalState.load()
        self._queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._request_threads: set[threading.Thread] = set()
        self.auth = OsuAuthStatus()
        self.search = OsuMapSearchState()
        self._lookup_states: dict[str, OsuBeatmapLookupState] = {}
        self._score_states: dict[int, OsuOfficialScoreState] = {}
        self._score_download_paths: dict[int, str] = {}
        self.event_handler: Callable[[str, dict], None] | None = None
        self._next_auth_poll_at = 0.0
        self._auth_poll_deadline = 0.0

    def update(self) -> None:
        self._drain_events()
        if self.auth.login_in_progress and time.time() >= self._next_auth_poll_at:
            self.refresh_auth_status(force=True)
            self._next_auth_poll_at = time.time() + 2.0
            if self._auth_poll_deadline and time.time() > self._auth_poll_deadline:
                self.auth.login_in_progress = False

    def _spawn(self, event_name: str, worker) -> None:
        thread = threading.Thread(target=self._run_worker, args=(event_name, worker), daemon=True)
        self._request_threads.add(thread)
        thread.start()

    def _run_worker(self, event_name: str, worker) -> None:
        try:
            payload = worker()
            self._queue.put((event_name, payload))
        except Exception as exc:  # noqa: BLE001
            self._queue.put(("error", (event_name, str(exc))))

    def _request_json(self, method: str, path: str, *, params=None, timeout: float = 20.0):
        response = requests.request(
            method,
            f"{self.base_url}{path}",
            params=params,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json() if response.content else {}

    def refresh_auth_status(self, *, force: bool = False) -> None:
        if self.auth.loading and not force:
            return
        self.auth.loading = True
        self._spawn(
            "auth_status",
            lambda: self._request_json("GET", "/osu/auth/status", params={"player_uuid": self.player_uuid}),
        )

    def start_login(self) -> None:
        self.auth.error = None
        self.auth.loading = True
        self._spawn(
            "auth_start",
            lambda: self._request_json("POST", "/osu/auth/start", params={"player_uuid": self.player_uuid}),
        )

    def refresh_session(self) -> None:
        self.auth.loading = True
        self._spawn(
            "auth_refresh",
            lambda: self._request_json("POST", "/osu/auth/refresh", params={"player_uuid": self.player_uuid}),
        )

    def logout(self) -> None:
        self.auth.loading = True
        self._spawn(
            "auth_logout",
            lambda: self._request_json("POST", "/osu/auth/logout", params={"player_uuid": self.player_uuid}),
        )

    def search_beatmapsets(
        self,
        query: str,
        *,
        status: str = "",
        mode: str = "osu",
        cursor_string: str = "",
    ) -> OsuMapSearchState:
        self.search.query = query
        self.search.status_filter = status
        self.search.mode_filter = mode
        self.search.cursor_string = cursor_string
        self.search.loading = True
        self.search.error = None
        self.search.requested_at = time.time()
        self._spawn(
            "beatmap_search",
            lambda: self._request_json(
                "GET",
                "/osu/beatmapsets/search",
                params={
                    "player_uuid": self.player_uuid,
                    "query": query,
                    "status": status,
                    "mode": mode,
                    "cursor_string": cursor_string,
                },
            ),
        )
        return self.search

    def lookup_beatmap(self, checksum: str, *, beatmap_id: int | None = None, force: bool = False) -> OsuBeatmapLookupState:
        key = checksum.strip() or (f"id:{int(beatmap_id)}" if beatmap_id is not None else "")
        state = self._lookup_states.setdefault(key, OsuBeatmapLookupState())
        if state.loading or (state.loaded_at > 0.0 and not force):
            return state
        state.loading = True
        state.error = None
        state.requested_at = time.time()
        self._spawn(
            f"lookup:{key}",
            lambda: self._request_json(
                "GET",
                "/osu/beatmaps/lookup",
                params={
                    "player_uuid": self.player_uuid,
                    "checksum": checksum,
                    "beatmap_id": beatmap_id,
                },
            ),
        )
        return state

    def beatmap_lookup_state(self, checksum: str, *, beatmap_id: int | None = None) -> OsuBeatmapLookupState:
        key = checksum.strip() or (f"id:{int(beatmap_id)}" if beatmap_id is not None else "")
        return self._lookup_states.setdefault(key, OsuBeatmapLookupState())

    def fetch_scores(self, beatmap_id: int, *, ruleset: str = "osu", force: bool = False) -> OsuOfficialScoreState:
        state = self._score_states.setdefault(int(beatmap_id), OsuOfficialScoreState())
        if state.loading or (state.loaded_at > 0.0 and not force):
            return state
        state.loading = True
        state.error = None
        state.requested_at = time.time()
        self._spawn(
            f"scores:{int(beatmap_id)}",
            lambda: self._request_json(
                "GET",
                f"/osu/beatmaps/{int(beatmap_id)}/scores",
                params={"player_uuid": self.player_uuid, "ruleset": ruleset},
            ),
        )
        return state

    def score_state(self, beatmap_id: int) -> OsuOfficialScoreState:
        return self._score_states.setdefault(int(beatmap_id), OsuOfficialScoreState())

    @staticmethod
    def _score_download_path_matches_score(local_path: str | None, expected_map_md5: str = "") -> bool:
        if not local_path:
            return False
        path = Path(local_path)
        if not path.is_file():
            return False
        expected_map_md5 = str(expected_map_md5 or "").strip().lower()
        if not expected_map_md5:
            return True
        try:
            from replay.data import ReplayData

            summary = ReplayData.peek_summary(str(path))
        except Exception:
            return False
        return str(summary.map_md5 or "").strip().lower() == expected_map_md5

    def download_score_replay(self, score: OsuOfficialScore, target_dir: str, *, expected_map_md5: str = "") -> None:
        if score.is_downloading:
            return
        existing_path = self._existing_score_download_path(score, target_dir, expected_map_md5=expected_map_md5)
        if existing_path:
            score.local_path = existing_path
            score.is_downloaded = True
            score.is_downloading = False
            score.download_progress = 1.0
            score.status_text = "Downloaded"
            self._score_download_paths[int(score.score_id)] = existing_path
            self._notify("score_download_finished", score_id=score.score_id, score=score, local_path=existing_path, progress=1.0)
            return
        score.is_downloading = True
        score.download_progress = 0.0
        score.status_text = "Downloading..."
        self._notify("score_download_started", score_id=score.score_id, score=score, progress=0.0)

        def worker():
            response = requests.get(
                f"{self.base_url}/osu/scores/{int(score.score_id)}/download",
                params={
                    "player_uuid": self.player_uuid,
                    "ruleset": score.ruleset or "osu",
                    "legacy_score_id": (int(score.legacy_score_id) if score.legacy_score_id else None),
                },
                stream=True,
                timeout=60,
            )
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            raw_name = f"{score.username or 'score'} [{int(score.score_id)}].osr"
            content_disposition = response.headers.get("content-disposition") or ""
            if "filename=" in content_disposition:
                raw_name = content_disposition.split("filename=")[-1].strip("\"'")
            safe_name = _sanitize_filename(raw_name, f"score-{int(score.score_id)}.osr")
            Path(target_dir).mkdir(parents=True, exist_ok=True)
            local_path = str(Path(target_dir) / safe_name)
            downloaded = 0
            with open(local_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=65536):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    downloaded += len(chunk)
                    progress = 0.0 if total <= 0 else min(1.0, downloaded / total)
                    self._queue.put(("score_download_progress", (score.score_id, progress)))
            if not self._score_download_path_matches_score(local_path, expected_map_md5):
                try:
                    Path(local_path).unlink()
                except OSError:
                    pass
                raise ValueError("Downloaded official replay does not match the selected difficulty.")
            return score.score_id, local_path

        self._spawn(f"score_download:{int(score.score_id)}", worker)

    def _existing_score_download_path(
        self,
        score: OsuOfficialScore,
        target_dir: str,
        *,
        expected_map_md5: str = "",
    ) -> str | None:
        if self._score_download_path_matches_score(score.local_path, expected_map_md5):
            return score.local_path
        score.local_path = None
        remembered = self.local_state.downloaded_path_for_score(int(score.score_id))
        if self._score_download_path_matches_score(remembered, expected_map_md5):
            return remembered
        if remembered:
            self.local_state.forget_download(int(score.score_id))
        if score.legacy_score_id:
            remembered_legacy = self.local_state.downloaded_path_for_score(int(score.legacy_score_id))
            if self._score_download_path_matches_score(remembered_legacy, expected_map_md5):
                return remembered_legacy
            if remembered_legacy:
                self.local_state.forget_download(int(score.legacy_score_id))
        known_path = self._score_download_paths.get(int(score.score_id))
        if self._score_download_path_matches_score(known_path, expected_map_md5):
            return known_path
        if known_path:
            self._score_download_paths.pop(int(score.score_id), None)
        root = Path(target_dir)
        if not root.is_dir():
            return None
        candidate_ids = [int(score.score_id)]
        if score.legacy_score_id and int(score.legacy_score_id) not in candidate_ids:
            candidate_ids.append(int(score.legacy_score_id))
        for path in root.glob("*.osr"):
            name = path.name
            for candidate_id in candidate_ids:
                if (
                    name.endswith(f"[{candidate_id}].osr") or name == f"score-{candidate_id}.osr"
                ) and self._score_download_path_matches_score(str(path), expected_map_md5):
                    return str(path)
        return None

    def download_mapset(self, beatmapset_id: int, maps_root: str) -> None:
        item = next((entry for entry in self.search.items if entry.beatmapset_id == beatmapset_id), None)
        if item is None or item.is_downloading:
            return
        item.is_downloading = True
        item.download_progress = 0.0
        item.download_status = "Downloading..."

        def worker():
            response = requests.get(
                f"{self.base_url}/osu/beatmapsets/{int(beatmapset_id)}/download",
                params={"player_uuid": self.player_uuid},
                stream=True,
                timeout=120,
            )
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            content_disposition = response.headers.get("content-disposition") or ""
            raw_name = f"beatmapset-{int(beatmapset_id)}.osz"
            if "filename=" in content_disposition:
                raw_name = content_disposition.split("filename=")[-1].strip("\"'")
            safe_name = _sanitize_filename(raw_name, f"beatmapset-{int(beatmapset_id)}.osz")
            maps_path = Path(maps_root)
            maps_path.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".osz") as temp_file:
                temp_path = Path(temp_file.name)
                downloaded = 0
                for chunk in response.iter_content(chunk_size=65536):
                    if not chunk:
                        continue
                    temp_file.write(chunk)
                    downloaded += len(chunk)
                    progress = 0.0 if total <= 0 else min(1.0, downloaded / total)
                    self._queue.put(("map_download_progress", (beatmapset_id, progress)))
            extract_dir = maps_path / _sanitize_filename(Path(safe_name).stem, f"beatmapset-{int(beatmapset_id)}")
            if extract_dir.exists():
                suffix = 1
                while (maps_path / f"{extract_dir.name}-{suffix}").exists():
                    suffix += 1
                extract_dir = maps_path / f"{extract_dir.name}-{suffix}"
            extract_dir.mkdir(parents=True, exist_ok=True)
            try:
                with zipfile.ZipFile(temp_path, "r") as archive:
                    archive.extractall(extract_dir)
            finally:
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            return beatmapset_id, str(extract_dir)

        self._spawn(f"map_download:{int(beatmapset_id)}", worker)

    def _notify(self, event_type: str, **payload) -> None:
        if self.event_handler is not None:
            self.event_handler(event_type, payload)

    def _drain_events(self) -> None:
        while True:
            try:
                name, payload = self._queue.get_nowait()
            except queue.Empty:
                break
            if name in {"auth_status", "auth_refresh"}:
                self.auth.loading = False
                self._apply_auth_status(payload)
            elif name == "auth_start":
                self.auth.loading = False
                self.auth.login_in_progress = True
                self._next_auth_poll_at = 0.0
                self._auth_poll_deadline = time.time() + 300.0
                auth_url = str((payload or {}).get("auth_url") or "")
                if auth_url:
                    webbrowser.open(auth_url)
                    self._notify("auth_browser_opened", auth_url=auth_url)
            elif name == "auth_logout":
                self.auth.loading = False
                self.auth = OsuAuthStatus(enabled=self.auth.enabled, linked=False)
                self._notify("auth_unlinked")
            elif name == "beatmap_search":
                self.search.loading = False
                self.search.loaded_at = time.time()
                self.search.error = None
                self._apply_search_results(payload)
            elif name.startswith("lookup:"):
                key = name.split(":", 1)[1]
                state = self._lookup_states.setdefault(key, OsuBeatmapLookupState())
                state.loading = False
                state.error = None
                state.loaded_at = time.time()
                state.item = OsuBeatmapLookup(
                    beatmap_id=int(payload.get("beatmap_id") or 0),
                    beatmapset_id=(int(payload["beatmapset_id"]) if payload.get("beatmapset_id") is not None else None),
                    checksum=str(payload.get("checksum") or ""),
                    mode=str(payload.get("mode") or "osu"),
                    title=str(payload.get("title") or ""),
                    artist=str(payload.get("artist") or ""),
                    creator=str(payload.get("creator") or ""),
                    version=str(payload.get("version") or ""),
                )
            elif name.startswith("scores:"):
                beatmap_id = int(name.split(":", 1)[1])
                state = self._score_states.setdefault(beatmap_id, OsuOfficialScoreState())
                state.loading = False
                state.error = None
                state.loaded_at = time.time()
                local_paths = {score.score_id: score.local_path for score in state.items}
                state.items = []
                for row in payload or []:
                    local_path = local_paths.get(int(row.get("score_id") or 0))
                    item = OsuOfficialScore(
                        score_id=int(row.get("score_id") or 0),
                        legacy_score_id=int(row.get("legacy_score_id") or 0),
                        username=str(row.get("username") or ""),
                        avatar_url=str(row.get("avatar_url") or ""),
                        country_code=str(row.get("country_code") or ""),
                        mods=[str(entry) for entry in row.get("mods") or ()],
                        total_score=int(row.get("total_score") or 0),
                        max_combo=int(row.get("max_combo") or 0),
                        accuracy=float(row.get("accuracy") or 0.0),
                        passed=bool(row.get("passed", True)),
                        perfect=bool(row.get("perfect", False)),
                        has_replay=bool(row.get("has_replay", False)),
                        ended_at=str(row.get("ended_at") or ""),
                        rank=str(row.get("rank") or ""),
                        ruleset=str(row.get("ruleset") or "osu"),
                        count_300=int(row.get("count_300") or 0),
                        count_100=int(row.get("count_100") or 0),
                        count_50=int(row.get("count_50") or 0),
                        count_miss=int(row.get("count_miss") or 0),
                        local_path=local_path,
                        is_downloaded=bool(local_path and Path(local_path).is_file()),
                        status_text="Downloaded" if local_path and Path(local_path).is_file() else "",
                    )
                    state.items.append(item)
            elif name == "score_download_progress":
                score_id, progress = payload
                score = self._find_score(score_id)
                if score is not None:
                    score.download_progress = progress
                    score.status_text = f"{int(progress * 100):d}%"
                    self._notify("score_download_progress", score_id=score_id, score=score, progress=progress)
            elif name.startswith("score_download:"):
                score_id, local_path = payload
                score = self._find_score(score_id)
                if score is not None:
                    score.local_path = local_path
                    score.is_downloaded = True
                    score.is_downloading = False
                    score.download_progress = 1.0
                    score.status_text = "Downloaded"
                    self._score_download_paths[int(score_id)] = local_path
                    self.local_state.remember_download(int(score_id), local_path)
                    if score.legacy_score_id:
                        self.local_state.remember_download(int(score.legacy_score_id), local_path)
                    self._notify("score_download_finished", score_id=score_id, score=score, local_path=local_path, progress=1.0)
            elif name == "map_download_progress":
                beatmapset_id, progress = payload
                item = next((entry for entry in self.search.items if entry.beatmapset_id == beatmapset_id), None)
                if item is not None:
                    item.download_progress = progress
                    item.download_status = f"{int(progress * 100):d}%"
                    self._notify("map_download_progress", beatmapset_id=beatmapset_id, item=item, progress=progress)
            elif name.startswith("map_download:"):
                beatmapset_id, extract_dir = payload
                item = next((entry for entry in self.search.items if entry.beatmapset_id == beatmapset_id), None)
                if item is not None:
                    item.is_downloading = False
                    item.download_progress = 1.0
                    item.download_status = "Installed"
                    item.downloaded_dir = extract_dir
                    self._notify("map_download_finished", beatmapset_id=beatmapset_id, item=item, extract_dir=extract_dir)
            elif name == "error":
                event_name = ""
                error_text = str(payload)
                if isinstance(payload, tuple):
                    event_name = str(payload[0])
                    error_text = str(payload[1])
                self._apply_error(event_name, error_text)

    def _apply_auth_status(self, payload: dict) -> None:
        account = None
        account_value = payload.get("account")
        if isinstance(account_value, dict):
            account = OsuAccount(
                user_id=int(account_value.get("user_id") or 0),
                username=str(account_value.get("username") or ""),
                avatar_url=str(account_value.get("avatar_url") or ""),
                country_code=str(account_value.get("country_code") or ""),
                cover_url=str(account_value.get("cover_url") or ""),
            )
        was_linked = self.auth.linked
        previous_username = self.auth.account.username if self.auth.account is not None else ""
        self.auth.enabled = bool(payload.get("enabled"))
        self.auth.linked = bool(payload.get("linked"))
        self.auth.scopes = [str(item) for item in payload.get("scopes") or ()]
        self.auth.account = account
        self.auth.expires_at = float(payload.get("expires_at")) if payload.get("expires_at") is not None else None
        self.auth.error = None
        if self.auth.linked:
            self.auth.login_in_progress = False
            if not was_linked or (account is not None and account.username and account.username != previous_username):
                self._notify("auth_linked", account=account)
        elif was_linked:
            self._notify("auth_unlinked")

    def _apply_search_results(self, payload: dict) -> None:
        current_items = {item.beatmapset_id: item for item in self.search.items}
        next_items: list[OsuBeatmapsetSearchItem] = []
        for row in payload.get("beatmapsets") or ():
            beatmapset_id = int(row.get("beatmapset_id") or 0)
            previous = current_items.get(beatmapset_id)
            item = OsuBeatmapsetSearchItem(
                beatmapset_id=beatmapset_id,
                artist=str(row.get("artist") or ""),
                title=str(row.get("title") or ""),
                creator=str(row.get("creator") or ""),
                status=str(row.get("status") or ""),
                favourite_count=int(row.get("favourite_count") or 0),
                play_count=int(row.get("play_count") or 0),
                source=str(row.get("source") or ""),
                tags=str(row.get("tags") or ""),
                cover_url=str(row.get("cover_url") or ""),
                preview_url=str(row.get("preview_url") or ""),
                beatmaps=list(row.get("beatmaps") or []),
                is_downloading=previous.is_downloading if previous is not None else False,
                download_progress=previous.download_progress if previous is not None else 0.0,
                download_status=previous.download_status if previous is not None else "",
                downloaded_dir=previous.downloaded_dir if previous is not None else None,
            )
            next_items.append(item)
        self.search.cursor_string = str(payload.get("cursor_string") or "")
        self.search.items = next_items

    def _apply_error(self, event_name: str, error_text: str) -> None:
        if event_name in {"auth_status", "auth_start", "auth_refresh", "auth_logout"}:
            self.auth.loading = False
            self.auth.error = error_text
            if event_name == "auth_start":
                self.auth.login_in_progress = False
        elif event_name == "beatmap_search":
            self.search.loading = False
            self.search.error = error_text
        elif event_name.startswith("lookup:"):
            state = self._lookup_states.setdefault(event_name.split(":", 1)[1], OsuBeatmapLookupState())
            state.loading = False
            state.error = error_text
        elif event_name.startswith("scores:"):
            beatmap_id = int(event_name.split(":", 1)[1])
            state = self._score_states.setdefault(beatmap_id, OsuOfficialScoreState())
            state.loading = False
            state.error = error_text
        elif event_name.startswith("score_download:"):
            score_id = int(event_name.split(":", 1)[1])
            score = self._find_score(score_id)
            if score is not None:
                score.is_downloading = False
                score.download_progress = 0.0
                score.status_text = "Download failed"
                self._notify("score_download_failed", score_id=score_id, score=score, error=error_text)
        elif event_name.startswith("map_download:"):
            beatmapset_id = int(event_name.split(":", 1)[1])
            item = next((entry for entry in self.search.items if entry.beatmapset_id == beatmapset_id), None)
            if item is not None:
                item.is_downloading = False
                item.download_progress = 0.0
                item.download_status = "Install failed"
                self._notify("map_download_failed", beatmapset_id=beatmapset_id, item=item, error=error_text)

    def _find_score(self, score_id: int) -> OsuOfficialScore | None:
        for state in self._score_states.values():
            for score in state.items:
                if score.score_id == int(score_id):
                    return score
        return None

    def forget_downloaded_score(self, score: OsuOfficialScore) -> None:
        self._score_download_paths.pop(int(score.score_id), None)
        self.local_state.forget_download(int(score.score_id))
        if score.legacy_score_id:
            self.local_state.forget_download(int(score.legacy_score_id))
