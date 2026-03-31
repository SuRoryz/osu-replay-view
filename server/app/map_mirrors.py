from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .config import Settings
from .osu_client import OsuBinaryResponse


class MirrorResolverError(RuntimeError):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = int(status_code)
        self.detail = detail


@dataclass(slots=True, frozen=True)
class MirrorProvider:
    name: str
    download_url_template: str

    def download_url(self, beatmapset_id: int) -> str:
        return self.download_url_template.format(beatmapset_id=int(beatmapset_id))


class MirrorResolver:
    def __init__(self, settings: Settings) -> None:
        self._timeout_seconds = float(settings.map_mirror_timeout_seconds)
        self._providers = tuple(self._build_providers(settings.map_mirror_download_templates))

    @property
    def providers(self) -> tuple[MirrorProvider, ...]:
        return self._providers

    def ensure_configured(self) -> None:
        if self._providers:
            return
        raise MirrorResolverError(
            503,
            "No beatmap mirror providers are configured on the server. Set OSU_MIRROR_DOWNLOAD_TEMPLATES.",
        )

    def download_beatmapset(self, beatmapset_id: int) -> OsuBinaryResponse:
        self.ensure_configured()
        failures: list[str] = []
        for provider in self._providers:
            try:
                return self._download_from_provider(provider, beatmapset_id)
            except MirrorResolverError as exc:
                failures.append(f"{provider.name}: {exc.detail}")
        raise MirrorResolverError(
            502,
            "Beatmap download failed from all configured mirrors. " + " | ".join(failures),
        )

    def _build_providers(self, templates: Iterable[str]) -> list[MirrorProvider]:
        providers: list[MirrorProvider] = []
        seen: set[str] = set()
        for index, template in enumerate(templates, start=1):
            value = str(template or "").strip()
            if not value:
                continue
            provider = MirrorProvider(
                name=self._provider_name(value, index),
                download_url_template=value,
            )
            if provider.download_url_template in seen:
                continue
            seen.add(provider.download_url_template)
            providers.append(provider)
        return providers

    def _provider_name(self, template: str, index: int) -> str:
        host = urlparse(template).netloc.strip().lower()
        if host:
            return host
        return f"mirror-{index}"

    def _download_from_provider(self, provider: MirrorProvider, beatmapset_id: int) -> OsuBinaryResponse:
        url = provider.download_url(beatmapset_id)
        request = Request(
            url,
            method="GET",
            headers={
                "Accept": "application/octet-stream,application/zip,application/x-osu-beatmap-archive,*/*",
                "User-Agent": "osu_replay_v2/1.0",
                "Referer": "https://osu.ppy.sh/",
            },
        )
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                payload = response.read()
                content_type = str(response.headers.get_content_type() or "application/octet-stream")
                filename = response.headers.get_filename() or f"beatmapset-{int(beatmapset_id)}.osz"
        except HTTPError as exc:
            message = f"HTTP {exc.code}"
            if exc.code == 404:
                message = "mapset not found"
            elif exc.code == 429:
                message = "rate limited"
            raise MirrorResolverError(exc.code, message) from exc
        except URLError as exc:
            raise MirrorResolverError(502, f"unreachable ({exc.reason})") from exc

        if not payload:
            raise MirrorResolverError(502, "returned an empty response")
        if content_type.startswith("text/html"):
            raise MirrorResolverError(502, "returned HTML instead of an .osz archive")
        if not payload.startswith(b"PK"):
            raise MirrorResolverError(502, "returned a non-zip payload")
        return OsuBinaryResponse(
            payload=payload,
            content_type=content_type,
            filename=filename,
        )
