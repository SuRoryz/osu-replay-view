"""Build-version helpers for dev and packaged runs."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from runtime_paths import APP_ROOT


REPO_ROOT = Path(__file__).resolve().parent
VERSION_STATE_NAME = "version_state.json"
BUILD_METADATA_NAME = "build_metadata.json"
DEFAULT_VERSION_STATE = {"major": 0, "minor": 2, "build": 0}


@dataclass(slots=True)
class VersionState:
    major: int
    minor: int
    build: int


@dataclass(slots=True)
class BuildMetadata:
    major: int
    minor: int
    build: int
    channel: str
    branch: str

    @property
    def version(self) -> str:
        return format_version(
            major=self.major,
            minor=self.minor,
            build=self.build,
            channel=self.channel,
            branch=self.branch,
        )


def _coerce_int(value: object, default: int) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def sanitize_branch_name(value: str) -> str:
    branch = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    return branch.strip("._-") or "local"


def version_state_path(root: Path | None = None) -> Path:
    return (root or REPO_ROOT) / VERSION_STATE_NAME


def build_metadata_path(root: Path | None = None) -> Path:
    return (root or REPO_ROOT) / BUILD_METADATA_NAME


def load_version_state(path: Path | None = None) -> VersionState:
    target = path or version_state_path()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, TypeError):
        payload = DEFAULT_VERSION_STATE
    return VersionState(
        major=_coerce_int(payload.get("major"), DEFAULT_VERSION_STATE["major"]),
        minor=_coerce_int(payload.get("minor"), DEFAULT_VERSION_STATE["minor"]),
        build=_coerce_int(payload.get("build"), DEFAULT_VERSION_STATE["build"]),
    )


def save_version_state(state: VersionState, path: Path | None = None) -> Path:
    target = path or version_state_path()
    target.write_text(json.dumps(asdict(state), indent=2) + "\n", encoding="utf-8")
    return target


def load_build_metadata(path: Path | None = None) -> BuildMetadata | None:
    target = path or build_metadata_path()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return None
    return BuildMetadata(
        major=_coerce_int(payload.get("major"), DEFAULT_VERSION_STATE["major"]),
        minor=_coerce_int(payload.get("minor"), DEFAULT_VERSION_STATE["minor"]),
        build=_coerce_int(payload.get("build"), DEFAULT_VERSION_STATE["build"]),
        channel=str(payload.get("channel") or "PROD").strip().upper() or "PROD",
        branch=sanitize_branch_name(str(payload.get("branch") or "local")),
    )


def save_build_metadata(metadata: BuildMetadata, path: Path | None = None) -> Path:
    target = path or build_metadata_path()
    target.write_text(json.dumps(asdict(metadata), indent=2) + "\n", encoding="utf-8")
    return target


def format_version(*, major: int, minor: int, build: int, channel: str, branch: str) -> str:
    return f"{major}.{minor}.{build}-{channel.strip().upper()}-{sanitize_branch_name(branch)}"


def resolve_branch_name(repo_root: Path | None = None) -> str:
    for key in ("OSU_REPLAY_BRANCH_NAME", "GITHUB_REF_NAME", "CI_COMMIT_REF_NAME", "BRANCH_NAME"):
        value = str(os.getenv(key) or "").strip()
        if value:
            return sanitize_branch_name(value)

    root = repo_root or REPO_ROOT
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--abbrev-ref", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return "local"

    branch = result.stdout.strip()
    if not branch or branch == "HEAD":
        return "local"
    return sanitize_branch_name(branch)


def get_display_version(
    *,
    frozen: bool | None = None,
    repo_root: Path | None = None,
    app_root: Path | None = None,
) -> str:
    is_frozen = getattr(sys, "frozen", False) if frozen is None else frozen
    runtime_root = app_root or APP_ROOT

    if is_frozen:
        metadata = load_build_metadata(runtime_root / BUILD_METADATA_NAME)
        if metadata is not None:
            return metadata.version

    state_root = runtime_root if is_frozen else (repo_root or REPO_ROOT)
    state = load_version_state(state_root / VERSION_STATE_NAME)
    channel = "PROD" if is_frozen else "DEV"
    build = state.build if is_frozen else state.build + 1
    branch = resolve_branch_name(repo_root=repo_root)
    return format_version(
        major=state.major,
        minor=state.minor,
        build=build,
        channel=channel,
        branch=branch,
    )


def prepare_build(repo_root: Path | None = None) -> BuildMetadata:
    root = repo_root or REPO_ROOT
    state = load_version_state(version_state_path(root))
    metadata = BuildMetadata(
        major=state.major,
        minor=state.minor,
        build=state.build + 1,
        channel="PROD",
        branch=resolve_branch_name(repo_root=root),
    )
    save_build_metadata(metadata, build_metadata_path(root))
    return metadata


def finalize_build(repo_root: Path | None = None) -> BuildMetadata:
    root = repo_root or REPO_ROOT
    metadata = load_build_metadata(build_metadata_path(root))
    if metadata is None:
        raise FileNotFoundError("Build metadata missing. Run prepare-build first.")
    save_version_state(
        VersionState(major=metadata.major, minor=metadata.minor, build=metadata.build),
        version_state_path(root),
    )
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage osu_replay build versions.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("prepare-build", help="Write metadata for the next production build.")
    subparsers.add_parser("finalize-build", help="Persist the prepared production build number.")
    subparsers.add_parser("show-dev", help="Print the current dev version string.")
    subparsers.add_parser("show-prod", help="Print the current prepared or fallback prod version string.")

    args = parser.parse_args(argv)

    if args.command == "prepare-build":
        print(prepare_build().version)
        return 0
    if args.command == "finalize-build":
        print(finalize_build().version)
        return 0
    if args.command == "show-dev":
        print(get_display_version(frozen=False))
        return 0
    if args.command == "show-prod":
        print(get_display_version(frozen=True))
        return 0
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
