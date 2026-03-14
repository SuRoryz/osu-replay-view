from __future__ import annotations


def normalize_channel_name(value: str) -> str:
    name = " ".join(part for part in value.strip().split() if part)
    if not name:
        raise ValueError("Channel name cannot be empty.")
    if name.startswith("#") or name.startswith("@"):
        return name[:128]
    return f"#{name[:127]}"


def dm_channel_name(player_uuid: str, target_player_uuid: str) -> str:
    left, right = sorted((player_uuid, target_player_uuid))
    return f"@dm:{left}:{right}"
