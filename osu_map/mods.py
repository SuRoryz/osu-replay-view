"""osu! mod bitmask constants and difficulty adjustment helpers."""

from __future__ import annotations

# Standard osu! mod bitmask values (matching .osr format)
NF = 1        # NoFail
EZ = 2        # Easy
HD = 8        # Hidden
HR = 16       # HardRock
SD = 32       # SuddenDeath
DT = 64       # DoubleTime
HT = 256      # HalfTime
NC = 512      # Nightcore (includes DT)
FL = 1024     # Flashlight
PF = 16384    # Perfect

_SUPPORTED_MODS = NF | EZ | HD | HR | DT | HT | FL

_INCOMPATIBLE_PAIRS = [(EZ, HR), (DT, HT)]
_MOD_SCORE_MULTIPLIERS: dict[int, float] = {
    NF: 0.50,
    EZ: 0.50,
    HD: 1.06,
    HR: 1.06,
    DT: 1.12,
    HT: 0.30,
    FL: 1.12,
}

MOD_FLAG_MAP: dict[str, int] = {
    "Easy": EZ,
    "NoFail": NF,
    "HalfTime": HT,
    "HardRock": HR,
    "Hidden": HD,
    "DoubleTime": DT,
    "Flashlight": FL,
}

MOD_SHORT: dict[int, str] = {
    NF: "NF", EZ: "EZ", HD: "HD", HR: "HR", DT: "DT", HT: "HT", FL: "FL",
}


def incompatible_with(flag: int) -> int:
    """Return the bitmask of all mods that are incompatible with *flag*."""
    flag = normalize_mods(flag)
    mask = 0
    for a, b in _INCOMPATIBLE_PAIRS:
        if flag & a:
            mask |= b
        if flag & b:
            mask |= a
    return mask


def normalize_mods(mods: int) -> int:
    """Collapse aliases, strip unsupported bits, and remove incompatible pairs."""
    normalized = int(mods)
    if normalized & NC:
        normalized |= DT
    if normalized & PF:
        normalized |= SD
    normalized &= _SUPPORTED_MODS
    for a, b in _INCOMPATIBLE_PAIRS:
        if normalized & a and normalized & b:
            normalized &= ~b
    return normalized


def apply_difficulty(ar: float, cs: float, od: float, hp: float,
                     mods: int) -> tuple[float, float, float, float]:
    """Return (ar, cs, od, hp) adjusted for EZ / HR mod multipliers."""
    if mods & EZ:
        ar *= 0.5
        cs *= 0.5
        od *= 0.5
        hp *= 0.5
    if mods & HR:
        ar = min(10.0, ar * 1.4)
        cs = min(10.0, cs * 1.3)
        od = min(10.0, od * 1.4)
        hp = min(10.0, hp * 1.4)
    return ar, cs, od, hp


def speed_multiplier(mods: int) -> float:
    """Return the speed multiplier for DT / HT (or 1.0 for no speed mod)."""
    if mods & DT:
        return 1.5
    if mods & HT:
        return 0.75
    return 1.0


def score_multiplier(mods: int) -> float:
    """Return the stable score multiplier for the supported mod set."""
    mult = 1.0
    for flag, value in _MOD_SCORE_MULTIPLIERS.items():
        if mods & flag:
            mult *= value
    return mult


def mod_string(mods: int) -> str:
    """Human-readable short string like '+HRHD' for a mods bitmask."""
    mods = normalize_mods(mods)
    parts = [short for flag, short in sorted(MOD_SHORT.items()) if mods & flag]
    return ("+" + "".join(parts)) if parts else ""
