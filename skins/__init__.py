from skins.base import Skin
from skins.default import DefaultSkin

SKIN_REGISTRY: list[Skin] = [DefaultSkin()]

__all__ = ["Skin", "DefaultSkin", "SKIN_REGISTRY"]
