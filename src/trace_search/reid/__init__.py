"""Person re-identification (ReID) with OSNet trained on consented footage."""

from .data import ReidSample, load_reid_samples
from .osnet_lite import OsnetLiteEmbedder
from .placeholder import UntrainedPlaceholderEmbedder
from .types import AppearanceEmbedder, AppearanceEmbedding

__all__ = [
    "AppearanceEmbedder",
    "AppearanceEmbedding",
    "OsnetLiteEmbedder",
    "ReidSample",
    "UntrainedPlaceholderEmbedder",
    "load_reid_samples",
]
