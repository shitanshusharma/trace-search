"""Person re-identification (ReID) with OSNet trained on consented footage."""

from .placeholder import UntrainedPlaceholderEmbedder
from .types import AppearanceEmbedder, AppearanceEmbedding

__all__ = [
    "AppearanceEmbedder",
    "AppearanceEmbedding",
    "UntrainedPlaceholderEmbedder",
]
