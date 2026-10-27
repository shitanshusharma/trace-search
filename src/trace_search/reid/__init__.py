"""Untrained placeholder ReID (code-only mode, PRD §0). No training on footage.

The embedders are ``UntrainedPlaceholderEmbedder`` (T-R-3) and
``OsnetLiteEmbedder`` (T-R-9).
"""

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
