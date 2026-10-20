"""
Person attribute extraction module (T4).

Deterministic color classification and bag detection.
"""

from .attributes import (
    AttributeExtractor,
    PersonAttributes,
    Color,
    AttributeQuery,
)
from .ac08_harness import (
    AC08Harness,
    LabelledRecord,
    AttributeSearchQuery,
)

__all__ = [
    "AttributeExtractor",
    "PersonAttributes",
    "Color",
    "AttributeQuery",
    "AC08Harness",
    "LabelledRecord",
    "AttributeSearchQuery",
]
