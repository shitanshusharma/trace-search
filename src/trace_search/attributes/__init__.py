"""
Person attribute extraction module (T4).

Deterministic color classification and bag detection.
"""

from .ac08_harness import (
    AC08Harness,
    AttributeSearchQuery,
    LabelledRecord,
)
from .attributes import (
    AttributeExtractor,
    AttributeQuery,
    Color,
    PersonAttributes,
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
