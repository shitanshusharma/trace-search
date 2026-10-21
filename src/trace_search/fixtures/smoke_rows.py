"""Which manifest rows are generated smoke clips.

Project-rules skips the on-disk file check for these rows. The regenerate
test hashes the same rows, so a new matching row is checked with the others.
"""

import re

_SMOKE_MP4 = re.compile(r"smoke/[^/]+\.mp4\Z")


def is_generated_smoke_clip(entry: object) -> bool:
    """True for a SMOKE_ONLY synthetic clip stored at smoke/*.mp4."""
    if not isinstance(entry, dict):
        return False
    path = entry.get("path")
    return (
        entry.get("type") == "synthetic_clip"
        and entry.get("use") == "SMOKE_ONLY"
        and isinstance(path, str)
        and _SMOKE_MP4.fullmatch(path) is not None
    )
