#!/usr/bin/env python3
"""Print Rank-1 / mAP JSON for one synthetic ReID evaluation.

Samples come from the in-memory synthetic set. They are
loaded with the ReID loader in the evaluation context before any
embedding. The manifest is ``licenses/manifest.json`` in this repo.
There is no data directory and no file input. Only the placeholder
embedder is available. The JSON evidence flag is false. The numbers
are not an AC-06 result.
"""

from __future__ import annotations

import sys

from trace_search.reid.eval import main

if __name__ == "__main__":
    sys.exit(main())
