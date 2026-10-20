"""Basic tests to ensure the package imports correctly."""

import trace_search


def test_version():
    """Package should have a version."""
    assert hasattr(trace_search, '__version__')
    assert trace_search.__version__ == "0.1.0"


def test_submodules_exist():
    """All planned submodules should exist (even if empty for now)."""
    from trace_search import (
        api,
        attributes,
        common,
        detection,
        ingest,
        matching,
        reid,
        tracking,
        ui,
    )

    # Just check they import without error
    assert ingest is not None
    assert detection is not None
    assert tracking is not None
    assert attributes is not None
    assert reid is not None
    assert matching is not None
    assert api is not None
    assert ui is not None
    assert common is not None
