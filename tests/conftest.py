import re
from collections.abc import Mapping, Sequence

import pytest


def _normalize_for_snapshot(value):
    if isinstance(value, re.Match):
        return repr(value)
    if isinstance(value, Mapping):
        return {key: _normalize_for_snapshot(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_normalize_for_snapshot(item) for item in value)
    if isinstance(value, list):
        return [_normalize_for_snapshot(item) for item in value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return type(value)(_normalize_for_snapshot(item) for item in value)
    if isinstance(value, (str, bytes, bytearray, int, float, bool, type(None))):
        return value
    return repr(value)


@pytest.fixture
def normalize_for_snapshot():
    return _normalize_for_snapshot
