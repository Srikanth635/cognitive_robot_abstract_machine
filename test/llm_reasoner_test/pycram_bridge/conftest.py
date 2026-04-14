"""Conftest for pycram_bridge tests."""
from __future__ import annotations

import pytest
from llm_reasoner.pycram_bridge.introspector import PycramIntrospector


@pytest.fixture
def introspector() -> PycramIntrospector:
    """Return a fresh PycramIntrospector instance."""
    return PycramIntrospector()
