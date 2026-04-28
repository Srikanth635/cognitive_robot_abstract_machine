"""Conftest for reasoning tests — shared ActionFieldIntrospector fixture."""

from __future__ import annotations

import pytest

from llmr.bridge.introspect import ActionFieldIntrospector


@pytest.fixture
def introspector() -> ActionFieldIntrospector:
    """Return a fresh ActionFieldIntrospector instance."""
    return ActionFieldIntrospector()
