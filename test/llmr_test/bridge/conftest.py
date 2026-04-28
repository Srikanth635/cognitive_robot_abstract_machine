"""Shared fixtures for the bridge test package."""

from __future__ import annotations

import pytest

from llmr.bridge.introspect import ActionFieldIntrospector


@pytest.fixture
def introspector() -> ActionFieldIntrospector:
    """Return a fresh :class:`ActionFieldIntrospector` for each test."""
    return ActionFieldIntrospector()
