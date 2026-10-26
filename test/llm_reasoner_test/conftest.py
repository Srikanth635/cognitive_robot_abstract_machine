"""
Fixtures shared by the knowledge component's tests.
"""

from __future__ import annotations

import pytest

from llm_reasoner.knowledge.admission.hermit import HermitCheck

# %% reasoning


@pytest.fixture(scope="module")
def hermit() -> HermitCheck:
    """
    HermiT with nothing in its background, or a skip when it is not installed.
    """
    if HermitCheck.shipped_directory() is None:
        pytest.skip("needs owlready2's HermiT")
    return HermitCheck(background_files=[], java_memory_megabytes=6000)
