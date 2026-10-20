"""
The checked-in pilot remains portable and explicitly unreviewed.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from repository_knowledge.adapters.okf import OkfCodec, OkfDirectory
from repository_knowledge.concepts import Lifecycle
from repository_knowledge.operations.knowledge import MetadataField

# %% pilot interchange


def test_pilot_matches_its_saved_source_selection() -> None:
    """
    Every planned pilot document is present as an unreviewed draft.
    """
    package = Path(__file__).parents[2] / "repository_knowledge"
    specification = tomllib.loads(
        (package / "docs/pilot.toml").read_text(encoding="utf-8")
    )
    bundle = OkfDirectory(package / "wiki").read_bundle()
    assert [concept.relative_path for concept in bundle.concepts] == sorted(
        item["path"] for item in specification["concepts"]
    )
    assert bundle.navigation_issues == {}
    for concept in bundle.concepts:
        assert concept.lifecycle == Lifecycle.DRAFT
        assert (MetadataField.VERIFIED in concept.metadata) is False


def test_pilot_concepts_round_trip_with_evidence() -> None:
    """
    Supported metadata, source bindings and body links survive exchange.
    """
    package = Path(__file__).parents[2] / "repository_knowledge"
    bundle = OkfDirectory(package / "wiki").read_bundle()
    codec = OkfCodec()
    for concept in bundle.concepts:
        assert codec.decode(concept.relative_path, codec.encode(concept)) == concept
