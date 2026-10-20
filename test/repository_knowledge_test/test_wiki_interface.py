"""
The wiki interface shares source freshness and bounded application queries.
"""

from __future__ import annotations


from repository_knowledge.adapters.okf import OkfDirectory
from repository_knowledge.concepts import KnowledgeBundle, Concept
from repository_knowledge.interfaces.wiki import (
    WikiParser,
    WikiCommand,
    WikiApplication,
)
from repository_knowledge.model import IndexData, IndexSelection, SelectionKind
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from repository_knowledge.adapters.sqlite_index import SQLiteIndex
    from pathlib import Path

# %% source evidence selection


def test_knowledge_selection_omits_relationships_and_analysis(
    indexed_storage: SQLiteIndex, source_index: IndexData
) -> None:
    """
    Wiki validation needs sources and declarations without the complete graph.
    """
    selected = indexed_storage.read(IndexSelection(SelectionKind.KNOWLEDGE))
    assert selected.sources == source_index.sources
    assert selected.symbols == source_index.symbols
    assert selected.references == []
    assert selected.analyses == []


def test_wiki_parser_selects_find_request(tmp_path: Path) -> None:
    """
    CLI discovery has explicit repository, bundle and page boundaries.
    """
    arguments = WikiParser().parse(
        ["--repository", str(tmp_path), "find", "base", "--limit", "1"]
    )
    assert arguments.command == WikiCommand.FIND
    assert arguments.query == "base"
    assert arguments.limit == 1
    assert arguments.bundle_path == tmp_path / "repository_knowledge/wiki"


def test_wiki_interface_can_export_portable_bundle(
    portable_concept: Concept, tmp_path: Path, source_index: IndexData
) -> None:
    """
    An explicit export creates a new proposal directory with preserved metadata.
    """
    bundle = KnowledgeBundle([portable_concept])
    destination = tmp_path / "export"
    arguments = WikiParser().parse(
        ["--repository", str(tmp_path), "export", str(destination)]
    )
    exported = WikiApplication().execute(arguments, bundle, source_index)
    OkfDirectory(destination).write_bundle(exported)
    assert OkfDirectory(destination).read_bundle() == bundle


def test_wiki_move_proposes_a_tombstone(
    evidenced_concept: Concept, tmp_path: Path, source_index: IndexData
) -> None:
    """
    A move request produces a separate bundle retaining the original identity.
    """
    arguments = WikiParser().parse(
        [
            "--repository",
            str(tmp_path),
            "move",
            evidenced_concept.relative_path,
            "packages/replacement.md",
            str(tmp_path / "proposal"),
        ]
    )
    proposal = WikiApplication().execute(
        arguments, KnowledgeBundle([evidenced_concept]), source_index
    )
    assert (
        proposal.concepts[0].provenance.replacement
        == proposal.concepts[1].relative_path
    )
