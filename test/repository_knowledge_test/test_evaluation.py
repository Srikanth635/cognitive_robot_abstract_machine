"""
Retrieval comparisons use independent committed-source expectations.
"""

from __future__ import annotations

import ast
import csv
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from repository_knowledge.adapters.git_source import GitSource
from repository_knowledge.adapters.pytest_source import PytestSource
from repository_knowledge.adapters.python_source import PythonSource
from repository_knowledge.adapters.sqlite_index import SQLiteIndex
from repository_knowledge.assistance import ContextRole
from repository_knowledge.concepts import KnowledgeBundle
from repository_knowledge.exceptions import SourceRevisionMismatch
from repository_knowledge.interfaces.evaluation import (
    CitationAudit,
    NavigationTask,
    RetrievalEvaluation,
    RetrievalMethod,
)
from repository_knowledge.model import IndexData
from repository_knowledge.operations.assistance import TaskAssistance
from repository_knowledge.operations.indexing import IndexBuilder

from .support import CommittedProject, FixturePath, FixtureSymbol

# %% comparable retrieval


@pytest.mark.parametrize("method", list(RetrievalMethod))
def test_each_retrieval_method_finds_the_same_committed_definition(
    committed_project: CommittedProject,
    source_index: IndexData,
    indexed_storage: SQLiteIndex,
    method: RetrievalMethod,
    tmp_path: Path,
) -> None:
    """
    All three interfaces locate the independently selected source declaration.
    """
    symbol = next(
        item
        for item in source_index.symbols
        if item.symbol_identifier == FixtureSymbol.BASE
    )
    task = NavigationTask(
        symbol.symbol_identifier,
        symbol.location.relative_path,
        symbol.location.start_line,
    )
    bundle = tmp_path / "empty-wiki"
    bundle.mkdir()
    evaluator = RetrievalEvaluation(
        committed_project.root,
        indexed_storage.database_path,
        committed_project.commit_identifier,
        bundle,
    )
    observation = evaluator.measure(task, method)
    assert observation.definition_found is True
    assert observation.required_paths_found == observation.required_paths_total == 1
    assert observation.valid_citations == observation.total_citations
    assert observation.exit_status == 0
    assert observation.agent_tool_calls == 1
    assert observation.agent_tokens is None
    assert observation.generated_task_success is None


@pytest.mark.parametrize("method", [RetrievalMethod.SOURCE, RetrievalMethod.STRUCTURE])
def test_absent_declaration_is_reported_without_a_false_match(
    committed_project: CommittedProject,
    indexed_storage: SQLiteIndex,
    method: RetrievalMethod,
) -> None:
    """
    A deliberately unknown identity cannot earn a positive navigation result.
    """
    evaluator = RetrievalEvaluation(
        committed_project.root,
        indexed_storage.database_path,
        committed_project.commit_identifier,
    )
    observation = evaluator.measure(NavigationTask("first.base.Absent", "", 0), method)
    assert observation.definition_found is False
    assert observation.returned_locations == 0
    assert observation.total_citations == 0


# %% evidence verification


def test_citation_audit_rejects_an_incorrect_file_digest(
    committed_project: CommittedProject, source_index: IndexData
) -> None:
    """
    A plausible path and line cannot compensate for a changed digest.
    """
    context = TaskAssistance(
        source_index, PytestSource().discover(source_index)
    ).prepare_context(FixtureSymbol.BASE, KnowledgeBundle([]))
    declaration = next(
        item for item in context.items if item.role == ContextRole.DECLARATION
    )
    context = replace(
        context,
        items=[
            replace(
                declaration,
                citations=[
                    replace(declaration.citations[0], content_digest="incorrect")
                ],
            )
        ],
    )
    audit = CitationAudit(
        GitSource(committed_project.root).read_snapshot(
            committed_project.commit_identifier
        )
    )
    evidence = audit.check_context(context)
    assert evidence.valid_citations == 0
    assert evidence.total_citations == 1


def test_evaluator_refuses_a_different_index_revision(
    committed_project: CommittedProject, indexed_storage: SQLiteIndex
) -> None:
    """
    A moved source revision fails before timing or scoring retrieval.
    """
    committed_project.git(
        "commit", "--quiet", "--allow-empty", "-m", "Advance evaluation source"
    )
    with pytest.raises(SourceRevisionMismatch):
        RetrievalEvaluation(
            committed_project.root,
            indexed_storage.database_path,
            committed_project.git("rev-parse", "HEAD"),
        )


def test_source_search_audit_rejects_changed_line_text(
    committed_project: CommittedProject, source_index: IndexData
) -> None:
    """
    A correct source coordinate cannot authenticate different returned text.
    """
    symbol = next(
        item
        for item in source_index.symbols
        if item.symbol_identifier == FixtureSymbol.BASE
    )
    audit = CitationAudit(
        GitSource(committed_project.root).read_snapshot(
            committed_project.commit_identifier
        )
    )
    response = f"{committed_project.commit_identifier}:{symbol.location.relative_path}:{symbol.location.start_line}:altered"
    evidence = audit.check_search(response)
    assert evidence.valid_citations == 0
    assert evidence.total_citations == 1


# %% mutations and executable reports


@pytest.mark.parametrize("method", list(RetrievalMethod))
def test_retrieval_after_rebuild_finds_a_replaced_declaration(
    committed_project: CommittedProject,
    indexed_storage: SQLiteIndex,
    method: RetrievalMethod,
) -> None:
    """
    All retrieval methods follow a committed replacement after an explicit rebuild.
    """
    replacement = Path(__file__).parent / FixturePath.REPLACEMENT
    shutil.copyfile(replacement, committed_project.root / FixturePath.API)
    committed_project.git("add", ".")
    committed_project.git("commit", "--quiet", "-m", "Replace evaluation declaration")
    revision = committed_project.git("rev-parse", "HEAD")
    indexed_storage.replace(
        IndexBuilder(GitSource(committed_project.root), PythonSource()).build(revision)
    )
    expected = next(
        node
        for node in ast.parse(replacement.read_text(encoding="utf-8")).body
        if isinstance(node, ast.ClassDef)
    )
    observation = RetrievalEvaluation(
        committed_project.root, indexed_storage.database_path, revision
    ).measure(
        NavigationTask(FixtureSymbol.REPLACEMENT, FixturePath.API, expected.lineno),
        method,
    )
    assert observation.definition_found is True
    assert observation.source_revision == revision
    assert observation.valid_citations == observation.total_citations


def test_evaluation_command_saves_all_methods_without_invented_model_metrics(
    committed_project: CommittedProject, indexed_storage: SQLiteIndex, tmp_path: Path
) -> None:
    """
    The executable report preserves each method and leaves model results unknown.
    """
    output = tmp_path / "measurements.csv"
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "repository_knowledge.interfaces.evaluation",
            "--repository",
            str(committed_project.root),
            "--database",
            str(indexed_storage.database_path),
            "--tasks",
            str(Path(__file__).parent / "dataset/evaluation.toml"),
            "--output",
            str(output),
        ],
        capture_output=True,
        timeout=30,
    )
    assert process.returncode == 0
    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["method"] for row in rows] == list(RetrievalMethod)
    assert [row["agent_tokens"] for row in rows] == [""] * len(RetrievalMethod)
    assert [row["generated_task_success"] for row in rows] == [""] * len(
        RetrievalMethod
    )


def test_evaluation_rejects_same_name_hits_for_an_expected_absent_symbol(
    committed_project: CommittedProject, indexed_storage: SQLiteIndex, tmp_path: Path
) -> None:
    """
    Unqualified search matches cannot establish an absent qualified declaration.
    """
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "repository_knowledge.interfaces.evaluation",
            "--repository",
            str(committed_project.root),
            "--database",
            str(indexed_storage.database_path),
            "--tasks",
            str(Path(__file__).parent / "dataset/absent_evaluation.toml"),
            "--output",
            str(tmp_path / "absent.csv"),
        ],
        capture_output=True,
        timeout=30,
    )
    assert process.returncode == 1


def test_evaluation_fails_when_independently_required_source_is_missing(
    committed_project: CommittedProject, indexed_storage: SQLiteIndex, tmp_path: Path
) -> None:
    """
    A correct declaration does not satisfy a broader evidence requirement.
    """
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "repository_knowledge.interfaces.evaluation",
            "--repository",
            str(committed_project.root),
            "--database",
            str(indexed_storage.database_path),
            "--tasks",
            str(Path(__file__).parent / "dataset/missing_evidence.toml"),
            "--output",
            str(tmp_path / "missing-evidence.csv"),
        ],
        capture_output=True,
        timeout=30,
    )
    assert process.returncode == 1
