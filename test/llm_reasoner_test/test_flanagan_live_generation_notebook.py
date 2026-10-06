from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import pytest

from llm_reasoner.knowledge.generation import CachingGenerator
from llm_reasoner.knowledge.hypothesis import (
    HypothesisRecord,
    HypothesisStatus,
    Provenance,
    ValidationReport,
    encode,
)
from llm_reasoner.knowledge.microworlds.flanagan.core import ActionPhaseStructure
from llm_reasoner.knowledge.microworlds.registry import ModuleSelection
from llm_reasoner.knowledge.ontological.decorators import ConceptReference

# %% notebook contract


class NotebookCell(StrEnum):
    """
    Identify each executable stage of the live Flanagan example.
    """

    IMPORTS = "imports"
    CONFIGURATION = "configuration"
    TASK = "task"
    SELECTION = "selection"
    SCHEMA = "schema"
    PIPELINE = "pipeline"
    GENERATION = "generation"
    RECORD_SUMMARY = "record-summary"
    PHASE_QUERY = "phase-query"
    CONTROL_QUERY = "control-query"
    RECOVERY_QUERY = "recovery-query"
    TYPE_QUERY = "type-query"
    JSON_VIEW = "json-view"


class RepositoryPath(StrEnum):
    """
    Identify repository files exercised by this test.
    """

    NOTEBOOK = "llm_reasoner/notebooks/flanagan_live_generation.ipynb"


class NotebookKey(StrEnum):
    """
    Name fields defined by the notebook file format.
    """

    CELLS = "cells"
    CELL_TYPE = "cell_type"
    EXECUTION_COUNT = "execution_count"
    IDENTIFIER = "id"
    OUTPUTS = "outputs"
    SOURCE = "source"


class NotebookCellType(StrEnum):
    """
    Identify executable notebook cells.
    """

    CODE = "code"


class RuntimeName(StrEnum):
    """
    Name values shared between the notebook's executable cells.
    """

    API_KEY_ENVIRONMENT_VARIABLE = "ApiKeyEnvironmentVariable"
    CACHING_GENERATOR = "caching_generator"
    ENCODED_RECORD = "encoded_record"
    HYPOTHESIS_GENERATOR = "hypothesis_generator"
    RECORD = "record"
    SELECTION = "selection"
    TASK = "task"
    VIEW = "view"


class CredentialPrefix(StrEnum):
    """
    Identify credential prefixes that must never appear in the notebook.
    """

    OPENAI = "sk-"
    ANTHROPIC = "sk-ant-"


# %% notebook loading


def load_flanagan_notebook() -> dict[str, Any]:
    """
    Load the live Flanagan notebook as notebook-format data.
    """
    repository_root = Path(__file__).resolve().parents[2]
    notebook_path = repository_root / RepositoryPath.NOTEBOOK
    return json.loads(notebook_path.read_text(encoding="utf-8"))


@dataclass
class NotebookRuntime:
    """
    Execute selected code cells in one shared notebook namespace.
    """

    notebook: dict[str, Any]
    """
    Notebook-format data containing the cells to execute.
    """

    namespace: dict[str, Any] = field(default_factory=dict)
    """
    Shared names created by prior cells.
    """

    def run(self, cell_identifier: NotebookCell) -> None:
        """
        Execute one named code cell in the shared namespace.
        """
        cell = next(
            candidate
            for candidate in self.notebook[NotebookKey.CELLS]
            if candidate[NotebookKey.IDENTIFIER] == cell_identifier.value
        )
        source = "".join(cell[NotebookKey.SOURCE])
        exec(
            compile(source, filename=cell_identifier.value, mode="exec"), self.namespace
        )


# %% offline generation replacement


@dataclass(frozen=True)
class ServableGeneration:
    """
    Carry the admitted record returned by an offline generation replacement.
    """

    record: HypothesisRecord
    """
    Validated hypothesis supplied to the notebook query cells.
    """


@dataclass(frozen=True)
class ServableGenerationProvider:
    """
    Return a prepared hypothesis without contacting a language-model service.
    """

    result: ServableGeneration
    """
    Generation result returned for every request.
    """

    def get(
        self,
        task: ConceptReference,
        action_label: str,
        selection: ModuleSelection,
        force: bool,
    ) -> ServableGeneration:
        """
        Return the prepared validated record.
        """
        return self.result


def validated_record(runtime: NotebookRuntime) -> HypothesisRecord:
    """
    Build a minimal admitted record for offline notebook execution.
    """
    task = runtime.namespace[RuntimeName.TASK]
    selection = runtime.namespace[RuntimeName.SELECTION]
    generator = runtime.namespace[RuntimeName.HYPOTHESIS_GENERATOR]
    root = ActionPhaseStructure(decomposes_task=task)
    provenance = Provenance(
        generator=ServableGenerationProvider.__name__,
        prompt_version=generator.prompt_version,
        module_versions=selection.module_versions,
    )
    record = HypothesisRecord.from_root(root, task, provenance)
    return record.validated(ValidationReport())


# %% workflow coverage


class TestFlanaganLiveGenerationNotebook:
    """
    Verify the notebook remains safe, ordered, and executable.
    """

    def test_notebook_contains_complete_ordered_workflow(self) -> None:
        """
        Require every generation and query stage in its intended order.
        """
        notebook = load_flanagan_notebook()

        code_cell_ids = [
            cell[NotebookKey.IDENTIFIER]
            for cell in notebook[NotebookKey.CELLS]
            if cell[NotebookKey.CELL_TYPE] == NotebookCellType.CODE
        ]

        assert code_cell_ids == [cell.value for cell in NotebookCell]

    def test_notebook_code_cells_are_valid_python(self) -> None:
        """
        Require every executable cell to compile independently.
        """
        notebook = load_flanagan_notebook()

        for cell in notebook[NotebookKey.CELLS]:
            if cell[NotebookKey.CELL_TYPE] == NotebookCellType.CODE:
                ast.parse(
                    "".join(cell[NotebookKey.SOURCE]),
                    filename=cell[NotebookKey.IDENTIFIER],
                )

    def test_notebook_contains_no_saved_execution_state_or_secret_literals(
        self,
    ) -> None:
        """
        Keep credentials and prior live responses out of the repository.
        """
        notebook = load_flanagan_notebook()
        code_cells = [
            cell
            for cell in notebook[NotebookKey.CELLS]
            if cell[NotebookKey.CELL_TYPE] == NotebookCellType.CODE
        ]
        string_literals = [
            node.value
            for cell in code_cells
            for node in ast.walk(ast.parse("".join(cell[NotebookKey.SOURCE])))
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        ]

        assert all(cell[NotebookKey.EXECUTION_COUNT] is None for cell in code_cells)
        assert all(cell[NotebookKey.OUTPUTS] == [] for cell in code_cells)
        assert not any(
            value.startswith(tuple(prefix.value for prefix in CredentialPrefix))
            for value in string_literals
        )

    def test_notebook_workflow_executes_without_live_model_call(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Run generation, queries, and encoding with an offline admitted record.
        """
        runtime = NotebookRuntime(load_flanagan_notebook())
        preparation_cells = (
            NotebookCell.IMPORTS,
            NotebookCell.CONFIGURATION,
            NotebookCell.TASK,
            NotebookCell.SELECTION,
            NotebookCell.SCHEMA,
        )
        for cell_identifier in preparation_cells:
            runtime.run(cell_identifier)

        environment_variable = runtime.namespace[
            RuntimeName.API_KEY_ENVIRONMENT_VARIABLE
        ].OPENAI.value
        monkeypatch.setenv(environment_variable, "offline-test-credential")
        runtime.run(NotebookCell.PIPELINE)
        assert isinstance(
            runtime.namespace[RuntimeName.CACHING_GENERATOR], CachingGenerator
        )

        expected_record = validated_record(runtime)
        runtime.namespace[RuntimeName.CACHING_GENERATOR] = ServableGenerationProvider(
            ServableGeneration(expected_record)
        )
        result_cells = (
            NotebookCell.GENERATION,
            NotebookCell.RECORD_SUMMARY,
            NotebookCell.PHASE_QUERY,
            NotebookCell.CONTROL_QUERY,
            NotebookCell.RECOVERY_QUERY,
            NotebookCell.TYPE_QUERY,
            NotebookCell.JSON_VIEW,
        )
        for cell_identifier in result_cells:
            runtime.run(cell_identifier)

        assert runtime.namespace[RuntimeName.RECORD] is expected_record
        assert (
            runtime.namespace[RuntimeName.RECORD].status is HypothesisStatus.VALIDATED
        )
        assert runtime.namespace[RuntimeName.VIEW].record is expected_record
        assert runtime.namespace[RuntimeName.ENCODED_RECORD] == encode(expected_record)
