"""
Notebook experiments exercise the SDK loop and real read-only MCP transport.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from litellm.types.utils import ModelResponse
from typing_extensions import TYPE_CHECKING

from repository_knowledge.adapters.serialization import RecordCodec
from repository_knowledge.adapters.sqlite_index import SQLiteIndex
from repository_knowledge.interfaces.agents import AgentTool
from repository_knowledge.interfaces.notebook import (
    ModelSettings,
    NotebookConnection,
    NotebookSession,
)
from repository_knowledge.interfaces.exceptions import (
    InvalidModelSettings,
    LiveRequestsDisabled,
)
from repository_knowledge.exceptions import (
    InvalidSerializedType,
    SourceRevisionMismatch,
    SourceSnapshotMismatch,
)
from repository_knowledge.model import FreshnessReport, SourceScope
from repository_knowledge.model_trials import ModelTrial, QueryOrigin

from .dataset.scripted_completion import ScriptedCompletion
from .dataset.advancing_completion import AdvancingCompletion
from .support import CommittedProject, FixtureSymbol

if TYPE_CHECKING:
    from mcp import Client

# %% notebook configuration


def test_provider_credential_is_absent_from_configuration_representation() -> None:
    """
    Inspecting the configured provider cannot display its credential.
    """
    settings = ModelSettings("openai", "offline-test", api_key="private-test-key")
    assert settings.api_key not in repr(settings)


def test_live_model_requests_require_explicit_enablement(
    committed_project: CommittedProject, indexed_storage: SQLiteIndex
) -> None:
    """
    Default notebook configuration cannot spend a provider credential.
    """
    session = NotebookSession(
        NotebookConnection(committed_project.root, indexed_storage.database_path)
    )
    with pytest.raises(LiveRequestsDisabled):
        asyncio.run(
            session.run("Describe the child.", ModelSettings("openai", "offline-test"))
        )


# %% offline end-to-end exchange


def test_model_uses_real_mcp_evidence_and_records_reported_usage(
    committed_project: CommittedProject,
    indexed_storage: SQLiteIndex,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A mocked provider can select a real source query through the actual SDK loop.
    """
    payload = json.loads(
        (Path(__file__).parent / "dataset/provider_responses.json").read_text(
            encoding="utf-8"
        )
    )
    responses = [ModelResponse.model_validate(item) for item in payload]
    provider = ScriptedCompletion(responses)
    monkeypatch.setattr("litellm.acompletion", provider)
    settings = ModelSettings(
        "openai", "offline-test", api_key="private-test-key", live_enabled=True
    )
    trial = asyncio.run(
        NotebookSession(
            NotebookConnection(committed_project.root, indexed_storage.database_path)
        ).run("Describe the child.", settings)
    )
    assert trial.answer == responses[-1].choices[0].message.content
    assert trial.input_tokens == sum(
        response.usage.prompt_tokens for response in responses
    )
    assert trial.output_tokens == sum(
        response.usage.completion_tokens for response in responses
    )
    assert trial.model_calls == len(responses)
    assert trial.tool_calls == 1
    assert trial.source_commit == committed_project.commit_identifier
    assert trial.source_scope == SourceScope.COMMITTED
    assert [observation.operation for observation in trial.observations] == [
        AgentTool.CHECK_FRESHNESS,
        AgentTool.DESCRIBE_SYMBOL,
    ]
    assert [observation.origin for observation in trial.observations] == [
        QueryOrigin.PREFLIGHT,
        QueryOrigin.MODEL,
    ]
    assert trial.observations[-1].symbol_identifier == FixtureSymbol.CHILD
    assert settings.api_key not in RecordCodec().encode(trial)
    assert RecordCodec().decode(RecordCodec().encode(trial), ModelTrial) == trial
    assert set(
        provider.requests[0]["tools"][0]["function"]["parameters"]["properties"]
    ).isdisjoint({"repository_root", "database_path", "command"})


def test_connection_inspection_does_not_need_a_provider(
    committed_project: CommittedProject, indexed_storage: SQLiteIndex
) -> None:
    """
    Offline inspection validates freshness and an exact declaration over stdio.
    """
    result = asyncio.run(
        NotebookConnection(
            committed_project.root, indexed_storage.database_path
        ).inspect(FixtureSymbol.CHILD)
    )
    assert result.freshness.matches_requested_revision is True
    assert result.definition.symbols[0].symbol_identifier == FixtureSymbol.CHILD
    assert result.available_tools == ["query_repository"]


def test_provider_configuration_cannot_be_saved_as_trial_evidence() -> None:
    """
    The evidence serializer refuses runtime credential settings.
    """
    with pytest.raises(InvalidSerializedType):
        RecordCodec().encode(
            ModelSettings("openai", "offline-test", api_key="private-test-key")
        )


def test_source_changed_during_a_model_turn_is_refused(
    committed_project: CommittedProject,
    indexed_storage: SQLiteIndex,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A model response cannot retain a current-source claim after HEAD changes.
    """
    payload = json.loads(
        (Path(__file__).parent / "dataset/provider_responses.json").read_text(
            encoding="utf-8"
        )
    )
    provider = AdvancingCompletion(
        responses=[ModelResponse.model_validate(item) for item in payload],
        project=committed_project,
    )
    monkeypatch.setattr("litellm.acompletion", provider)
    session = NotebookSession(
        NotebookConnection(committed_project.root, indexed_storage.database_path)
    )
    with pytest.raises(SourceRevisionMismatch):
        asyncio.run(
            session.run(
                "Describe the child.",
                ModelSettings("openai", "offline-test", live_enabled=True),
            )
        )


def test_notebook_finds_the_repository_from_its_subdirectory(tmp_path: Path) -> None:
    """
    Opening a notebook below the repository still locates the source checkout.
    """
    package = tmp_path / "repository_knowledge"
    package.mkdir()
    (package / "pyproject.toml").touch()
    notebooks = package / "notebooks"
    notebooks.mkdir()
    assert NotebookConnection.find_repository(notebooks) == tmp_path


def test_unreported_provider_usage_stays_unknown(
    committed_project: CommittedProject,
    indexed_storage: SQLiteIndex,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Missing usage is never represented as measured zero-token work.
    """
    payload = json.loads(
        (Path(__file__).parent / "dataset/provider_responses.json").read_text(
            encoding="utf-8"
        )
    )
    responses = [ModelResponse.model_validate(item) for item in payload]
    for response in responses:
        response.usage = None
    monkeypatch.setattr("litellm.acompletion", ScriptedCompletion(responses))
    trial = asyncio.run(
        NotebookSession(
            NotebookConnection(committed_project.root, indexed_storage.database_path)
        ).run(
            "Describe the child.",
            ModelSettings("openai", "offline-test", live_enabled=True),
        )
    )
    assert trial.input_tokens is None
    assert trial.output_tokens is None
    assert trial.model_calls is None
    assert trial.tool_calls == 1


def test_stale_source_is_refused_before_contacting_a_provider(
    committed_project: CommittedProject,
    indexed_storage: SQLiteIndex,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Source refusal cannot spend a provider request before the first model turn.
    """
    provider = ScriptedCompletion([])
    monkeypatch.setattr("litellm.acompletion", provider)
    committed_project.git(
        "commit", "--quiet", "--allow-empty", "-m", "Advance before model turn"
    )
    session = NotebookSession(
        NotebookConnection(committed_project.root, indexed_storage.database_path)
    )
    with pytest.raises(SourceRevisionMismatch):
        asyncio.run(
            session.run(
                "Describe the child.",
                ModelSettings("openai", "offline-test", live_enabled=True),
            )
        )
    assert provider.requests == []


def test_returned_answer_redacts_the_runtime_credential(
    committed_project: CommittedProject,
    indexed_storage: SQLiteIndex,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A returned credential cannot enter a saved trial answer.
    """
    payload = json.loads(
        (Path(__file__).parent / "dataset/provider_responses.json").read_text(
            encoding="utf-8"
        )
    )
    settings = ModelSettings(
        "openai", "offline-test", api_key="private-test-key", live_enabled=True
    )
    responses = [ModelResponse.model_validate(item) for item in payload]
    responses[-1].choices[0].message.content = settings.api_key
    monkeypatch.setattr("litellm.acompletion", ScriptedCompletion(responses))
    trial = asyncio.run(
        NotebookSession(
            NotebookConnection(committed_project.root, indexed_storage.database_path)
        ).run("Describe the child.", settings)
    )
    assert settings.api_key not in RecordCodec().encode(trial)


@pytest.mark.parametrize(
    "settings",
    [
        ModelSettings("", "model"),
        ModelSettings("openai", ""),
        ModelSettings("openai", "model", maximum_turns=0),
    ],
)
def test_incomplete_provider_selection_is_refused(settings: ModelSettings) -> None:
    """
    A provider request needs complete names and a positive request budget.
    """
    with pytest.raises(InvalidModelSettings):
        settings.select_model()


def test_inspection_refuses_different_source_captures(
    committed_project: CommittedProject,
    indexed_storage: SQLiteIndex,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Inspection cannot label a declaration with another capture's identity.
    """

    async def previous_source(
        connection: NotebookConnection, client: Client
    ) -> FreshnessReport:
        """
        Report a valid earlier source capture before declaration discovery.
        """
        return FreshnessReport(
            "previous-capture",
            committed_project.commit_identifier,
            committed_project.commit_identifier,
            True,
        )

    monkeypatch.setattr(NotebookConnection, "check_source", previous_source)
    with pytest.raises(SourceSnapshotMismatch):
        asyncio.run(
            NotebookConnection(
                committed_project.root, indexed_storage.database_path
            ).inspect(FixtureSymbol.CHILD)
        )
