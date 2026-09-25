"""
Tests for generating a pouring structure from a natural-language instruction.

The generation pipeline itself belongs to ``llm_reasoner`` and is tested there; what
matters here is the policy this package adds on top -- how an instruction becomes a
cacheable task, and how a produced record becomes the knowledge the viewer displays
and queries.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest
from llm_reasoner.knowledge.hypothesis.record import HypothesisRecord, ValidationReport
from llm_reasoner.knowledge.microworlds.flanagan.pouring import PouringPhaseStructure
from llm_reasoner.knowledge.microworlds.registry import select_modules

from cram_viz import hypothesis_knowledge
from cram_viz.hypothesis_knowledge import build_hypothesis_record
from cram_viz.hypothesis_knowledge.generation import (
    HypothesisGenerationError,
    HypothesisGenerationService,
    HypothesisSource,
    MissingCredentialsError,
    ModelHypothesisSource,
)
from cram_viz.hypothesis_knowledge.pouring_structure import (
    build_pouring_phase_structure,
)


@dataclass
class FakeCachedResult:
    """Stands in for ``llm_reasoner``'s ``CachedResult`` -- the two fields
    ``ModelHypothesisSource.record_for`` actually reads."""

    record: HypothesisRecord
    servable: bool


@dataclass
class FakeCachingGenerator:
    """Stands in for ``llm_reasoner``'s ``CachingGenerator``: returns a prepared
    result instead of calling a model."""

    result: FakeCachedResult

    def get(self, task, action_label=None, selection=None, force=False):
        return self.result


@dataclass
class RecordingHypothesisSource(HypothesisSource):
    """A source that returns a prepared record and remembers what it was asked for."""

    record: HypothesisRecord
    """The record handed back for any instruction."""

    requested_instructions: list = None
    """Every instruction this source was asked to produce a record for."""

    def __post_init__(self) -> None:
        self.requested_instructions = []

    def record_for(self, instruction: str) -> HypothesisRecord:
        self.requested_instructions.append(instruction)
        return self.record


@dataclass
class FailingHypothesisSource(HypothesisSource):
    """A source that always refuses, like a generation that never became
    admissible."""

    message: str = "was refuted (status: refuted): [C23] ERROR ..."

    def record_for(self, instruction: str) -> HypothesisRecord:
        raise HypothesisGenerationError(self.message)


@pytest.fixture
def restore_active_knowledge():
    """Put the module-level active knowledge back after a test swaps it."""
    previous = hypothesis_knowledge.active()
    yield
    hypothesis_knowledge.activate(previous)


@pytest.fixture
def sample_source():
    structure = build_pouring_phase_structure()
    return RecordingHypothesisSource(record=build_hypothesis_record(structure))


# %% instruction -> knowledge
class TestHypothesisGenerationService:
    def test_generated_knowledge_wraps_the_records_own_root(
        self, sample_source, restore_active_knowledge
    ):
        service = HypothesisGenerationService(source=sample_source)

        knowledge = service.generate("Pour milk from the carton into the mug")

        assert knowledge.structure is sample_source.record.root
        assert knowledge.instruction == "Pour milk from the carton into the mug"
        assert knowledge.generated

    def test_the_instruction_is_passed_through_to_the_source(
        self, sample_source, restore_active_knowledge
    ):
        service = HypothesisGenerationService(source=sample_source)

        service.generate("Pour cereal into the bowl")

        assert sample_source.requested_instructions == ["Pour cereal into the bowl"]

    def test_generating_replaces_what_the_viewer_shows_and_queries(
        self, sample_source, restore_active_knowledge
    ):
        service = HypothesisGenerationService(source=sample_source)

        knowledge = service.generate("Pour cereal into the bowl")

        assert hypothesis_knowledge.active() is knowledge

    def test_the_projection_carries_all_three_representations(
        self, sample_source, restore_active_knowledge
    ):
        service = HypothesisGenerationService(source=sample_source)

        knowledge = service.generate("Pour cereal into the bowl")

        type_names = {node.type_name for node in knowledge.projection.nodes}
        assert "PouringPhaseStructure" in type_names
        assert "MinskyActionFrame" in type_names
        assert "ImageSchemaProfile" in type_names

    def test_the_instruction_becomes_the_graphs_centre_label(
        self, sample_source, restore_active_knowledge
    ):
        service = HypothesisGenerationService(source=sample_source)

        knowledge = service.generate("Pour milk into the mug")

        assert knowledge.projection.instruction.label == "Pour milk into the mug"


class TestHypothesisGenerationServiceFallback:
    """
    Live generation can refuse an instruction the model cannot yet render
    admissibly (see :mod:`cram_viz.hypothesis_knowledge.generation`'s own
    ``HypothesisGenerationError`` for why). Rather than leave the viewer with
    nothing, generation falls back to the checked-in, hand-authored pouring
    sample -- still a real, EQL-queryable ``PouringPhaseStructure``, just not
    the one the instruction actually asked for.
    """

    def test_a_refused_generation_falls_back_to_the_checked_in_sample(
        self, restore_active_knowledge
    ):
        service = HypothesisGenerationService(source=FailingHypothesisSource())

        knowledge = service.generate("Pour cereal into the bowl")

        assert isinstance(knowledge.structure, PouringPhaseStructure)
        assert knowledge.structure.defines_phase

    def test_the_fallback_is_reported_as_not_generated(self, restore_active_knowledge):
        service = HypothesisGenerationService(source=FailingHypothesisSource())

        knowledge = service.generate("Pour cereal into the bowl")

        assert knowledge.generated is False

    def test_the_fallback_still_becomes_the_active_knowledge(
        self, restore_active_knowledge
    ):
        service = HypothesisGenerationService(source=FailingHypothesisSource())

        knowledge = service.generate("Pour cereal into the bowl")

        assert hypothesis_knowledge.active() is knowledge

    def test_the_fallback_still_carries_the_typed_instruction(
        self, restore_active_knowledge
    ):
        service = HypothesisGenerationService(source=FailingHypothesisSource())

        knowledge = service.generate("Pour cereal into the bowl")

        assert knowledge.instruction == "Pour cereal into the bowl"
        assert knowledge.projection.instruction.label == "Pour cereal into the bowl"

    def test_a_successful_generation_does_not_fall_back(
        self, sample_source, restore_active_knowledge
    ):
        service = HypothesisGenerationService(source=sample_source)

        knowledge = service.generate("Pour cereal into the bowl")

        assert knowledge.structure is sample_source.record.root
        assert knowledge.generated is True


# %% instruction -> cacheable task
class TestModelHypothesisSourceTaskMinting:
    def test_different_instructions_get_different_task_identities(self):
        """
        The store caches on the task IRI, so two instructions must not collide --
        otherwise the second one would silently serve the first one's hypothesis.
        """
        first = ModelHypothesisSource.task_for("Pour cereal into the bowl")
        second = ModelHypothesisSource.task_for("Pour milk into the mug")

        assert first.iri != second.iri

    def test_the_same_instruction_is_stable_across_calls(self):
        """A repeated instruction must hit the cache rather than regenerate."""
        assert (
            ModelHypothesisSource.task_for("Pour cereal into the bowl").iri
            == ModelHypothesisSource.task_for("Pour cereal into the bowl").iri
        )

    def test_the_instruction_is_carried_as_the_task_label(self):
        task = ModelHypothesisSource.task_for("Pour cereal into the bowl")

        assert task.label == "Pour cereal into the bowl"

    def test_a_minted_task_still_selects_the_pouring_microworld(self):
        """
        A fresh IRI must keep the inherited task ancestors, or module selection
        degrades and the model loses the pouring vocabulary entirely.
        """
        selection = select_modules(ModelHypothesisSource.task_for("Pour into a bowl"))

        assert not selection.degraded


# %% servability
class TestModelHypothesisSourceRefusesUnservableRecords:
    """
    ``CachingGenerator.get`` deliberately returns a refuted record instead of
    silently retrying (see its own docstring: "returning the refuted record says
    more than spending calls to fail the same way again") -- so this layer, not
    ``llm_reasoner``, is responsible for turning that into a reported failure
    instead of quietly displaying knowledge that failed every repair attempt.
    """

    def test_a_refuted_record_is_reported_as_a_generation_error(self, tmp_path):
        structure = build_pouring_phase_structure()
        record = build_hypothesis_record(structure)
        record.validated(
            ValidationReport(
                errors=(
                    "[C23] ERROR Subgoal<subgoal-12>: SlipEvent<slip-event> at "
                    "'attainment_signalled_by' has is_failure_event=True, expected False",
                )
            )
        )
        source = ModelHypothesisSource(store_directory=tmp_path, client=object())
        source._generator = FakeCachingGenerator(
            FakeCachedResult(record=record, servable=False)
        )

        with pytest.raises(HypothesisGenerationError) as raised:
            source.record_for("Pour cereal into the bowl")

        assert "SlipEvent" in str(raised.value)

    def test_a_validated_record_is_returned(self, tmp_path):
        structure = build_pouring_phase_structure()
        record = build_hypothesis_record(structure)
        record.validated(ValidationReport())
        source = ModelHypothesisSource(store_directory=tmp_path, client=object())
        source._generator = FakeCachingGenerator(
            FakeCachedResult(record=record, servable=True)
        )

        assert source.record_for("Pour cereal into the bowl") is record


# %% credentials
class TestModelHypothesisSourceCredentials:
    def test_a_missing_key_is_reported_as_a_named_configuration_problem(
        self, monkeypatch, tmp_path
    ):
        """
        Without this the Anthropic SDK raises a bare TypeError about "authentication
        method", which says nothing about which variable the viewer wants set.
        """
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        source = ModelHypothesisSource(
            store_directory=tmp_path, environment_file=tmp_path / "absent.env"
        )

        with pytest.raises(MissingCredentialsError) as raised:
            source.record_for("Pour cereal into the bowl")

        assert "ANTHROPIC_API_KEY" in str(raised.value)

    def test_the_key_is_read_from_an_environment_file(self, monkeypatch, tmp_path):
        """The viewer is started from a shell that has never sourced ``.env``."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        environment_file = tmp_path / ".env"
        environment_file.write_text("ANTHROPIC_API_KEY=sk-ant-from-file\n")
        source = ModelHypothesisSource(
            store_directory=tmp_path, environment_file=environment_file
        )

        source.load_credentials()

        assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-from-file"

    def test_an_already_set_key_is_not_overwritten_by_the_file(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-shell")
        environment_file = tmp_path / ".env"
        environment_file.write_text("ANTHROPIC_API_KEY=sk-ant-from-file\n")
        source = ModelHypothesisSource(
            store_directory=tmp_path, environment_file=environment_file
        )

        source.load_credentials()

        assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-from-shell"

    def test_an_injected_client_needs_no_credentials(self, monkeypatch, tmp_path):
        """A caller supplying its own client must not be asked for an API key."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        source = ModelHypothesisSource(
            store_directory=tmp_path,
            client=object(),
            environment_file=tmp_path / "absent.env",
        )

        source.load_credentials()


# %% live generation
@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="live generation needs ANTHROPIC_API_KEY",
)
class TestLiveGeneration:
    def test_a_real_generation_produces_a_queryable_structure(
        self, restore_active_knowledge, tmp_path
    ):
        service = HypothesisGenerationService(
            source=ModelHypothesisSource(store_directory=tmp_path)
        )

        knowledge = service.generate("Pour cereal from the cereal box into the bowl")

        assert knowledge.structure.defines_phase
        assert knowledge.structure.defines_role
