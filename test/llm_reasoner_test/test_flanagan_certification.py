from __future__ import annotations

from importlib.util import find_spec

from llm_reasoner.knowledge.generation.backend import (
    GenerationPromptVersion,
    HypothesisGenerator,
)
from llm_reasoner.knowledge.hypothesis.provenance import CacheKey, Provenance
from llm_reasoner.knowledge.hypothesis.record import (
    HypothesisRecord,
    ValidationReport,
)
from llm_reasoner.knowledge.hypothesis.scope import collect_reachable
from llm_reasoner.knowledge.hypothesis.status import Defeater
from llm_reasoner.knowledge.hypothesis.store import InMemoryStore
from llm_reasoner.knowledge.microworlds.flanagan.canonical import (
    build_pouring_phase_structure,
    build_water_pouring_phase_structure,
)
from llm_reasoner.knowledge.microworlds.flanagan.constraints import SUITE
from llm_reasoner.knowledge.microworlds.flanagan.core import FLAN, ActionPhaseStructure
from llm_reasoner.knowledge.microworlds.flanagan.gaze import GAZE
from llm_reasoner.knowledge.microworlds.flanagan.learning import LEARNING
from llm_reasoner.knowledge.microworlds.flanagan.object_dynamics import (
    OBJECT_DYNAMICS,
)
from llm_reasoner.knowledge.microworlds.flanagan.observation import OBSERVATION
from llm_reasoner.knowledge.microworlds.flanagan.perception import PERCEPTION
from llm_reasoner.knowledge.microworlds.flanagan.pouring import POUR
from llm_reasoner.knowledge.microworlds.flanagan.prehension import PREHENSION
from llm_reasoner.knowledge.microworlds.registry import MODULES, current_versions
from llm_reasoner.knowledge.ontological.decorators import ConceptReference
from llm_reasoner.knowledge.query import KnowledgeService

PROJECT_NAMESPACE = "http://www.ease-crc.org/ont/llm_reasoner/flanagan/"
"""
Stable namespace root owned by the project.
"""

# %% namespace migration


class TestCertifiedNamespaces:
    """
    Flanagan terms use one stable project-owned namespace family.
    """

    def test_every_module_uses_the_project_namespace(self) -> None:
        namespaces = (
            FLAN,
            OBJECT_DYNAMICS,
            PREHENSION,
            POUR,
            GAZE,
            OBSERVATION,
            LEARNING,
            PERCEPTION,
        )

        assert all(namespace.startswith(PROJECT_NAMESPACE) for namespace in namespaces)
        assert all("example.org" not in namespace for namespace in namespaces)
        assert len(set(namespaces)) == len(namespaces)


# %% fixture ownership


class TestCanonicalFixtureOwnership:
    """
    Reference structures are owned by the microworld they certify.
    """

    def test_canonical_structure_module_belongs_to_llm_reasoner(self) -> None:
        assert (
            find_spec("llm_reasoner.knowledge.microworlds.flanagan.canonical")
            is not None
        )

    def test_obsolete_turtle_loader_is_absent(self) -> None:
        assert find_spec("llm_reasoner.knowledge.microworlds.flanagan.fixtures") is None

    def test_cereal_reference_has_no_acceptance_diagnostics(self) -> None:
        structure = build_pouring_phase_structure()

        assert SUITE.evaluate(collect_reachable(structure)) == []

    def test_water_reference_has_no_acceptance_diagnostics(self) -> None:
        structure = build_water_pouring_phase_structure()

        assert SUITE.evaluate(collect_reachable(structure)) == []


# %% cache migration


class TestCertifiedKnowledgeVersions:
    """
    Pre-certification generated knowledge is deliberately expired.
    """

    def test_prompt_and_all_module_versions_reach_the_certified_contract(self) -> None:
        assert HypothesisGenerator.__dataclass_fields__["prompt_version"].default == (
            GenerationPromptVersion.REPRESENTATION_OWNERSHIP.value
        )
        assert current_versions() == {module.name: "1.0.0" for module in MODULES}

    def test_prompt_change_invalidates_an_otherwise_current_cache_key(self) -> None:
        versions = current_versions()
        key = CacheKey(
            action_type="soma:PouringInto",
            module_versions=tuple(sorted(versions.items())),
            generator="test",
            prompt_version=GenerationPromptVersion.CONTINUOUS_CONTROL.value,
        )

        assert key.invalidated_by(
            versions,
            prompt_version=GenerationPromptVersion.REPRESENTATION_OWNERSHIP.value,
        )

    def test_legacy_key_without_module_versions_is_expired(self) -> None:
        key = CacheKey(
            action_type="soma:PouringInto",
            module_versions=(),
            generator="test",
            prompt_version=GenerationPromptVersion.PRE_CERTIFICATION.value,
        )

        assert key.invalidated_by(
            current_versions(),
            prompt_version=GenerationPromptVersion.REPRESENTATION_OWNERSHIP.value,
        )

    def test_refuted_record_from_an_old_prompt_is_stale_not_currently_refuted(
        self,
    ) -> None:
        task = ConceptReference(iri="soma:PouringInto", label="PouringInto")
        record = HypothesisRecord.from_root(
            ActionPhaseStructure(local_id="structure", decomposes_task=task),
            action_type=task,
            provenance=Provenance(
                generator="test",
                prompt_version=GenerationPromptVersion.CONTINUOUS_CONTROL.value,
                module_versions=current_versions(),
            ),
        )
        record.refute(Defeater(reason="old failure", source="test"))
        store = InMemoryStore()
        store.put(record)

        lookup = store.lookup(
            task.iri,
            current_versions(),
            prompt_version=GenerationPromptVersion.REPRESENTATION_OWNERSHIP.value,
        )

        assert lookup.stale == (record,)
        assert lookup.refuted == ()

    def test_knowledge_service_does_not_serve_an_old_prompt_record(self) -> None:
        task = ConceptReference(iri="soma:PouringInto", label="PouringInto")
        record = HypothesisRecord.from_root(
            ActionPhaseStructure(local_id="structure", decomposes_task=task),
            action_type=task,
            provenance=Provenance(
                generator="test",
                prompt_version=GenerationPromptVersion.CONTINUOUS_CONTROL.value,
                module_versions=current_versions(),
            ),
        ).validated(ValidationReport())
        store = InMemoryStore()
        store.put(record)
        service = KnowledgeService(
            store=store,
            module_versions=current_versions(),
            prompt_version=GenerationPromptVersion.REPRESENTATION_OWNERSHIP.value,
        )

        lookup = service.explain(task)

        assert lookup.record is None
        assert lookup.stale == (record,)
