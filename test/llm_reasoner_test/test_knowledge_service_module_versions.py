from __future__ import annotations

from llm_reasoner.knowledge.hypothesis.provenance import Provenance
from llm_reasoner.knowledge.hypothesis.record import (
    HypothesisRecord,
    ValidationReport,
)
from llm_reasoner.knowledge.hypothesis.store import InMemoryStore
from llm_reasoner.knowledge.microworlds.flanagan.core import ActionPhaseStructure
from llm_reasoner.knowledge.microworlds.registry import (
    GRASPING_PROFILE,
    POURING_PROFILE,
    current_versions,
    select_modules,
)
from llm_reasoner.knowledge.ontological.decorators import ConceptReference
from llm_reasoner.knowledge.query import KnowledgeService

# %% fixtures


def validated_record_for(task: ConceptReference) -> HypothesisRecord:
    """
    Return a servable record generated under the modules the task selects.
    """
    return HypothesisRecord.from_root(
        ActionPhaseStructure(local_id="structure", decomposes_task=task),
        action_type=task,
        provenance=Provenance(
            generator="test",
            prompt_version="test",
            module_versions=select_modules(task).module_versions,
        ),
    ).validated(ValidationReport())


# %% versions derived from the action


class TestServiceModuleVersionsFollowTheAction:
    """
    A service without explicit versions judges each record by its own action's modules.
    """

    def test_record_generated_under_the_selected_modules_is_served(self) -> None:
        task = POURING_PROFILE.task
        record = validated_record_for(task)
        store = InMemoryStore()
        store.put(record)
        service = KnowledgeService(store=store)

        view = service.view_for(task, generate_if_missing=False)

        assert view is not None
        assert view.record is record

    def test_one_service_serves_actions_with_different_module_selections(
        self,
    ) -> None:
        pouring = validated_record_for(POURING_PROFILE.task)
        grasping = validated_record_for(GRASPING_PROFILE.task)
        store = InMemoryStore()
        store.put(pouring)
        store.put(grasping)
        service = KnowledgeService(store=store)

        pouring_view = service.view_for(POURING_PROFILE.task, generate_if_missing=False)
        grasping_view = service.view_for(
            GRASPING_PROFILE.task, generate_if_missing=False
        )

        assert pouring_view is not None
        assert grasping_view is not None
        assert pouring_view.record is pouring
        assert grasping_view.record is grasping

    def test_explanation_reports_a_served_record_as_a_hit(self) -> None:
        task = POURING_PROFILE.task
        record = validated_record_for(task)
        store = InMemoryStore()
        store.put(record)
        service = KnowledgeService(store=store)

        lookup = service.explain(task)

        assert lookup.record is record
        assert lookup.stale == ()


# %% explicit versions


class TestExplicitModuleVersionsRemainAuthoritative:
    """
    Versions supplied to the service override the ones an action would select.
    """

    def test_record_generated_under_other_versions_is_stale(self) -> None:
        task = POURING_PROFILE.task
        record = validated_record_for(task)
        store = InMemoryStore()
        store.put(record)
        service = KnowledgeService(store=store, module_versions=current_versions())

        lookup = service.explain(task)

        assert lookup.record is None
        assert lookup.stale == (record,)
