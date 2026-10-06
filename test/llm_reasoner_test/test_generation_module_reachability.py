from __future__ import annotations

import pytest
from llm_reasoner.clients.base import ScriptedClient
from llm_reasoner.knowledge.generation.backend import (
    HypothesisGenerator,
    UnreachableModules,
)
from llm_reasoner.knowledge.generation.schema import derive_schema
from llm_reasoner.knowledge.microworlds.flanagan.gaze import GazeControlProfile
from llm_reasoner.knowledge.microworlds.flanagan.pouring import PouringPhaseStructure
from llm_reasoner.knowledge.microworlds.registry import (
    DEFAULT_PROFILE,
    FLANAGAN_CORE,
    FLANAGAN_GAZE,
    FLANAGAN_OBJECT_DYNAMICS,
    FLANAGAN_POURING,
    FLANAGAN_PREHENSION,
    POURING_PROFILE,
    Capability,
    ModuleSelection,
    select_modules,
)
from llm_reasoner.knowledge.ontological.constraints import ConstraintSuite

# %% fixtures

POURING_MODULES = (
    FLANAGAN_CORE.name,
    FLANAGAN_OBJECT_DYNAMICS.name,
    FLANAGAN_PREHENSION.name,
    FLANAGAN_POURING.name,
)
"""
The modules a pouring action selects on its own.
"""


def generator() -> tuple[HypothesisGenerator, ScriptedClient]:
    """
    Return an offline generator and the client it would call.
    """
    client = ScriptedClient()
    return (
        HypothesisGenerator(
            client=client,
            suite=ConstraintSuite(name="construction", constraints=()),
        ),
        client,
    )


# %% schema reporting


class TestSchemaReportsUnreachableModules:
    """
    A schema names the selected modules that its root cannot reach.
    """

    def test_module_the_root_does_not_link_to_is_reported(self) -> None:
        schema = derive_schema(
            modules=(*POURING_MODULES, FLANAGAN_GAZE.name),
            root=PouringPhaseStructure.__name__,
        )

        assert schema.unreachable_modules == (FLANAGAN_GAZE.name,)

    def test_modules_a_root_uses_are_not_reported(self) -> None:
        schema = derive_schema(
            modules=POURING_MODULES, root=PouringPhaseStructure.__name__
        )

        assert schema.unreachable_modules == ()

    def test_optional_module_is_reachable_from_its_own_root(self) -> None:
        schema = derive_schema(
            modules=(FLANAGAN_CORE.name, FLANAGAN_GAZE.name),
            root=GazeControlProfile.__name__,
        )

        assert schema.unreachable_modules == ()


# %% generator refusal


class TestGeneratorRefusesUnreachableSelections:
    """
    Generation stops before any model call when a selected module cannot contribute.
    """

    def selection_with_gaze_under_the_pouring_root(self) -> ModuleSelection:
        return select_modules(
            POURING_PROFILE.task,
            additional_capabilities=frozenset({Capability.GAZE_CONTROL}),
        )

    def test_generation_names_the_unreachable_module_and_spends_no_call(self) -> None:
        hypothesis_generator, client = generator()
        selection = self.selection_with_gaze_under_the_pouring_root()

        with pytest.raises(UnreachableModules) as refusal:
            hypothesis_generator.generate(POURING_PROFILE.task, selection=selection)

        assert refusal.value.modules == (FLANAGAN_GAZE.name,)
        assert refusal.value.root == PouringPhaseStructure.__name__
        assert client.calls == []

    def test_prompt_preview_refuses_the_same_selection(self) -> None:
        hypothesis_generator, _ = generator()
        selection = self.selection_with_gaze_under_the_pouring_root()

        with pytest.raises(UnreachableModules):
            hypothesis_generator.describe(POURING_PROFILE.task, selection=selection)

    def test_optional_module_with_its_own_root_is_accepted(self) -> None:
        hypothesis_generator, _ = generator()
        hypothesis_generator.root_class = GazeControlProfile.__name__
        selection = ModuleSelection(
            task=POURING_PROFILE.task,
            profile=DEFAULT_PROFILE,
            modules=(FLANAGAN_CORE, FLANAGAN_GAZE),
        )

        description = hypothesis_generator.describe(
            POURING_PROFILE.task, selection=selection
        )

        assert GazeControlProfile.__name__ in description
