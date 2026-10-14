from __future__ import annotations

import pytest
from llm_reasoner.knowledge_updated.microworlds.flanagan.namespaces import (
    FlanaganNamespace,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.ontology import (
    CORE,
    FLANAGAN,
    GAZE,
    LEARNING,
    OBJECT_DYNAMICS,
    OBSERVATION,
    PERCEPTION,
    POURING,
    PREHENSION,
)
from llm_reasoner.knowledge_updated.ontological.documentation import (
    DocumentationReader,
)

# %% fixtures

MODULES_BY_NAMESPACE = {
    FlanaganNamespace.CORE: CORE,
    FlanaganNamespace.OBJECT_DYNAMICS: OBJECT_DYNAMICS,
    FlanaganNamespace.PREHENSION: PREHENSION,
    FlanaganNamespace.POURING: POURING,
    FlanaganNamespace.GAZE: GAZE,
    FlanaganNamespace.OBSERVATION: OBSERVATION,
    FlanaganNamespace.LEARNING: LEARNING,
    FlanaganNamespace.PERCEPTION: PERCEPTION,
}
"""
Each module scope beside the namespace it declares under.
"""

# %% modules


class TestFlanaganModules:
    """
    The microworld is divided into modules that each declare under their own namespace.
    """

    def test_ontology_holds_exactly_the_eight_modules(self) -> None:
        assert set(FLANAGAN.modules()) == set(MODULES_BY_NAMESPACE.values())

    @pytest.mark.parametrize("namespace", list(MODULES_BY_NAMESPACE))
    def test_each_module_declares_under_its_namespace(
        self, namespace: FlanaganNamespace
    ) -> None:
        assert MODULES_BY_NAMESPACE[namespace].namespace.iri == namespace

    def test_module_names_are_unique(self) -> None:
        names = [module.name for module in FLANAGAN.modules()]

        assert len(set(names)) == len(names)

    def test_every_module_has_a_version(self) -> None:
        assert all(FLANAGAN.versions().values())

    def test_namespaces_are_documented(self) -> None:
        assert DocumentationReader().find_undocumented_members(FlanaganNamespace) == ()


# %% dependencies


class TestFlanaganDependencies:
    """
    Dependencies point one way: nothing action-specific is reachable from the core.
    """

    def test_core_depends_on_nothing(self) -> None:
        assert CORE.depends_on == ()

    def test_pouring_reaches_the_core_through_prehension(self) -> None:
        assert POURING.transitive_dependencies() == (
            CORE,
            OBJECT_DYNAMICS,
            PREHENSION,
        )

    def test_prehension_builds_on_object_dynamics(self) -> None:
        assert PREHENSION.transitive_dependencies() == (CORE, OBJECT_DYNAMICS)

    def test_observation_builds_on_gaze(self) -> None:
        assert OBSERVATION.transitive_dependencies() == (CORE, GAZE)

    def test_learning_builds_on_gaze_and_object_dynamics(self) -> None:
        assert set(LEARNING.transitive_dependencies()) == {
            CORE,
            GAZE,
            OBJECT_DYNAMICS,
        }

    def test_perception_builds_only_on_the_core(self) -> None:
        assert PERCEPTION.transitive_dependencies() == (CORE,)

    def test_optional_capabilities_do_not_depend_on_each_other_needlessly(self) -> None:
        optional = (GAZE, OBSERVATION, LEARNING, PERCEPTION)
        pouring_chain = {PREHENSION, POURING}

        assert all(
            pouring_chain.isdisjoint(module.transitive_dependencies())
            for module in optional
        )
