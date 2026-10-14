from __future__ import annotations

import llm_reasoner.knowledge_updated.microworlds.flanagan.object_dynamics  # noqa: F401  declares the module
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.parameters import (
    DynamicsEstimate,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.ontology import (
    CORE,
    FLANAGAN,
    OBJECT_DYNAMICS,
)
from llm_reasoner.knowledge_updated.ontological.validation import (
    Check,
    OntologyValidator,
)

# %% the whole module


class TestObjectDynamicsIsValid:
    """
    Whatever the module declares satisfies every rule that needs the whole ontology.
    """

    def test_module_has_no_errors(self) -> None:
        report = OntologyValidator(FLANAGAN).validate(OBJECT_DYNAMICS.name)

        assert report.errors == ()

    def test_module_has_no_warnings(self) -> None:
        report = OntologyValidator(FLANAGAN).validate(OBJECT_DYNAMICS.name)

        assert report.warnings == ()

    def test_module_declares_something_to_validate(self) -> None:
        assert FLANAGAN.concepts(OBJECT_DYNAMICS.name)
        assert FLANAGAN.value_partitions(OBJECT_DYNAMICS.name)


# %% what the module gives the core


class TestCoreExtensionPoints:
    """
    The module specialises the extension points the core leaves abstract.
    """

    def test_dynamics_estimates_can_now_appear_in_a_hypothesis(self) -> None:
        report = OntologyValidator(FLANAGAN).validate(CORE.name)
        flagged = {
            violation.subject
            for violation in report.of(Check.ABSTRACT_WITHOUT_SUBCONCEPT)
        }

        assert DynamicsEstimate.__name__ not in flagged
