from __future__ import annotations

import llm_reasoner.knowledge_updated.microworlds.flanagan.prehension  # noqa: F401  declares the module
from llm_reasoner.knowledge_updated.microworlds.flanagan.ontology import (
    CORE,
    FLANAGAN,
    OBJECT_DYNAMICS,
    PREHENSION,
)
from llm_reasoner.knowledge_updated.ontological.validation import (
    Check,
    OntologyValidator,
)

# %% the whole module


class TestPrehensionIsValid:
    """
    Whatever the module declares satisfies every rule that needs the whole ontology.
    """

    def test_module_has_no_errors(self) -> None:
        assert OntologyValidator(FLANAGAN).validate(PREHENSION.name).errors == ()

    def test_module_has_no_warnings(self) -> None:
        assert OntologyValidator(FLANAGAN).validate(PREHENSION.name).warnings == ()

    def test_module_declares_something_to_validate(self) -> None:
        assert FLANAGAN.concepts(PREHENSION.name)


class TestModulesItBuildsOnStayValid:
    """
    Specialising the core and object dynamics leaves them coherent.
    """

    def test_core_has_no_errors(self) -> None:
        assert OntologyValidator(FLANAGAN).validate(CORE.name).errors == ()

    def test_core_no_longer_has_siblings_that_may_share_instances(self) -> None:
        report = OntologyValidator(FLANAGAN).validate(CORE.name)

        assert report.of(Check.SIBLINGS_OVERLAP) == ()

    def test_object_dynamics_has_no_warnings(self) -> None:
        report = OntologyValidator(FLANAGAN).validate(OBJECT_DYNAMICS.name)

        assert report.warnings == ()
