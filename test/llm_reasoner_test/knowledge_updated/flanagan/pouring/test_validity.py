from __future__ import annotations

import llm_reasoner.knowledge_updated.microworlds.flanagan.pouring  # noqa: F401  declares the module
from llm_reasoner.knowledge_updated.microworlds.flanagan.ontology import (
    CORE,
    FLANAGAN,
    OBJECT_DYNAMICS,
    POURING,
    PREHENSION,
)
from llm_reasoner.knowledge_updated.ontological.validation import (
    Check,
    OntologyValidator,
)

# %% the whole module


class TestPouringIsValid:
    """
    Whatever the module declares satisfies every rule that needs the whole ontology.
    """

    def test_module_has_no_errors(self) -> None:
        assert OntologyValidator(FLANAGAN).validate(POURING.name).errors == ()

    def test_module_has_no_warnings(self) -> None:
        assert OntologyValidator(FLANAGAN).validate(POURING.name).warnings == ()


class TestModulesItBuildsOnStayValid:
    """
    Specialising the modules below leaves them coherent.
    """

    def test_no_module_has_errors(self) -> None:
        for module in (CORE, OBJECT_DYNAMICS, PREHENSION):
            assert OntologyValidator(FLANAGAN).validate(module.name).errors == ()

    def test_the_core_has_no_siblings_that_may_share_instances(self) -> None:
        report = OntologyValidator(FLANAGAN).validate(CORE.name)

        assert report.of(Check.SIBLINGS_OVERLAP) == ()

    def test_the_only_core_finding_left_is_an_abstract_concept_of_a_later_module(
        self,
    ) -> None:
        report = OntologyValidator(FLANAGAN).validate(CORE.name)

        assert {v.check for v in report.warnings} <= {Check.ABSTRACT_WITHOUT_SUBCONCEPT}
