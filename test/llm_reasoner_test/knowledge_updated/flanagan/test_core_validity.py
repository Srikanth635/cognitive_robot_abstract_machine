from __future__ import annotations

import llm_reasoner.knowledge_updated.microworlds.flanagan.core  # noqa: F401  declares the core
from llm_reasoner.knowledge_updated.microworlds.flanagan.ontology import CORE, FLANAGAN
from llm_reasoner.knowledge_updated.ontological.validation import (
    Check,
    OntologyValidator,
)

# %% the whole core


class TestCoreIsValid:
    """
    Whatever the core declares satisfies every rule that needs the whole ontology.

    The core leaves abstract extension points, such as the kinds of estimate, for the
    prehension and pouring modules to specialise. Until those modules are declared, the
    only finding tolerated is an abstract concept that nothing specialises yet.
    """

    TOLERATED_UNTIL_ALL_MODULES_EXIST = {Check.ABSTRACT_WITHOUT_SUBCONCEPT}

    def test_core_has_no_errors(self) -> None:
        report = OntologyValidator(FLANAGAN).validate(CORE.name)

        assert report.errors == ()

    def test_core_has_no_warnings_beyond_the_tolerated_ones(self) -> None:
        report = OntologyValidator(FLANAGAN).validate(CORE.name)

        assert {
            violation.check for violation in report.warnings
        } <= self.TOLERATED_UNTIL_ALL_MODULES_EXIST

    def test_core_declares_something_to_validate(self) -> None:
        assert FLANAGAN.concepts(CORE.name)
        assert FLANAGAN.value_partitions(CORE.name)
