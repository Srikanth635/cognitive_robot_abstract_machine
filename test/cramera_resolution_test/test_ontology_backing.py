"""
Tests for naming ontology terms with their prefixes and the ontology they come from.
"""

from pathlib import Path

import pytest

krrood = pytest.importorskip("krrood", reason="EQL requires krrood")

from llm_reasoner.knowledge.access import (  # noqa: E402
    core,
    CRAM_AICOR_NAMESPACE,
    HYPOTHESIS_CONCEPT_NAMESPACE,
    PREDICTIVE_MANIPULATION_NAMESPACE_ROOT,
    PredictiveManipulationOntologyFile,
)

from cramera_resolution.hypothesis_knowledge.ontology_backing import (  # noqa: E402
    OntologyBacking,
    TermSource,
)


@pytest.fixture
def backing():
    return OntologyBacking()


# %% prefixed names


class TestPrefixedNames:
    def test_every_microworld_module_has_a_namespace(self, backing):
        assert {
            PREDICTIVE_MANIPULATION_NAMESPACE_ROOT + Path(module).stem + "#"
            for module in PredictiveManipulationOntologyFile
        } <= {namespace.iri for namespace in backing.namespaces}

    def test_a_term_is_written_with_its_namespaces_prefix(self, backing):
        for namespace in backing.namespaces:
            assert backing.prefixed(namespace.iri + "Term") == (
                f"{namespace.prefix}:Term"
            )

    def test_a_term_outside_every_namespace_is_written_in_full(self, backing):
        iri = "http://example.org/elsewhere#Term"
        assert backing.prefixed(iri) == iri
        assert backing.source_of(iri) is TermSource.UNKNOWN

    @pytest.mark.parametrize(
        "namespace, source",
        [
            (CRAM_AICOR_NAMESPACE, TermSource.CRAM_AICOR),
            (HYPOTHESIS_CONCEPT_NAMESPACE, TermSource.HYPOTHESIS),
            (
                core.ActionPhase.term.iri.partition("#")[0] + "#",
                TermSource.PREDICTIVE_MANIPULATION,
            ),
        ],
    )
    def test_a_term_is_attributed_to_its_ontology(self, backing, namespace, source):
        assert backing.source_of(namespace + "Term") is source
