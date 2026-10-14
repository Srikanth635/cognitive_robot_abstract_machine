from __future__ import annotations

import pytest
from llm_reasoner.knowledge_updated.ontological.references import (
    Alignment,
    ConceptReference,
    InvalidNamespace,
    Namespace,
    Term,
)
from llm_reasoner.knowledge_updated.ontological.vocabulary import AlignmentRelation

# %% fixtures

SOMA = "http://www.ease-crc.org/ont/SOMA.owl#"
PHYSICAL_TASK = SOMA + "PhysicalTask"
ACTUATING = SOMA + "Actuating"
POURING = SOMA + "Pouring"
GRASPING = SOMA + "Grasping"

POURING_REFERENCE = ConceptReference(
    iri=POURING,
    label="Pouring",
    ancestors=frozenset({ACTUATING, PHYSICAL_TASK}),
)
"""
A task whose ancestors are materialised, as a caller supplies them.
"""

# %% subsumption


class TestConceptSubsumption:
    """
    A reference answers subsumption from its materialised ancestors alone.
    """

    def test_reference_is_subsumed_by_itself(self) -> None:
        assert POURING_REFERENCE.is_subsumed_by(POURING_REFERENCE)

    def test_reference_is_subsumed_by_an_ancestor_reference(self) -> None:
        ancestor = ConceptReference(iri=PHYSICAL_TASK)

        assert POURING_REFERENCE.is_subsumed_by(ancestor)

    def test_reference_is_subsumed_by_an_ancestor_iri(self) -> None:
        assert POURING_REFERENCE.is_subsumed_by(ACTUATING)

    def test_reference_is_not_subsumed_by_an_unrelated_term(self) -> None:
        assert not POURING_REFERENCE.is_subsumed_by(GRASPING)

    def test_ancestor_is_not_subsumed_by_its_descendant(self) -> None:
        ancestor = ConceptReference(iri=PHYSICAL_TASK)

        assert not ancestor.is_subsumed_by(POURING_REFERENCE)

    def test_reference_without_ancestors_is_subsumed_only_by_itself(self) -> None:
        bare = ConceptReference(iri=POURING)

        assert bare.is_subsumed_by(POURING)
        assert not bare.is_subsumed_by(PHYSICAL_TASK)


# %% naming


class TestConceptNaming:
    """
    A reference names its term by label, falling back to the end of its IRI.
    """

    @pytest.mark.parametrize(
        ("iri", "local_name"),
        [
            (POURING, "Pouring"),
            ("http://example.org/ontology/Pouring", "Pouring"),
            ("Pouring", "Pouring"),
        ],
    )
    def test_local_name_is_the_last_segment_of_the_iri(
        self, iri: str, local_name: str
    ) -> None:
        assert ConceptReference(iri=iri).local_name == local_name

    def test_text_form_prefers_the_label(self) -> None:
        reference = ConceptReference(iri=POURING, label="Pouring liquid")

        assert str(reference) == reference.label

    def test_text_form_falls_back_to_the_local_name(self) -> None:
        reference = ConceptReference(iri=POURING)

        assert str(reference) == reference.local_name


# %% value semantics


class TestReferencesAreValues:
    """
    References compare and hash by content, so they can key a mapping.
    """

    def test_equal_content_is_equal_and_hashes_alike(self) -> None:
        first = ConceptReference(iri=POURING, ancestors=frozenset({ACTUATING}))
        second = ConceptReference(iri=POURING, ancestors=frozenset({ACTUATING}))

        assert first == second
        assert hash(first) == hash(second)

    def test_different_ancestors_are_different_references(self) -> None:
        with_ancestors = ConceptReference(iri=POURING, ancestors=frozenset({ACTUATING}))
        without_ancestors = ConceptReference(iri=POURING)

        assert with_ancestors != without_ancestors


# %% alignment


class TestAlignment:
    """
    An alignment records which foreign term a concept relates to, and how.
    """

    def test_alignment_carries_its_term_and_relation(self) -> None:
        alignment = Alignment(iri=PHYSICAL_TASK, relation=AlignmentRelation.SUBCLASS_OF)

        assert alignment.iri == PHYSICAL_TASK
        assert alignment.relation is AlignmentRelation.SUBCLASS_OF

    def test_alignments_differing_in_relation_are_different(self) -> None:
        subclass = Alignment(iri=PHYSICAL_TASK, relation=AlignmentRelation.SUBCLASS_OF)
        related = Alignment(iri=PHYSICAL_TASK, relation=AlignmentRelation.RELATED_TO)

        assert subclass != related

    def test_alignment_names_its_term_by_local_name(self) -> None:
        alignment = Alignment(iri=PHYSICAL_TASK, relation=AlignmentRelation.RELATED_TO)

        assert alignment.local_name == "PhysicalTask"


# %% namespaces

CORE_NAMESPACE = "http://example.org/ontology/core#"


class TestNamespaceTerms:
    """
    A namespace builds the identifiers of the terms declared under it.
    """

    def test_class_term_is_the_name_appended_to_the_namespace(self) -> None:
        namespace = Namespace(iri=CORE_NAMESPACE)

        assert namespace.term("ActionPhase") == CORE_NAMESPACE + "ActionPhase"

    @pytest.mark.parametrize(
        ("field_name", "term"),
        [
            ("duration", "duration"),
            ("scientific_status", "scientificStatus"),
            ("event_on_role", "eventOnRole"),
            ("has_condition_requirement", "hasConditionRequirement"),
        ],
    )
    def test_property_term_is_the_field_name_in_lower_camel_case(
        self, field_name: str, term: str
    ) -> None:
        namespace = Namespace(iri=CORE_NAMESPACE)

        assert namespace.property_term(field_name) == CORE_NAMESPACE + term

    def test_terms_can_be_recognised_by_their_local_name(self) -> None:
        namespace = Namespace(iri=CORE_NAMESPACE)

        assert Term(iri=namespace.term("ActionPhase")).local_name == "ActionPhase"

    @pytest.mark.parametrize("iri", ["http://example.org/core", "core", ""])
    def test_namespace_must_end_where_a_term_can_begin(self, iri: str) -> None:
        with pytest.raises(InvalidNamespace) as rejection:
            Namespace(iri=iri)

        assert rejection.value.iri == iri

    def test_slash_namespace_is_accepted(self) -> None:
        namespace = Namespace(iri="http://example.org/ontology/")

        assert namespace.term("Pouring") == "http://example.org/ontology/Pouring"
