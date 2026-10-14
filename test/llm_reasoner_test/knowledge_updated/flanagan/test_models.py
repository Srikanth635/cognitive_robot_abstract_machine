from __future__ import annotations

import pytest
from llm_reasoner.knowledge_updated.microworlds.flanagan.alignments import (
    DUL,
    subclass_of,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.models import (
    EstimateRepresentation,
    ForwardModel,
    InternalModel,
    InverseModel,
    ObjectRepresentation,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.parameters import Estimate
from llm_reasoner.knowledge_updated.microworlds.flanagan.ontology import CORE, FLANAGAN
from llm_reasoner.knowledge_updated.ontological.documentation import (
    DocumentationReader,
)
from llm_reasoner.knowledge_updated.ontological.entities import KnowledgeEntity
from llm_reasoner.knowledge_updated.ontological.properties import PropertyMeta
from llm_reasoner.knowledge_updated.ontological.ranges import RangeResolver
from llm_reasoner.knowledge_updated.ontological.validation import (
    Check,
    OntologyValidator,
)

# %% fixtures

ALL_MODELS = (
    InternalModel,
    ForwardModel,
    InverseModel,
    EstimateRepresentation,
    ObjectRepresentation,
)
"""
Every concept of the internal-model vocabulary.
"""

# %% taxonomy


class TestInternalModelTaxonomy:
    """
    An internal model predicts or inverts, and a representation supplies content to it.
    """

    @pytest.mark.parametrize("cls", ALL_MODELS)
    def test_concept_is_a_core_entity(self, cls: type) -> None:
        assert issubclass(cls, KnowledgeEntity)
        assert FLANAGAN.find_concept(cls).module == CORE.name

    def test_only_the_grouping_concepts_are_abstract(self) -> None:
        abstract = {cls for cls in ALL_MODELS if FLANAGAN.find_concept(cls).abstract}

        assert abstract == {InternalModel, EstimateRepresentation}

    @pytest.mark.parametrize(
        ("child", "parent"),
        [
            (ForwardModel, InternalModel),
            (InverseModel, InternalModel),
            (EstimateRepresentation, InternalModel),
            (ObjectRepresentation, EstimateRepresentation),
        ],
    )
    def test_specialisations(self, child: type, parent: type) -> None:
        assert issubclass(child, parent)

    def test_internal_models_are_aligned_as_descriptions(self) -> None:
        assert FLANAGAN.find_concept(InternalModel).alignments == (
            subclass_of(DUL, "Description"),
        )

    def test_prediction_and_inversion_are_different_functions(self) -> None:
        axioms = FLANAGAN.find_concept(ForwardModel).axioms

        assert InverseModel.__name__ in axioms.disjoint_with

    def test_a_model_and_the_content_it_uses_are_not_presumed_disjoint(self) -> None:
        axioms = FLANAGAN.find_concept(EstimateRepresentation).axioms

        assert {ForwardModel.__name__, InverseModel.__name__} <= set(
            axioms.may_overlap_with
        )

    def test_the_kinds_of_model_raise_no_sibling_warning(self) -> None:
        report = OntologyValidator(FLANAGAN).validate(CORE.name)
        flagged = {v.subject for v in report.of(Check.SIBLINGS_OVERLAP)}

        assert flagged.isdisjoint({cls.__name__ for cls in ALL_MODELS})


# %% properties


class TestRepresentationContent:
    """
    A representation owns the estimates it holds.
    """

    def test_representation_holds_estimates(self) -> None:
        described = PropertyMeta.declared_on(EstimateRepresentation)["holds_estimate"]

        assert described.functional is False
        assert RangeResolver().resolve(
            EstimateRepresentation, "holds_estimate"
        ).element_types == (Estimate,)

    def test_the_models_that_predict_or_invert_hold_no_properties(self) -> None:
        assert PropertyMeta.declared_on(ForwardModel) == {}
        assert PropertyMeta.declared_on(InverseModel) == {}


# %% documentation


class TestModelsAreDocumented:
    """
    Every model concept and property says what it means.
    """

    @pytest.mark.parametrize("cls", ALL_MODELS)
    def test_concept_and_properties_are_documented(self, cls: type) -> None:
        reader = DocumentationReader()

        assert reader.read_class(cls) is not None
        assert reader.find_undocumented_fields(cls) == ()
