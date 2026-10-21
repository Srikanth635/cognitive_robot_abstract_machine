"""
Tests of the situation features a knowledge structure accepts: the functional fields
whose values define its variant classes, read from the ontology rather than listed.
"""

from __future__ import annotations

from krrood.entity_query_language.verbalization.rendering.source_documentation import (
    _attribute_docstrings,
)

from llm_reasoner.knowledge.ontologies.action_perspectives import action_core
from llm_reasoner.knowledge.ontologies.predictive_manipulation import core, pouring
from llm_reasoner.knowledge.ontologies.situation import SituationFeatures
from llm_reasoner.knowledge.ontologies.terms import PropertyRestriction, Quantifier


def features_by_field(root_type: type) -> dict:
    """
    :return: The situation features of a root class, by field name.
    """
    return {
        feature.field_name: feature for feature in SituationFeatures().of(root_type)
    }


class TestSituationFeatures:
    def test_a_value_defining_a_variant_makes_its_field_a_feature(self):
        material = features_by_field(pouring.PouringPhaseStructure)[
            "has_material_regime"
        ]
        assert material.values == tuple(action_core.MaterialRegime)
        assert (
            material.variants[action_core.MaterialRegime.GRANULAR]
            is pouring.GranularPouringStructure
        )

    def test_every_variant_is_defined_by_its_value(self):
        for feature in SituationFeatures().of(pouring.PouringPhaseStructure):
            for value, variant in feature.variants.items():
                assert (
                    PropertyRestriction(
                        property_iri=feature.property_iri,
                        quantifier=Quantifier.VALUE,
                        value=value.value,
                    )
                    in variant.term.definition.operands
                )

    def test_every_field_defining_a_pouring_variant_is_a_feature(self):
        assert set(features_by_field(pouring.PouringPhaseStructure)) == {
            "has_material_regime",
            "has_termination_reason",
        }

    def test_a_feature_states_the_meaning_of_its_field(self):
        material = features_by_field(pouring.PouringPhaseStructure)[
            "has_material_regime"
        ]
        assert (
            material.meaning
            == _attribute_docstrings(pouring.PouringPhaseStructure)[
                "has_material_regime"
            ]
        )

    def test_a_structure_without_variants_by_value_has_no_features(self):
        assert SituationFeatures().of(core.ActionPhaseStructure) == []

    def test_a_feature_reads_its_value_from_a_structure(self):
        material = features_by_field(pouring.PouringPhaseStructure)[
            "has_material_regime"
        ]
        structure = pouring.PouringPhaseStructure(
            has_material_regime=action_core.MaterialRegime.COHESIVE
        )
        assert material.value_in(structure) is action_core.MaterialRegime.COHESIVE

    def test_a_feature_left_unset_reads_as_nothing(self):
        material = features_by_field(pouring.PouringPhaseStructure)[
            "has_material_regime"
        ]
        assert material.value_in(pouring.PouringPhaseStructure()) is None
