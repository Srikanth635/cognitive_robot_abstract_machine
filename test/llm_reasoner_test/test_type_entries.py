"""
Tests that the model is told every field and requirement of a fragment type it may
write: a type adds to the bases the prompt describes, and takes in the fields and
requirements of bases the prompt does not describe.
"""

from __future__ import annotations

import json
from dataclasses import fields

import pytest

from llm_reasoner.knowledge.generation.prompts import (
    GenerationPrompt,
    GenerationSection,
    TypeKey,
)
from llm_reasoner.knowledge.generation.schema import GenerationSchema
from llm_reasoner.knowledge.ontologies.action_perspectives import (
    action_core,
    action_frame,
    failures,
    image_schemas,
)

from .test_generation import bodies

# %% helpers


def entries_told(root: type) -> dict[str, dict]:
    """
    :return: The type entries the system prompt for a root gives, by type name.
    """
    system = GenerationPrompt().system(GenerationSchema.for_root(root))
    (types,) = bodies(system, GenerationSection.TYPES)
    return {entry[TypeKey.NAME]: entry for entry in json.loads(types)}


def field_names(entry: dict) -> set[str]:
    """
    :return: The names of the fields an entry lists.
    """
    return {field_entry["name"] for field_entry in entry[TypeKey.FIELDS]}


QUANTITIES = [
    (action_core.PouringFrame, action_core.ParameterDimension),
    (image_schemas.SchemaStereotype, image_schemas.DecidingQuantity),
    (failures.FailureStereotype, failures.Indicator),
]
"""
Each kind of slot quantity with the root of the hypotheses it is written in.
"""


# %% a base the prompt does not describe


class TestUndescribedBase:
    @pytest.mark.parametrize("root, kind", QUANTITIES)
    def test_its_fields_are_listed_on_the_type(self, root, kind):
        entry = entries_told(root)[kind.__name__]
        assert {declared.name for declared in fields(action_core.SlotQuantity)} - {
            "label"
        } <= field_names(entry)

    @pytest.mark.parametrize("root, kind", QUANTITIES)
    def test_its_requirements_are_listed_on_the_type(self, root, kind):
        entry = entries_told(root)[kind.__name__]
        assert len(entry[TypeKey.REQUIRES]) >= len(
            action_core.SlotQuantity.term.restrictions
        )

    @pytest.mark.parametrize("root, kind", QUANTITIES)
    def test_the_type_is_not_said_to_be_a_kind_of_it(self, root, kind):
        entry = entries_told(root)[kind.__name__]
        assert action_core.SlotQuantity.__name__ not in entry[TypeKey.KIND_OF]


# %% a base the prompt describes


class TestDescribedBase:
    def test_a_type_adds_only_its_own_fields_to_it(self):
        entries = entries_told(action_frame.FrameStereotype)
        assert action_core.SlotCondition.__name__ in entries
        feature = entries[action_frame.FeatureCondition.__name__]
        assert action_core.SlotCondition.__name__ in feature[TypeKey.KIND_OF]
        assert field_names(feature) == {"condition_requires_part"}
