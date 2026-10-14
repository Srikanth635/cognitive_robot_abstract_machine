from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from krrood.symbol_graph.symbol_graph import Symbol
from llm_reasoner.knowledge_updated.ontological.documentation import (
    DocumentationReader,
)
from llm_reasoner.knowledge_updated.ontological.entities import (
    LOCAL_ID_LENGTH,
    KnowledgeEntity,
)
from llm_reasoner.knowledge_updated.ontological.properties import (
    PropertyMeta,
    attribute,
    relation,
)

# %% fixtures


@dataclass(eq=False)
class Step(KnowledgeEntity):
    """
    A described entity with a property of each kind.
    """

    duration: Optional[float] = attribute(functional=True)
    """
    How long the step lasts.
    """

    next_steps: list[Step] = relation()
    """
    The steps that may follow.
    """


# %% identity


class TestEntityIdentity:
    """
    Every entity carries an identifier that survives serialisation.
    """

    def test_each_entity_receives_its_own_identifier(self) -> None:
        assert KnowledgeEntity().local_id != KnowledgeEntity().local_id

    def test_generated_identifier_has_the_declared_length(self) -> None:
        assert len(KnowledgeEntity().local_id) == LOCAL_ID_LENGTH

    def test_supplied_identifier_is_kept(self) -> None:
        assert KnowledgeEntity(local_id="reach").local_id == "reach"

    def test_entities_with_equal_content_are_still_different_entities(self) -> None:
        first = KnowledgeEntity(local_id="same", label="same")
        second = KnowledgeEntity(local_id="same", label="same")

        assert first != second
        assert len({first, second}) == 2


# %% label


class TestEntityLabel:
    """
    Every entity has a human-readable label.
    """

    def test_label_is_empty_unless_given(self) -> None:
        assert KnowledgeEntity().label == ""

    def test_supplied_label_is_kept(self) -> None:
        assert KnowledgeEntity(label="reach for the handle").label == (
            "reach for the handle"
        )


# %% queryability


class TestEntityIsQueryable:
    """
    Entities are symbols, so the entity query language can range over them.
    """

    def test_entity_is_a_symbol(self) -> None:
        assert isinstance(KnowledgeEntity(), Symbol)


# %% subclassing


class TestEntitySubclasses:
    """
    A subclass adds its own properties to the identity fields.
    """

    def test_subclass_keeps_identity_and_adds_properties(self) -> None:
        step = Step(local_id="lift", label="lift", duration=1.5)

        assert (step.local_id, step.label, step.duration) == ("lift", "lift", 1.5)
        assert step.next_steps == []

    def test_only_added_fields_are_described_properties(self) -> None:
        assert list(PropertyMeta.declared_on(Step)) == ["duration", "next_steps"]


# %% documentation


class TestEntityVocabularyIsDocumented:
    """
    Every field of the base entity carries documentation.
    """

    def test_every_field_is_documented(self) -> None:
        assert DocumentationReader().find_undocumented_fields(KnowledgeEntity) == ()

    def test_class_is_documented(self) -> None:
        assert DocumentationReader().read_class(KnowledgeEntity) is not None
