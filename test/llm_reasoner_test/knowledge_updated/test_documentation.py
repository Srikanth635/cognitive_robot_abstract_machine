from __future__ import annotations

from dataclasses import fields

import pytest
from llm_reasoner.knowledge_updated.ontological.documentation import (
    DocumentationReader,
    DocumentationSourceUnavailable,
)
from llm_reasoner.knowledge_updated.ontological.references import (
    Alignment,
    ConceptReference,
    Term,
)
from llm_reasoner.knowledge_updated.ontological.vocabulary import (
    AlignmentRelation,
    ConceptLevel,
    EvidenceStatus,
    PropertyKind,
    Rigidity,
    Severity,
)

from .documented_examples import (
    Colour,
    ElectricVehicle,
    Outer,
    StrayString,
    UndocumentedVehicle,
    Vehicle,
    build_local_class,
)

# %% attributes


class TestAttributeDocumentation:
    """
    The string directly under an attribute is that attribute's documentation.
    """

    def test_field_documentation_is_read(self) -> None:
        reader = DocumentationReader()

        assert (
            reader.read_attribute(Vehicle, "wheels")
            == "How many wheels touch the ground."
        )

    def test_enum_member_documentation_is_read(self) -> None:
        reader = DocumentationReader()

        assert reader.read_attribute(Colour, "RED") == "The colour of blood."

    def test_single_line_string_is_read_like_an_indented_block(self) -> None:
        reader = DocumentationReader()

        assert reader.read_attribute(Colour, "GREEN") == "The colour of grass."

    def test_multi_paragraph_documentation_keeps_its_paragraphs_and_loses_indentation(
        self,
    ) -> None:
        reader = DocumentationReader()

        assert reader.read_attribute(Vehicle, "passengers") == (
            "Names of the people aboard.\n\nListed in boarding order."
        )

    def test_attribute_without_a_string_has_no_documentation(self) -> None:
        reader = DocumentationReader()

        assert reader.read_attribute(Vehicle, "registration") is None
        assert reader.read_attribute(Colour, "BLUE") is None

    def test_string_after_a_method_documents_nothing(self) -> None:
        reader = DocumentationReader()

        assert reader.read_attribute(StrayString, "method") is None

    def test_repeated_reads_agree(self) -> None:
        reader = DocumentationReader()

        first = reader.read_attribute(Vehicle, "wheels")

        assert reader.read_attribute(Vehicle, "wheels") == first


# %% inheritance


class TestInheritedDocumentation:
    """
    A subclass inherits the documentation of its bases, and its own text wins.
    """

    def test_inherited_field_uses_the_base_documentation(self) -> None:
        reader = DocumentationReader()

        assert reader.read_attribute(ElectricVehicle, "passengers") == (
            reader.read_attribute(Vehicle, "passengers")
        )

    def test_redeclared_field_uses_the_subclass_documentation(self) -> None:
        reader = DocumentationReader()

        assert reader.read_attribute(ElectricVehicle, "wheels") != (
            reader.read_attribute(Vehicle, "wheels")
        )
        assert (
            reader.read_attribute(ElectricVehicle, "wheels")
            == "How many wheels touch the ground on this model."
        )


# %% classes


class TestClassDocumentation:
    """
    A class's documentation is the docstring it declares.
    """

    def test_declared_docstring_is_read(self) -> None:
        reader = DocumentationReader()

        assert reader.read_class(Vehicle) == "Something that moves people."

    def test_dataclass_without_a_docstring_has_no_documentation(self) -> None:
        """
        ``dataclass`` fills ``__doc__`` with a signature, which is not documentation.
        """
        reader = DocumentationReader()

        assert UndocumentedVehicle.__doc__ is not None
        assert reader.read_class(UndocumentedVehicle) is None


# %% nesting


class TestNestedClasses:
    """
    Classes declared inside other scopes are found by their qualified name.
    """

    def test_class_nested_in_a_class(self) -> None:
        reader = DocumentationReader()

        assert (
            reader.read_attribute(Outer.Inner, "depth") == "How deep the nesting goes."
        )

    def test_class_declared_in_a_function(self) -> None:
        reader = DocumentationReader()
        local_class = build_local_class()

        assert reader.read_attribute(local_class, "size") == "How large it is."


# %% completeness


class TestUndocumentedAttributes:
    """
    Attributes lacking documentation can be listed.
    """

    def test_undocumented_enum_member_is_listed(self) -> None:
        reader = DocumentationReader()

        assert reader.find_undocumented_members(Colour) == ("BLUE",)

    def test_undocumented_field_is_listed(self) -> None:
        reader = DocumentationReader()

        assert reader.find_undocumented_fields(Vehicle) == ("registration",)

    def test_inherited_fields_are_checked_too(self) -> None:
        reader = DocumentationReader()

        assert reader.find_undocumented_fields(ElectricVehicle) == ("registration",)

    def test_fully_documented_class_lists_nothing(self) -> None:
        reader = DocumentationReader()

        assert reader.find_undocumented_fields(Outer.Inner) == ()


# %% unavailable source


class TestUnavailableSource:
    """
    A class with no source file to read is reported, never treated as undocumented.
    """

    def test_class_of_an_unknown_module_is_reported(self) -> None:
        dynamic = type("Dynamic", (), {"__module__": "no.such.module", "anything": 1})
        reader = DocumentationReader()

        with pytest.raises(DocumentationSourceUnavailable) as unavailable:
            reader.read_attribute(dynamic, "anything")

        assert unavailable.value.owner is dynamic

    def test_class_of_a_built_in_module_is_reported(self) -> None:
        reader = DocumentationReader()

        with pytest.raises(DocumentationSourceUnavailable):
            reader.read_class(int)


# %% the ontological vocabulary


VOCABULARY_ENUMERATIONS = (
    Rigidity,
    ConceptLevel,
    PropertyKind,
    Severity,
    AlignmentRelation,
    EvidenceStatus,
)
"""
Every enumeration of the ontological vocabulary.
"""

VOCABULARY_DATACLASSES = (Term, ConceptReference, Alignment)
"""
Every dataclass of the ontological vocabulary.
"""


class TestOntologicalVocabularyIsDocumented:
    """
    Every member and field of the ontological vocabulary carries documentation.
    """

    @pytest.mark.parametrize("enumeration", VOCABULARY_ENUMERATIONS)
    def test_every_enum_member_is_documented(self, enumeration: type) -> None:
        reader = DocumentationReader()

        assert reader.find_undocumented_members(enumeration) == ()

    @pytest.mark.parametrize("dataclass_type", VOCABULARY_DATACLASSES)
    def test_every_field_is_documented(self, dataclass_type: type) -> None:
        reader = DocumentationReader()

        assert reader.find_undocumented_fields(dataclass_type) == ()

    def test_dataclasses_under_test_have_fields(self) -> None:
        assert all(fields(dataclass_type) for dataclass_type in VOCABULARY_DATACLASSES)
