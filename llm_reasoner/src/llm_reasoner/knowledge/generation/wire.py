"""
Typed validation for generated flat hypothesis graphs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from llm_reasoner.exceptions import LLMReasonerError
from llm_reasoner.knowledge.generation.schema import (
    FieldKind,
    FieldSchema,
    GenerationSchema,
)

# %% Wire declarations


class WireFormatError(LLMReasonerError):
    """
    A generated object does not satisfy its derived flat-wire contract.
    """


@dataclass(frozen=True)
class FragmentField:
    """
    One schema-declared value carried by a fragment declaration.
    """

    name: str
    """
    Ontological property name.
    """

    value: Any
    """
    Wire value before ontology references and enums are resolved.
    """


@dataclass(frozen=True)
class FragmentDeclaration:
    """
    A validated declaration from the generated flat graph.
    """

    identifier: str
    """
    Response-scoped identifier used only to wire declarations together.
    """

    type_name: str
    """
    Registered ontology class selected for the fragment.
    """

    description: str = ""
    """
    Human-readable description supplied through the wire ``label`` field.
    """

    fields: tuple[FragmentField, ...] = ()
    """
    Schema-declared field values in response order.
    """

    def value_of(self, name: str) -> Any:
        """
        Return a field value, or ``None`` when the field was omitted.
        """
        for fragment_field in self.fields:
            if fragment_field.name == name:
                return fragment_field.value
        return None


@dataclass(frozen=True)
class HypothesisDeclaration:
    """
    A validated set of declarations with one distinguished root.
    """

    fragments: tuple[FragmentDeclaration, ...]
    """
    Every declaration in response order.
    """

    root: FragmentDeclaration
    """
    The single declaration whose type equals the schema root.
    """


# %% Contract validation


@dataclass(frozen=True)
class GenerationWireDecoder:
    """
    Validate untyped model output against a derived generation schema.
    """

    schema: GenerationSchema
    """
    Contract that determines legal types, fields, and value shapes.
    """

    error_type: type[LLMReasonerError] = WireFormatError
    """
    Exception type used to report contract violations to the caller.
    """

    def decode(self, payload: Mapping[str, Any]) -> HypothesisDeclaration:
        """
        Validate and return typed declarations for a flat graph payload.
        """
        unexpected_top_level = set(payload) - {"fragments"}
        if unexpected_top_level:
            raise self.error_type(
                "Unknown top-level fields: " + ", ".join(sorted(unexpected_top_level))
            )

        entries = payload.get("fragments")
        if not isinstance(entries, list) or not entries:
            raise self.error_type(
                'Expected a non-empty "fragments" array in the object.'
            )

        declarations: list[FragmentDeclaration] = []
        identifiers: set[str] = set()
        for position, entry in enumerate(entries):
            declaration = self._decode_fragment(entry, position)
            if declaration.identifier in identifiers:
                raise self.error_type(
                    f"Duplicate fragment id: {declaration.identifier}"
                )
            identifiers.add(declaration.identifier)
            declarations.append(declaration)

        roots = [
            declaration
            for declaration in declarations
            if declaration.type_name == self.schema.root
        ]
        if len(roots) != 1:
            raise self.error_type(
                f"Expected exactly one {self.schema.root} fragment, "
                f"found {len(roots)}."
            )
        return HypothesisDeclaration(tuple(declarations), roots[0])

    def _decode_fragment(self, entry: Any, position: int) -> FragmentDeclaration:
        """
        Validate one fragment declaration.
        """
        if not isinstance(entry, Mapping):
            raise self.error_type(
                f"Fragment at position {position} must be a JSON object."
            )

        identifier = entry.get("id")
        type_name = entry.get("type")
        if not isinstance(identifier, str) or not identifier.strip():
            raise self.error_type(
                f"Fragment at position {position} needs a non-empty string id."
            )
        if not isinstance(type_name, str) or not type_name.strip():
            raise self.error_type(
                f"Fragment {identifier!r} needs a non-empty string type."
            )

        class_schema = self.schema.get(type_name)
        if class_schema is None:
            raise WireFormatError(
                f"Unknown type {type_name!r}. Use only the listed types."
            )
        description = entry.get("label", "")
        if not isinstance(description, str):
            raise self.error_type(f"{identifier}.label must be a string.")

        fields_by_name = {
            field_schema.name: field_schema for field_schema in class_schema.fields
        }
        unexpected_fields = set(entry) - {
            "id",
            "type",
            "label",
            *fields_by_name,
        }
        if unexpected_fields:
            raise self.error_type(
                f"{type_name} has no field "
                + ", ".join(repr(name) for name in sorted(unexpected_fields))
                + "."
            )

        fields: list[FragmentField] = []
        for name, field_schema in fields_by_name.items():
            if name not in entry:
                if field_schema.required:
                    raise self.error_type(f"{identifier}.{name} is required.")
                continue
            value = entry[name]
            self._validate_field(identifier, field_schema, value)
            fields.append(FragmentField(name=name, value=value))
        return FragmentDeclaration(
            identifier=identifier,
            type_name=type_name,
            description=description,
            fields=tuple(fields),
        )

    def _validate_field(
        self,
        owner_identifier: str,
        field_schema: FieldSchema,
        value: Any,
    ) -> None:
        """
        Validate collection shape, cardinality, and scalar vocabulary.
        """
        field_identifier = f"{owner_identifier}.{field_schema.name}"
        if field_schema.collection:
            if not isinstance(value, list):
                raise self.error_type(f"{field_identifier} must be an array.")
            if len(value) < field_schema.required:
                raise self.error_type(
                    f"{field_identifier} needs at least "
                    f"{field_schema.required} value(s)."
                )
            for item in value:
                self._validate_scalar(field_identifier, field_schema, item)
            return
        if isinstance(value, list):
            raise self.error_type(f"{field_identifier} must not be an array.")
        if value is None:
            raise self.error_type(
                f"{field_identifier} must be omitted instead of null."
            )
        self._validate_scalar(field_identifier, field_schema, value)

    def _validate_scalar(
        self,
        field_identifier: str,
        field_schema: FieldSchema,
        value: Any,
    ) -> None:
        """
        Validate one non-collection wire value.
        """
        if field_schema.kind in {FieldKind.FRAGMENT, FieldKind.CONCEPT}:
            if not isinstance(value, str) or not value.strip():
                raise self.error_type(f"{field_identifier} must be a non-empty string.")
        elif field_schema.type_name == "bool" and not isinstance(value, bool):
            raise self.error_type(f"{field_identifier} must be true or false.")
        elif field_schema.type_name == "int" and (
            not isinstance(value, int) or isinstance(value, bool)
        ):
            raise self.error_type(f"{field_identifier} must be an integer.")
        elif field_schema.type_name == "float" and (
            not isinstance(value, (int, float)) or isinstance(value, bool)
        ):
            raise self.error_type(f"{field_identifier} must be a number.")
        elif field_schema.type_name == "str" and not isinstance(value, str):
            raise self.error_type(f"{field_identifier} must be a string.")

        if (
            field_schema.options
            and field_schema.kind in {FieldKind.VALUE, FieldKind.CONCEPT}
            and value not in field_schema.options
            and value not in field_schema.wire_value_aliases
        ):
            raise self.error_type(
                f"{field_identifier} must be one of: "
                + " | ".join(field_schema.options)
            )


# %% Canonical hypothesis identity


@dataclass
class CanonicalFragmentIdentityAllocator:
    """
    Allocate deterministic record-local identities from ontology types.
    """

    counts: dict[str, int] = field(default_factory=dict, init=False)
    """
    Number of identities already allocated for each canonical stem.
    """

    def allocate(self, type_name: str) -> str:
        """
        Return the next kebab-case identity for an ontology type name.
        """
        separated = re.sub(r"([a-z0-9])([A-Z])", r"\1-\2", type_name)
        stem = re.sub(r"[^A-Za-z0-9]+", "-", separated).strip("-").lower()
        occurrence = self.counts.get(stem, 0) + 1
        self.counts[stem] = occurrence
        if occurrence == 1:
            return stem
        return f"{stem}-{occurrence}"


__all__ = [
    "CanonicalFragmentIdentityAllocator",
    "FragmentDeclaration",
    "FragmentField",
    "GenerationWireDecoder",
    "HypothesisDeclaration",
    "WireFormatError",
]
