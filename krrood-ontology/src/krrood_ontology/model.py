"""Generation decisions derived from the asserted ontology graph."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PropertyKind(str, Enum):
    """Distinguish references to objects from literal values."""

    OBJECT = "object"
    DATATYPE = "datatype"


class ExpressionKind(str, Enum):
    """Positive class-expression forms supported by scoped query translation."""

    CLASS = "class"
    INTERSECTION = "intersection"
    UNION = "union"
    ENUMERATION = "enumeration"
    HAS_VALUE = "has_value"
    SOME = "some"
    ALL = "all"
    MIN = "min"
    MAX = "max"
    EXACT = "exact"
    DATATYPE = "datatype"


@dataclass(frozen=True)
class LiteralDefinition:
    """An RDF field literal retained with its datatype and lexical form."""

    lexical: str
    datatype: str


@dataclass(frozen=True)
class ClassExpression:
    """A bounded expression tree for metadata and EQL condition construction."""

    kind: ExpressionKind
    iri: str | None = None
    property_iri: str | None = None
    literal: LiteralDefinition | None = None
    operands: tuple[ClassExpression, ...] = ()
    individuals: tuple[str, ...] = ()
    cardinality: int | None = None


@dataclass(frozen=True)
class ExpressionDefinition:
    """A named class's anonymous equivalent-class query definition."""

    class_iri: str
    expression: ClassExpression


@dataclass(frozen=True)
class PropertyDefinition:
    """A property whose explicit domain and range have a supported mapping."""

    iri: str
    name: str
    domain_iri: str
    domain_name: str
    range_iri: str
    range_type: str
    kind: PropertyKind
    container: str = "list"
    constructor: bool = True


@dataclass(frozen=True)
class DescriptorDefinition:
    """A KRROOD descriptor declaration and its supported characteristics."""

    property: PropertyDefinition
    name: str
    parent_name: str | None = None
    parent_iri: str | None = None
    inverse_name: str | None = None
    inverse_iri: str | None = None
    transitive: bool = False
    symmetric: bool = False


@dataclass(frozen=True)
class ClassDefinition:
    """A named class with its parent and directly owned properties."""

    iri: str
    name: str
    parent_name: str | None
    properties: tuple[PropertyDefinition, ...]
    parent_names: tuple[str, ...] = ()
    asserted_parent_iris: tuple[str, ...] = ()


@dataclass(frozen=True)
class CompilationDiagnostic:
    """A construct excluded from the generated structural model."""

    code: str
    message: str
    iri: str | None = None


@dataclass(frozen=True)
class ModelDefinition:
    """An ordered generation plan, not a second ontology store."""

    classes: tuple[ClassDefinition, ...]
    properties: tuple[PropertyDefinition, ...]
    diagnostics: tuple[CompilationDiagnostic, ...]
    descriptors: tuple[DescriptorDefinition, ...] = ()
    expressions: tuple[ExpressionDefinition, ...] = ()
    property_bindings: tuple[PropertyDefinition, ...] = ()
    binding_policy: str = "explicit"
    schema_axioms: tuple[dict, ...] = ()
    scoped_semantics: bool = False
    class_aliases: tuple[tuple[str, tuple[str, ...]], ...] = ()
    root_properties: tuple[PropertyDefinition, ...] = ()
    reflexive_properties: tuple[str, ...] = ()


class OntologyTranslationError(ValueError):
    """Report unsupported or ambiguous ontology-to-Python mappings."""

    def __init__(self, diagnostics: tuple[CompilationDiagnostic, ...]) -> None:
        self.diagnostics = diagnostics
        super().__init__(
            "\n".join(
                f"{item.code}: {item.message}" + (f" [{item.iri}]" if item.iri else "")
                for item in diagnostics
            )
        )
