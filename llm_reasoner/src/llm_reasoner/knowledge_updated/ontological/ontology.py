"""
The ontology of a microworld: its modules, and the concepts and value partitions they
declare.

A microworld creates one :class:`Ontology`. Each of its modules is a scope with its own
namespace, version and dependencies, and a module's decorators record the classes written
under it. Nothing is stored globally, so two ontologies never see each other's
declarations.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, TypeVar

from llm_reasoner.exceptions import LLMReasonerError
from llm_reasoner.knowledge_updated.ontological.axioms import ClassAxioms
from llm_reasoner.knowledge_updated.ontological.concepts import ConceptMeta
from llm_reasoner.knowledge_updated.ontological.properties import PropertyMeta
from llm_reasoner.knowledge_updated.ontological.references import Alignment, Namespace
from llm_reasoner.knowledge_updated.ontological.vocabulary import (
    ConceptLevel,
    EvidenceStatus,
    Rigidity,
)

T = TypeVar("T", bound=type)

# %% refused declarations


class DeclarationFault(StrEnum):
    """
    Why the ontology refuses a declaration.
    """

    DUPLICATE_IRI = "duplicate_iri"
    """
    Another concept or value partition already has this identifier.
    """

    CLASS_ALREADY_DECLARED = "class_already_declared"
    """
    The class is already declared in this ontology.
    """

    BASE_FROM_UNDECLARED_DEPENDENCY = "base_from_undeclared_dependency"
    """
    The class specialises a concept of a module that the declaring module does not depend
    on.
    """

    NOT_AN_ENUMERATION = "not_an_enumeration"
    """
    A value partition is declared on something that is not an enumeration.
    """


@dataclass
class InvalidDeclaration(LLMReasonerError):
    """
    The ontology refuses to record a declaration.
    """

    subject: type
    """
    The class being declared.
    """

    reason: DeclarationFault
    """
    Why it is refused.
    """

    def __str__(self) -> str:
        return f"cannot declare {self.subject.__qualname__}: {self.reason.value}"


@dataclass
class DuplicateModule(LLMReasonerError):
    """
    A module name is used twice in one ontology.
    """

    name: str
    """
    The name that is already taken.
    """

    def __str__(self) -> str:
        return f"the ontology already has a module named '{self.name}'"


# %% recorded declarations


@dataclass(frozen=True)
class ValuePartitionMeta:
    """
    An enumeration recorded as a closed set of alternatives.
    """

    enumeration: type[Enum]
    """
    The enumeration.
    """

    iri: str
    """
    The partition's identifier.
    """

    module: str
    """
    Name of the module that declares the partition.
    """


@dataclass(frozen=True)
class PropertyDeclaration:
    """
    A property as written in the body of the concept that introduces it.
    """

    owner: type
    """
    The concept whose body declares the field.
    """

    field_name: str
    """
    The name of the field.
    """

    meta: PropertyMeta
    """
    The property's characteristics.
    """

    iri: str
    """
    The property's identifier.
    """

    module: str
    """
    Name of the module that declares the owning concept.
    """


# %% modules


@dataclass(eq=False)
class OntologyModule:
    """
    A part of an ontology with its own namespace, version and dependencies.
    """

    ontology: Ontology = field(repr=False)
    """
    The ontology the module belongs to.
    """

    name: str
    """
    The module's name, unique within its ontology.
    """

    version: str
    """
    The module's version. A change marks knowledge generated under the module as stale.
    """

    namespace: Namespace
    """
    The prefix of the identifiers the module declares.
    """

    depends_on: tuple[OntologyModule, ...] = ()
    """
    The modules whose concepts this module may specialise.
    """

    def transitive_dependencies(self) -> tuple[OntologyModule, ...]:
        """
        Return every module this one depends on, directly or not, dependencies first.
        """
        ordered: list[OntologyModule] = []
        for dependency in self.depends_on:
            for module in (*dependency.transitive_dependencies(), dependency):
                if module not in ordered:
                    ordered.append(module)
        return tuple(ordered)

    def concept(
        self,
        *,
        iri: Optional[str] = None,
        rigidity: Rigidity = Rigidity.RIGID,
        level: ConceptLevel = ConceptLevel.DESCRIPTION,
        alignments: Sequence[Alignment] = (),
        abstract: bool = False,
        disjoint_with: Iterable[str] = (),
        disjoint_siblings: bool = False,
        may_overlap_with: Iterable[str] = (),
        fixed_values: Optional[Mapping[str, Any]] = None,
        allowed_values: Optional[Mapping[str, Iterable[Any]]] = None,
        role_taker: Optional[type] = None,
        status: Optional[EvidenceStatus] = None,
        claims: Iterable[str] = (),
    ) -> Callable[[T], T]:
        """
        Declare a class as a concept of this module, leaving the class unchanged.

        :param iri: The concept's identifier, derived from the namespace and the class
            name when omitted.
        :raises InvalidConcept: when the class breaks a rule that concerns it alone.
        :raises InvalidAxiom: when an axiom cannot hold of the class.
        :raises InvalidDeclaration: when the ontology refuses the declaration.
        """

        def declare(cls: T) -> T:
            self.ontology.add_concept(
                ConceptMeta(
                    cls=cls,
                    iri=iri or self.namespace.term(cls.__name__),
                    module=self.name,
                    rigidity=rigidity,
                    level=level,
                    alignments=tuple(alignments),
                    abstract=abstract,
                    axioms=ClassAxioms.of(
                        disjoint_with=disjoint_with,
                        disjoint_siblings=disjoint_siblings,
                        may_overlap_with=may_overlap_with,
                        fixed_values=fixed_values,
                        allowed_values=allowed_values,
                    ),
                    role_taker=role_taker,
                    status=status,
                    claims=tuple(claims),
                )
            )
            return cls

        return declare

    def value_partition(self, *, iri: Optional[str] = None) -> Callable[[T], T]:
        """
        Declare an enumeration as a closed set of alternatives of this module.

        :param iri: The partition's identifier, derived from the namespace and the class
            name when omitted.
        :raises InvalidDeclaration: when the class is not an enumeration or the ontology
            refuses the declaration.
        """

        def declare(enumeration: T) -> T:
            if not issubclass(enumeration, Enum):
                raise InvalidDeclaration(
                    enumeration, DeclarationFault.NOT_AN_ENUMERATION
                )
            self.ontology.add_value_partition(
                ValuePartitionMeta(
                    enumeration=enumeration,
                    iri=iri or self.namespace.term(enumeration.__name__),
                    module=self.name,
                )
            )
            return enumeration

        return declare


# %% the ontology


@dataclass
class Ontology:
    """
    The modules of a microworld and everything they declare.
    """

    name: str
    """
    The microworld's name.
    """

    _modules: dict[str, OntologyModule] = field(
        default_factory=dict, init=False, repr=False
    )
    """
    Modules by name, in declaration order.
    """

    _concepts: dict[type, ConceptMeta] = field(
        default_factory=dict, init=False, repr=False
    )
    """
    Concepts by class, in declaration order.
    """

    _partitions: dict[type, ValuePartitionMeta] = field(
        default_factory=dict, init=False, repr=False
    )
    """
    Value partitions by enumeration, in declaration order.
    """

    _declared_by_iri: dict[str, type] = field(
        default_factory=dict, init=False, repr=False
    )
    """
    The class each identifier belongs to, whether a concept or a value partition.
    """

    def module(
        self,
        name: str,
        version: str,
        namespace: str,
        depends_on: Sequence[OntologyModule] = (),
    ) -> OntologyModule:
        """
        Add a module to the ontology.

        :param namespace: The IRI prefix of the module's identifiers.
        :param depends_on: The modules whose concepts the new module may specialise.
        :raises DuplicateModule: when the name is already used.
        """
        if name in self._modules:
            raise DuplicateModule(name)
        module = OntologyModule(
            ontology=self,
            name=name,
            version=version,
            namespace=Namespace(namespace),
            depends_on=tuple(depends_on),
        )
        self._modules[name] = module
        return module

    def modules(self) -> tuple[OntologyModule, ...]:
        """
        Return the modules in declaration order.
        """
        return tuple(self._modules.values())

    def versions(self) -> dict[str, str]:
        """
        Return each module's version by module name.
        """
        return {module.name: module.version for module in self._modules.values()}

    def add_concept(self, meta: ConceptMeta) -> None:
        """
        Record a described concept, or refuse it and record nothing.

        :raises InvalidDeclaration: when the class is already declared, specialises a
            concept of a module the declaring module does not depend on, or shares its
            identifier with another declaration.
        """
        self._check_new_class(meta.cls)
        self._check_ancestors(meta)
        self._check_new_iri(meta.cls, meta.iri)
        self._concepts[meta.cls] = meta
        self._declared_by_iri[meta.iri] = meta.cls

    def add_value_partition(self, meta: ValuePartitionMeta) -> None:
        """
        Record a value partition, or refuse it and record nothing.

        :raises InvalidDeclaration: when the enumeration is already declared or shares
            its identifier with another declaration.
        """
        self._check_new_class(meta.enumeration)
        self._check_new_iri(meta.enumeration, meta.iri)
        self._partitions[meta.enumeration] = meta
        self._declared_by_iri[meta.iri] = meta.enumeration

    def find_concept(self, cls: type) -> Optional[ConceptMeta]:
        """
        Return the concept a class realises, or ``None`` when it is not declared.
        """
        return self._concepts.get(cls)

    def find_concept_by_iri(self, iri: str) -> Optional[ConceptMeta]:
        """
        Return the concept with this identifier, or ``None`` when there is none.
        """
        declared = self._declared_by_iri.get(iri)
        return None if declared is None else self._concepts.get(declared)

    def find_value_partition(self, enumeration: type) -> Optional[ValuePartitionMeta]:
        """
        Return the partition an enumeration realises, or ``None`` when it is not declared.
        """
        return self._partitions.get(enumeration)

    def concepts(self, module: Optional[str] = None) -> tuple[ConceptMeta, ...]:
        """
        Return the concepts in declaration order.

        :param module: Restrict the result to the concepts this module declares.
        """
        return tuple(
            meta
            for meta in self._concepts.values()
            if module is None or meta.module == module
        )

    def value_partitions(
        self, module: Optional[str] = None
    ) -> tuple[ValuePartitionMeta, ...]:
        """
        Return the value partitions in declaration order.

        :param module: Restrict the result to the partitions this module declares.
        """
        return tuple(
            meta
            for meta in self._partitions.values()
            if module is None or meta.module == module
        )

    def property_declarations(
        self, module: Optional[str] = None
    ) -> tuple[PropertyDeclaration, ...]:
        """
        Return every property, each listed once under the concept whose body writes it.

        :param module: Restrict the result to the concepts this module declares.
        """
        declarations: list[PropertyDeclaration] = []
        for meta in self.concepts(module):
            described = PropertyMeta.declared_on(meta.cls)
            namespace = self._modules[meta.module].namespace
            for field_name in inspect.get_annotations(meta.cls):
                if field_name not in described:
                    continue
                property_meta = described[field_name]
                declarations.append(
                    PropertyDeclaration(
                        owner=meta.cls,
                        field_name=field_name,
                        meta=property_meta,
                        iri=property_meta.iri or namespace.property_term(field_name),
                        module=meta.module,
                    )
                )
        return tuple(declarations)

    def _check_new_class(self, cls: type) -> None:
        """
        Refuse a class that is already declared here.
        """
        if cls in self._concepts or cls in self._partitions:
            raise InvalidDeclaration(cls, DeclarationFault.CLASS_ALREADY_DECLARED)

    def _check_new_iri(self, cls: type, iri: str) -> None:
        """
        Refuse an identifier that another declaration already has.
        """
        if iri in self._declared_by_iri:
            raise InvalidDeclaration(cls, DeclarationFault.DUPLICATE_IRI)

    def _check_ancestors(self, meta: ConceptMeta) -> None:
        """
        Refuse a concept that specialises one of a module it does not depend on.
        """
        declaring = self._modules[meta.module]
        allowed = {
            declaring.name,
            *(dependency.name for dependency in declaring.transitive_dependencies()),
        }
        for ancestor in meta.cls.__mro__[1:]:
            described = self._concepts.get(ancestor)
            if described is not None and described.module not in allowed:
                raise InvalidDeclaration(
                    meta.cls, DeclarationFault.BASE_FROM_UNDECLARED_DEPENDENCY
                )
