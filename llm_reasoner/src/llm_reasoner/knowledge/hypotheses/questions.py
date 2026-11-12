"""
The questions a language model asks about stored hypotheses read as one body of
knowledge, answered as data: the fragments of a vocabulary class, and what one fragment
is, states and is linked from. Fragments are named as their citations name them, after
the module of their hypothesis's root class when it has one, such as
``pouring/source``, so that names stay apart across the hypotheses read.
"""

from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from operator import attrgetter
from pathlib import Path

from typing_extensions import Any, Optional

from llm_reasoner.exceptions import LLMReasonerError
from llm_reasoner.knowledge.hypotheses.view import HypothesisView
from llm_reasoner.knowledge.vocabulary.ontology_modules import OntologyModuleFile
from llm_reasoner.knowledge.vocabulary.terms import (
    ConceptReference,
    DescriptionEntity,
    PropertyTerm,
    is_inferred,
)
from llm_reasoner.knowledge.vocabulary.lookup import Vocabulary
from llm_reasoner.knowledge.storage.citation import Citation

NAME_SEPARATOR = "/"
"""
What separates the module of a hypothesis from a fragment's name in the name the tools
give the fragment.
"""


# %% errors


class KnowledgeQuestionError(LLMReasonerError):
    """
    A question names something the hypothesis or its vocabulary does not have, which is
    the asker's mistake to correct.
    """


@dataclass
class UnknownFragment(KnowledgeQuestionError):
    """
    No fragment of the hypothesis has the name asked about.
    """

    name: str
    """
    The name asked about.
    """

    def __post_init__(self):
        super().__init__(f"the knowledge has no fragment named '{self.name}'")


@dataclass
class UnknownVocabularyClass(KnowledgeQuestionError):
    """
    No class of the knowledge's vocabulary has the name asked about.
    """

    kind: str
    """
    The class name asked about.
    """

    def __post_init__(self):
        super().__init__(f"the knowledge's vocabulary has no class named '{self.kind}'")


# %% answers


@dataclass(frozen=True)
class KnowledgeAnswer:
    """
    An answer to a question about a hypothesis, written as JSON data.
    """

    def as_data(self) -> Any:
        """
        :return: The answer as JSON data, absent values left out.
        """
        return self.without_absent(asdict(self))

    @staticmethod
    def without_absent(data: Any) -> Any:
        """
        :return: The data with every key whose value is ``None`` left out, at any depth.
        """
        if isinstance(data, dict):
            return {
                key: KnowledgeAnswer.without_absent(value)
                for key, value in data.items()
                if value is not None
            }
        if isinstance(data, list):
            return [KnowledgeAnswer.without_absent(value) for value in data]
        return data


@dataclass(frozen=True)
class KindSummary(KnowledgeAnswer):
    """
    A vocabulary class and what it means.
    """

    name: str
    """
    The class's name.
    """

    meaning: str
    """
    The class's docstring.
    """

    @classmethod
    def of(cls, kind: type) -> KindSummary:
        """
        :return: The summary of a vocabulary class.
        """
        return cls(name=kind.__name__, meaning=inspect.getdoc(kind) or "")


@dataclass(frozen=True)
class FragmentReference(KnowledgeAnswer):
    """
    A fragment named for further questions.
    """

    name: str
    """
    The fragment's name in its hypothesis.
    """

    kind: str
    """
    The name of the fragment's class.
    """

    label: Optional[str] = None
    """
    The fragment's human-readable name, when it has one.
    """


@dataclass(frozen=True)
class FragmentsFound(KnowledgeAnswer):
    """
    The fragments of a vocabulary class in the hypothesis.
    """

    kind: str
    """
    The class asked about.
    """

    fragments: list[FragmentReference]
    """
    Its fragments, in the order the hypothesis holds them.
    """


@dataclass(frozen=True)
class FieldStatement(KnowledgeAnswer):
    """
    What a fragment states for one of its ontology properties.
    """

    field_name: str
    """
    The field standing for the property.
    """

    meaning: str
    """
    The field's docstring.
    """

    fragments: Optional[list[str]] = None
    """
    The names of the fragments the field links to, when it links to fragments.
    """

    value: Any = None
    """
    The field's value, when it holds values rather than fragments: an enumeration
    member by its name, a concept by its IRI.
    """


@dataclass(frozen=True)
class FragmentLink(KnowledgeAnswer):
    """
    A fragment linking to another through one of its fields.
    """

    name: str
    """
    The linking fragment's name.
    """

    kind: str
    """
    The name of the linking fragment's class.
    """

    field_name: str
    """
    The field it links through.
    """


@dataclass(frozen=True)
class FragmentDescription(KnowledgeAnswer):
    """
    What a fragment is, states and is linked from.
    """

    name: str
    """
    The fragment's name.
    """

    kind: KindSummary
    """
    The fragment's class and what it means.
    """

    statements: list[FieldStatement] = field(default_factory=list)
    """
    What the fragment states, one entry per ontology property it gives values for.
    """

    linked_from: list[FragmentLink] = field(default_factory=list)
    """
    The fragments linking to it.
    """

    label: Optional[str] = None
    """
    The fragment's human-readable name, when it has one.
    """


# %% asking


@dataclass
class KnowledgeQuestions:
    """
    Answers questions about stored hypotheses read as one body of knowledge.
    """

    hypotheses: list[HypothesisView]
    """
    The hypotheses read, each with the hypotheses it builds on.
    """

    vocabulary: Vocabulary = field(default_factory=Vocabulary)
    """
    The generated vocabulary, which knows the classes and their fields' meanings.
    """

    # %% the hypotheses read

    def views(self) -> list[HypothesisView]:
        """
        :return: Every hypothesis read, each once, in the order given.
        """
        return list({id(view): view for view in self.hypotheses}.values())

    def module_of(self, view: HypothesisView) -> Optional[OntologyModuleFile]:
        """
        :return: The ontology module of a hypothesis's root class.
        """
        return self.vocabulary.module_file_of(type(view.root))

    def fragments(self) -> list[DescriptionEntity]:
        """
        :return: Every fragment of the hypotheses read, each once.
        """
        return [fragment for view in self.views() for fragment in view.fragments]

    def view_holding(self, fragment: DescriptionEntity) -> HypothesisView:
        """
        :return: The hypothesis read that holds a fragment.
        """
        return next(view for view in self.views() if id(fragment) in view.identifiers)

    def find(self, kind: str) -> FragmentsFound:
        """
        :param kind: The name of a vocabulary class.
        :return: The hypothesis's fragments of that class.
        :raises UnknownVocabularyClass: When the vocabulary has no class of that name.
        """
        found = self.vocabulary.class_named(kind)
        if found is None:
            raise UnknownVocabularyClass(kind=kind)
        return FragmentsFound(
            kind=kind,
            fragments=[
                self.reference(fragment)
                for fragment in self.fragments()
                if isinstance(fragment, found)
            ],
        )

    def describe(self, name: str) -> FragmentDescription:
        """
        :param name: The name of a fragment of the hypothesis.
        :return: What the fragment is, states and is linked from.
        :raises UnknownFragment: When the hypothesis has no fragment of that name.
        """
        fragment = self.fragment_named(name)
        meanings = self.vocabulary.field_meanings(type(fragment))
        return FragmentDescription(
            name=name,
            kind=KindSummary.of(type(fragment)),
            label=fragment.label or None,
            statements=[
                self.statement(field_name, meanings.get(field_name, ""), value)
                for field_name, value in self.stated_values(fragment).items()
            ],
            linked_from=[
                FragmentLink(
                    name=self.name_of(other),
                    kind=type(other).__name__,
                    field_name=field_name,
                )
                for other in self.fragments()
                for field_name, value in self.stated_values(other).items()
                if any(held is fragment for held in self.listed(value))
            ],
        )

    # %% naming and reading fragments

    def name_of(self, fragment: DescriptionEntity) -> str:
        """
        :return: The name the tools give a fragment: its name in its hypothesis, after
            the module of the hypothesis's root class when it has one.
        """
        view = self.view_holding(fragment)
        module_file = self.module_of(view)
        prefix = (
            "" if module_file is None else Path(module_file.value).stem + NAME_SEPARATOR
        )
        return prefix + view.citation_of(fragment).fragment_name

    def fragment_named(self, name: str) -> DescriptionEntity:
        """
        :return: The fragment the tools give the name.
        :raises UnknownFragment: When none has it.
        """
        named = self.fragments_named(name)
        if not named:
            raise UnknownFragment(name=name)
        return named[0]

    def citation_named(self, name: str) -> Optional[Citation]:
        """
        :return: The citation of the fragment the tools give the name, ``None`` when
            none has it.
        """
        named = self.fragments_named(name)
        if not named:
            return None
        return self.view_holding(named[0]).citation_of(named[0])

    def fragments_named(self, name: str) -> list[DescriptionEntity]:
        """
        :return: The fragments the tools give the name; one at most.
        """
        return [
            fragment for fragment in self.fragments() if self.name_of(fragment) == name
        ]

    def reference(self, fragment: DescriptionEntity) -> FragmentReference:
        """
        :return: A fragment named for further questions.
        """
        return FragmentReference(
            name=self.name_of(fragment),
            kind=type(fragment).__name__,
            label=fragment.label or None,
        )

    def written_value(self, value: Any) -> Any:
        """
        :return: A field's value as JSON writes it, a list for a property taking
            several values.
        """
        if isinstance(value, list):
            return [self.written(held) for held in value]
        return self.written(value)

    @staticmethod
    def stated_values(fragment: DescriptionEntity) -> dict[str, Any]:
        """
        :return: The value a fragment states for each ontology property it gives a value
            for, by field name: a list for a property taking several values. Fields
            taking their values from sub-properties are left out.
        """
        stated: dict[str, Any] = {}
        for declared in fields(fragment):
            if PropertyTerm.of_field(type(fragment), declared.name) is None or (
                is_inferred(type(fragment), declared.name)
            ):
                continue
            value = attrgetter(declared.name)(fragment)
            if value is None or (isinstance(value, list) and not value):
                continue
            stated[declared.name] = value
        return stated

    @staticmethod
    def listed(value: Any) -> list[Any]:
        """
        :return: The values a field's value holds: the list itself, or the one value.
        """
        return value if isinstance(value, list) else [value]

    def statement(self, field_name: str, meaning: str, value: Any) -> FieldStatement:
        """
        :return: What a fragment states for one property: the names of the fragments it
            links to, or its value as JSON writes it, a list for a property taking
            several values.
        """
        values = self.listed(value)
        if all(isinstance(held, DescriptionEntity) for held in values):
            return FieldStatement(
                field_name=field_name,
                meaning=meaning,
                fragments=[self.name_of(held) for held in values],
            )
        return FieldStatement(
            field_name=field_name, meaning=meaning, value=self.written_value(value)
        )

    @staticmethod
    def written(value: Any) -> Any:
        """
        :return: A value as JSON writes it: an enumeration member by its name, a concept
            by its IRI, anything else as it is.
        """
        if isinstance(value, Enum):
            return value.name
        if isinstance(value, ConceptReference):
            return value.iri
        return value
