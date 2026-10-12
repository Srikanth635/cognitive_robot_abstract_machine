"""
The questions a language model asks about one stored hypothesis, answered as data: an
overview of the action, the fragments of a vocabulary class, and what one fragment is,
states and is linked from. Fragments are named as their citations name them.
"""

from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from operator import attrgetter

from typing_extensions import Any, Optional

from llm_reasoner.exceptions import LLMReasonerError
from llm_reasoner.knowledge.hypotheses.view import HypothesisView
from llm_reasoner.knowledge.ontologies.situation import SituationFeatures
from llm_reasoner.knowledge.ontologies.terms import (
    ConceptReference,
    DescriptionEntity,
    PropertyTerm,
    is_inferred,
)
from llm_reasoner.knowledge.ontologies.vocabulary import Vocabulary

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
class PhaseSummary(KnowledgeAnswer):
    """
    A phase the action goes through.
    """

    name: str
    """
    The phase's name in its hypothesis.
    """

    kind: KindSummary
    """
    The phase's class and what it means.
    """

    label: Optional[str] = None
    """
    The phase's human-readable name, when it has one.
    """

    aims_for: Optional[str] = None
    """
    The name of the subgoal the phase aims for, when it states one.
    """


@dataclass(frozen=True)
class KnowledgeOverview(KnowledgeAnswer):
    """
    What the action is, the situation the knowledge is for, and what the action goes
    through.
    """

    structure: KindSummary
    """
    The class of the structure describing the action, and what it means.
    """

    situation: dict[str, str]
    """
    The situation features the knowledge was generated for, by field name, each by the
    name of its value; features left unknown are absent.
    """

    phases: list[PhaseSummary]
    """
    The phases in the order the action goes through them.
    """

    roles: list[FragmentReference]
    """
    The roles taking part in the action.
    """

    action: Optional[str] = None
    """
    The IRI of the action concept the structure decomposes.
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
    Answers questions about one stored hypothesis.
    """

    hypothesis: HypothesisView
    """
    The hypothesis asked about.
    """

    vocabulary: Vocabulary = field(default_factory=Vocabulary)
    """
    The generated vocabulary, which knows the classes and their fields' meanings.
    """

    def overview(self) -> KnowledgeOverview:
        """
        :return: What the action is, the situation the knowledge is for, and what the
            action goes through.
        """
        root = self.hypothesis.root
        features = SituationFeatures(vocabulary=self.vocabulary).of(type(root))
        return KnowledgeOverview(
            action=None if root.decomposes_task is None else root.decomposes_task.iri,
            structure=KindSummary.of(type(root)),
            situation={
                feature.field_name: feature.value_in(root).name
                for feature in features
                if feature.value_in(root) is not None
            },
            phases=[
                PhaseSummary(
                    name=self.name_of(phase),
                    kind=KindSummary.of(type(phase)),
                    label=phase.label or None,
                    aims_for=(
                        None if phase.aims_for is None else self.name_of(phase.aims_for)
                    ),
                )
                for phase in self.hypothesis.phases()
            ],
            roles=[self.reference(role) for role in root.defines_role],
        )

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
                for fragment in self.hypothesis.instances_of(found)
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
                for other in self.hypothesis.fragments
                for field_name, value in self.stated_values(other).items()
                if any(held is fragment for held in self.listed(value))
            ],
        )

    # %% naming and reading fragments

    def name_of(self, fragment: DescriptionEntity) -> str:
        """
        :return: The name a fragment has in the hypothesis.
        """
        return self.hypothesis.citation_of(fragment).fragment_name

    def fragment_named(self, name: str) -> DescriptionEntity:
        """
        :return: The hypothesis's fragment of the given name.
        :raises UnknownFragment: When it has none.
        """
        named = [
            fragment
            for fragment in self.hypothesis.fragments
            if self.name_of(fragment) == name
        ]
        if not named:
            raise UnknownFragment(name=name)
        return named[0]

    def reference(self, fragment: DescriptionEntity) -> FragmentReference:
        """
        :return: A fragment named for further questions.
        """
        return FragmentReference(
            name=self.name_of(fragment),
            kind=type(fragment).__name__,
            label=fragment.label or None,
        )

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
            field_name=field_name,
            meaning=meaning,
            value=(
                [self.written(held) for held in values]
                if isinstance(value, list)
                else self.written(value)
            ),
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
