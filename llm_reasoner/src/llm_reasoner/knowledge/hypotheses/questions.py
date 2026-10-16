"""
The questions a language model asks about stored hypotheses read as one body of
knowledge, answered as data: an overview of the action, the fragments of a vocabulary
class, and what one fragment is, states and is linked from. Fragments are named as their
citations name them, those of an action perspective hypothesis after its module, such as
``action-core/source``, so that names stay apart across the hypotheses read.
"""

from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass, field, fields, replace
from enum import Enum
from operator import attrgetter
from pathlib import Path

from typing_extensions import Any, Optional

from llm_reasoner.exceptions import LLMReasonerError
from llm_reasoner.knowledge.hypotheses.view import HypothesisView
from llm_reasoner.knowledge.ontologies import (
    ActionPerspectiveFile,
    ActionPerspectiveModule,
    OntologyModuleFile,
)
from llm_reasoner.knowledge.ontologies.action_perspectives import action_core
from llm_reasoner.knowledge.ontologies.situation import SituationFeatures
from llm_reasoner.knowledge.ontologies.terms import (
    ConceptReference,
    DescriptionEntity,
    PropertyTerm,
    is_inferred,
)
from llm_reasoner.knowledge.ontologies.vocabulary import Vocabulary
from llm_reasoner.knowledge.storage.citation import Citation

NAME_SEPARATOR = "/"
"""
What separates the module of an action perspective hypothesis from a fragment's name in
the name the tools give the fragment.
"""


def perspective_of_name(name: str) -> Optional[ActionPerspectiveFile]:
    """
    :param name: A fragment's name, as the knowledge tools give it.
    :return: The action perspective module the name is named after, ``None`` for a
        fragment of a phase structure.
    """
    prefix, separator, _ = name.partition(NAME_SEPARATOR)
    if not separator:
        return None
    return next(
        (
            module_file
            for module_file in ActionPerspectiveFile
            if Path(module_file.value).stem == prefix
        ),
        None,
    )


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
class FragmentSummary(KnowledgeAnswer):
    """
    A fragment with the values it states, the fragments holding only values that it
    links to written in place, and the names of the other fragments it links to.
    """

    name: str
    """
    The fragment's name in its hypothesis.
    """

    kind: str
    """
    The name of the fragment's class.
    """

    values: dict[str, Any]
    """
    The values the fragment states, by field name: values as JSON writes them, and a
    linked fragment holding only values as an object of its own values.
    """

    links: dict[str, list[str]] = field(default_factory=dict)
    """
    The names of the other fragments it links to, by field name.
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

    parameters: list[FragmentSummary] = field(default_factory=list)
    """
    The parameters in force during the phase, with the values they hold.
    """

    requirements: list[FragmentSummary] = field(default_factory=list)
    """
    What the phase demands of the world, with the values that say when and how.
    """


@dataclass(frozen=True)
class FrameOverview(KnowledgeAnswer):
    """
    The action's frame in the action core: the slots its participants fill, what it is
    for, what it brings about and the choices it leaves open.
    """

    frame: FragmentReference
    """
    The frame.
    """

    slots: list[FragmentSummary]
    """
    The slots its participants and their parts fill.
    """

    objectives: list[FragmentSummary]
    """
    What the action is for.
    """

    consequences: list[FragmentSummary]
    """
    What the action brings about.
    """

    parameter_dimensions: list[FragmentSummary]
    """
    The choices the action leaves open.
    """


@dataclass(frozen=True)
class PerspectiveSummary(KnowledgeAnswer):
    """
    A perspective on the action read with its core.
    """

    module: str
    """
    The file of the perspective's ontology module, such as ``action-frame.ttl``.
    """

    root: FragmentReference
    """
    The fragment the perspective's knowledge is rooted in.
    """

    meaning: str
    """
    What the class of that fragment means.
    """


@dataclass(frozen=True)
class KnowledgeOverview(KnowledgeAnswer):
    """
    What the action is, and what the knowledge read says of it: the situation and the
    phases of a phase structure, and the frame and the perspectives of an action core.
    """

    structure: Optional[KindSummary] = None
    """
    The class of the phase structure describing the action, and what it means, when one
    is read.
    """

    situation: Optional[dict[str, str]] = None
    """
    The situation features the phase structure was generated for, by field name, each
    by the name of its value; features left unknown are absent.
    """

    phases: Optional[list[PhaseSummary]] = None
    """
    The phases in the order the action goes through them, when a phase structure is
    read.
    """

    roles: Optional[list[FragmentReference]] = None
    """
    The roles taking part in the action, when a phase structure is read.
    """

    action: Optional[str] = None
    """
    The IRI of the action concept the knowledge describes.
    """

    frame: Optional[FrameOverview] = None
    """
    The action's frame, when an action core is read.
    """

    perspectives: Optional[list[PerspectiveSummary]] = None
    """
    The perspectives read with the action core, when one is read.
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

    def overview(self) -> KnowledgeOverview:
        """
        :return: What the action is, the situation and the phases a phase structure
            says it goes through, and the frame and the perspectives of an action core.
        """
        phase_view = self.phase_structure_view()
        core_view = self.view_of_module(ActionPerspectiveFile.ACTION_CORE)
        overview = KnowledgeOverview(
            action=self.action_concept(phase_view, core_view),
            frame=None if core_view is None else self.frame_overview(core_view.root),
            perspectives=None if core_view is None else self.perspectives(),
        )
        if phase_view is None:
            return overview
        root = phase_view.root
        features = SituationFeatures(vocabulary=self.vocabulary).of(type(root))
        return replace(
            overview,
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
                    parameters=[
                        self.summary(parameter)
                        for parameter in phase.has_active_parameter
                    ],
                    requirements=[
                        self.summary(requirement)
                        for requirement in phase.has_condition_requirement
                    ],
                )
                for phase in phase_view.phases()
            ],
            roles=[self.reference(role) for role in root.defines_role],
        )

    def frame_overview(self, frame: action_core.ActionFrame) -> FrameOverview:
        """
        :return: The frame of an action core with its slots, objectives, consequences
            and parameter dimensions.
        """
        return FrameOverview(
            frame=self.reference(frame),
            slots=[self.summary(slot) for slot in frame.defines_slot],
            objectives=[
                self.summary(objective) for objective in frame.states_objective
            ],
            consequences=[
                self.summary(consequence) for consequence in frame.brings_about
            ],
            parameter_dimensions=[
                self.summary(dimension) for dimension in frame.leaves_open
            ],
        )

    def perspectives(self) -> list[PerspectiveSummary]:
        """
        :return: The perspectives read, each by its module and its root.
        """
        return [
            PerspectiveSummary(
                module=self.module_of(view).value,
                root=self.reference(view.root),
                meaning=KindSummary.of(type(view.root)).meaning,
            )
            for view in self.views()
            if isinstance(self.module_of(view), ActionPerspectiveFile)
            and self.module_of(view) is not ActionPerspectiveFile.ACTION_CORE
        ]

    def action_concept(
        self,
        phase_view: Optional[HypothesisView],
        core_view: Optional[HypothesisView],
    ) -> Optional[str]:
        """
        :return: The IRI of the action concept the phase structure decomposes, or else
            the one the core's frame frames.
        """
        if phase_view is not None and phase_view.root.decomposes_task is not None:
            return phase_view.root.decomposes_task.iri
        if core_view is not None and core_view.root.frames_action is not None:
            return core_view.root.frames_action.iri
        return None

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

    def view_of_module(
        self, module_file: OntologyModuleFile
    ) -> Optional[HypothesisView]:
        """
        :return: The first hypothesis read whose root belongs to the module.
        """
        return next(
            (view for view in self.views() if self.module_of(view) is module_file),
            None,
        )

    def phase_structure_view(self) -> Optional[HypothesisView]:
        """
        :return: The first hypothesis read that is not about an action perspective, a
            phase structure.
        """
        return next(
            (
                view
                for view in self.views()
                if not isinstance(self.module_of(view), ActionPerspectiveModule)
            ),
            None,
        )

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
            the module of an action perspective hypothesis.
        """
        view = self.view_holding(fragment)
        module_file = self.module_of(view)
        prefix = (
            Path(module_file.value).stem + NAME_SEPARATOR
            if isinstance(module_file, ActionPerspectiveModule)
            else ""
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

    def summary(self, fragment: DescriptionEntity) -> FragmentSummary:
        """
        :return: A fragment with the values it states, the fragments holding only
            values that it links to written in place, and the names of the other
            fragments it links to.
        """
        values: dict[str, Any] = {}
        links: dict[str, list[str]] = {}
        for field_name, value in self.stated_values(fragment).items():
            held = self.listed(value)
            if not any(isinstance(linked, DescriptionEntity) for linked in held):
                values[field_name] = self.written_value(value)
            elif all(self.holds_only_values(linked) for linked in held):
                values[field_name] = (
                    [self.own_values(linked) for linked in held]
                    if isinstance(value, list)
                    else self.own_values(value)
                )
            else:
                links[field_name] = [self.name_of(linked) for linked in held]
        return FragmentSummary(
            name=self.name_of(fragment),
            kind=type(fragment).__name__,
            label=fragment.label or None,
            values=values,
            links=links,
        )

    def holds_only_values(self, fragment: Any) -> bool:
        """
        :return: Whether a fragment states values and links to no other fragment, as a
            range of magnitudes does.
        """
        stated = self.stated_values(fragment)
        return bool(stated) and not any(
            isinstance(held, DescriptionEntity)
            for value in stated.values()
            for held in self.listed(value)
        )

    def own_values(self, fragment: DescriptionEntity) -> dict[str, Any]:
        """
        :return: The values a fragment holding only values states, by field name.
        """
        return {
            field_name: self.written_value(value)
            for field_name, value in self.stated_values(fragment).items()
        }

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
