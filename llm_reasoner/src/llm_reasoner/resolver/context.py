from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from krrood.entity_query_language.query.match import Match
from typing_extensions import Any, List

from llm_reasoner.resolver.prompt_format import (
    PromptBlock,
    PromptSection,
    render_blocks,
)
from llm_reasoner.world.classes import is_individual
from llm_reasoner.world.describer import IndividualDescriber
from llm_reasoner.world.rendering import IndividualRenderer

CANDIDATE_LINE_BUDGET = 15
"""
The most lines one candidate is described in, so that a slot offering many candidates
keeps the prompt readable.
"""

# %% situational knowledge


@dataclass
class PromptContextProvider(ABC):
    """
    Supplies knowledge the query does not carry.

    Implementations that read a world live wherever the world lives and are injected
    into the backend, which keeps this package free of world dependencies.

    A provider names the section its text appears in, because the knowledge reaching
    a model is not all of one kind: what is currently true and what the caller is trying
    to achieve answer different questions and are needed for different slots. Collapsing
    them into one block invites the model to read intent as fact.
    """

    @abstractmethod
    def context_for(self, expression: Match) -> str:
        """
        :param expression: The match being resolved.
        :return: Text for this provider's section, empty to add nothing.
        """

    @property
    def section(self) -> PromptSection:
        """:return: The section this provider's text appears in."""
        return PromptSection.SITUATION


@dataclass
class SchemaOnlyContextProvider(PromptContextProvider):
    """
    Adds no situational knowledge, leaving the model only the reflected schema.

    Serves as an uninformative baseline and as the default for environments with no
    world available.
    """

    def context_for(self, expression: Match) -> str:
        return ""


# %% intent


@dataclass
class TaskContextProvider(PromptContextProvider):
    """
    States what the action is meant to achieve.

    Some parameters are determined by intent rather than by world state: how thickly to
    slice, how long to stir, which cutting technique to use. Identical worlds give
    different right answers depending on the goal, so no amount of situational detail
    settles them. This is the only route by which that knowledge reaches the model.
    """

    goal: str = ""
    """
    What the caller is trying to achieve, in plain language.
    """

    constraints: str = ""
    """
    Any preferences or restrictions the caller wants respected.
    """

    def context_for(self, expression: Match) -> str:
        parts = [part for part in (self.goal, self.constraints) if part.strip()]
        return "\n\n".join(part.strip() for part in parts)

    @property
    def section(self) -> PromptSection:
        return PromptSection.TASK


# %% composition


@dataclass
class CompositeContextProvider(PromptContextProvider):
    """
    Presents several providers as one.

    Composition rather than a second seam: a task provider, a world provider and a
    domain provider are the same kind of thing and differ only in what they know, so
    they compose instead of each needing its own place on the backend.

    Providers are kept in order, and one contributing nothing is omitted rather than
    leaving an empty section.
    """

    providers: List[PromptContextProvider] = field(default_factory=list)
    """
    The providers consulted, in the order their sections should be read.
    """

    def context_for(self, expression: Match) -> str:
        return render_blocks(self.sections_for(expression))

    def sections_for(self, expression: Match) -> List[PromptBlock]:
        """
        :param expression: The match being resolved.
        :return: One section per provider that has something to say.
        """
        sections: List[PromptBlock] = []
        for provider in self.providers:
            sections.extend(context_sections(provider, expression))
        return sections


def context_sections(
    provider: PromptContextProvider, expression: Match
) -> List[PromptBlock]:
    """
    Flatten a provider into the sections it contributes.

    Composites are expanded so that nesting them changes the grouping of providers
    without changing the shape of the prompt.

    :param provider: The provider to consult.
    :param expression: The match being resolved.
    :return: One section per contributing provider, empty ones dropped.
    """
    if isinstance(provider, CompositeContextProvider):
        return provider.sections_for(expression)
    text = provider.context_for(expression)
    if not text.strip():
        return []
    return [PromptBlock(section=provider.section, body=text.strip())]


# %% candidate description


@dataclass
class CandidateRenderer(ABC):
    """
    Describes an individual to the model.

    The model receives a description addressed by an integer token and never the object
    itself, so a resolved value can be a live world individual while this package stays
    decoupled from the world.
    """

    @abstractmethod
    def render(self, candidate: Any, expression: Match) -> str:
        """
        :param candidate: One element of a referent slot's extension.
        :param expression: The match being resolved, so a candidate can be described
            relative to the task rather than in isolation. What distinguishes two
            candidates is usually their fitness for *this* query.
        :return: A description; lines after the first continue it.
        """


@dataclass
class ReprCandidateRenderer(CandidateRenderer):
    """
    Describes an individual by its type name and representation.
    """

    max_length: int = 160
    """
    Longest description emitted before truncation.
    """

    def render(self, candidate: Any, expression: Match) -> str:
        if isinstance(candidate, type):
            return self.render_kind(candidate)
        text = f"{type(candidate).__name__}  {candidate!r}"
        if len(text) <= self.max_length:
            return text
        return text[: self.max_length - 1] + "\u2026"

    @staticmethod
    def render_kind(candidate: type) -> str:
        """
        Describe a class offered as a candidate.

        A bare list of class names says little; the docstring is what lets a model tell
        one kind from another.

        :param candidate: A class offered for a sortal slot.
        :return: Its name and the first line of its docstring.
        """
        docstring = inspect.getdoc(candidate)
        if not docstring:
            return candidate.__name__
        return f"{candidate.__name__}  {docstring.strip().splitlines()[0]}"


@dataclass
class StructuralCandidateRenderer(CandidateRenderer):
    """
    Describes an individual by walking its fields: what kind of thing it is, what its
    stored values mean, and which individuals hold it.

    The walk reads only dataclass fields and their docstrings, so a world individual is
    described without this package knowing its world model.
    """

    describer: IndividualDescriber = field(default_factory=IndividualDescriber)
    """
    Describes an individual.
    """

    individual_renderer: IndividualRenderer = field(
        default_factory=lambda: IndividualRenderer(line_budget=CANDIDATE_LINE_BUDGET)
    )
    """
    Writes a description as text.
    """

    def render(self, candidate: Any, expression: Match) -> str:
        if isinstance(candidate, type):
            return ReprCandidateRenderer.render_kind(candidate)
        if not is_individual(candidate):
            return ReprCandidateRenderer().render(candidate, expression)
        return self.individual_renderer.render(self.describer.describe(candidate))
