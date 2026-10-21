"""
Outcomes of an answer: a caller that can perform what an answer constructs, in a
simulation for instance, tells the resolver when the outcome falls short, and the model
reads that outcome to answer again; a caller that can predict it offers the model tools
to predict it before answering.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from typing_extensions import Any, List, Sequence, Tuple

from llm_reasoner.clients.tools import Tool
from llm_reasoner.resolver.prompt_format import PromptRule


class OutcomeCheck(ABC):
    """
    Tries the instances an answer constructs and refuses the answer when what they do
    falls short of what they are for.
    """

    @property
    def withheld_slots(self) -> Tuple[str, ...]:
        """
        The fields of the action that the query leaves out and the model is not asked
        for until a refusal opens them; the action's defaults stand until then.
        """
        return ()

    @abstractmethod
    def check(self, instances: Sequence[Any]) -> None:
        """
        Try the instances an answer constructs.

        :param instances: The instances, already meeting the query's conditions.
        :raises OutcomeFellShort: When what they do falls short, stating the outcome and
            the withheld slots it opens, if any.
        """


class OutcomePrediction(ABC):
    """
    Predicts what the instances an answer would construct bring about, as tools the
    model calls before it answers.
    """

    @abstractmethod
    def tools(self) -> List[Tool]:
        """
        :return: The tools predicting what an answer would bring about.
        """

    @property
    @abstractmethod
    def rule(self) -> PromptRule:
        """
        The rule saying when and how the model uses the tools.
        """
