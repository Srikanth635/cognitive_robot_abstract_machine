"""
Trying an answer before it is accepted: a caller that can perform what an answer
constructs, in a simulation for instance, tells the resolver when the outcome falls
short, and the model reads that outcome to answer again.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from typing_extensions import Any, Sequence


class OutcomeCheck(ABC):
    """
    Tries the instances an answer constructs and refuses the answer when what they do
    falls short of what they are for.
    """

    @abstractmethod
    def check(self, instances: Sequence[Any]) -> None:
        """
        Try the instances an answer constructs.

        :param instances: The instances, already meeting the query's conditions.
        :raises OutcomeFellShort: When what they do falls short, stating the outcome.
        """
