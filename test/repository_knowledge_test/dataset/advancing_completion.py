"""
Move disposable source history while a model turn is in progress.
"""

from __future__ import annotations

from dataclasses import dataclass

from litellm.types.utils import ModelResponse
from typing_extensions import Any

from .scripted_completion import ScriptedCompletion
from ..support import CommittedProject

# %% interleaved source change


@dataclass(kw_only=True)
class AdvancingCompletion(ScriptedCompletion):
    """
    Advance fixture history before returning the last scripted response.
    """

    project: CommittedProject
    """
    The disposable history changed during the provider turn.
    """

    async def __call__(self, **arguments: Any) -> ModelResponse:
        response = await super().__call__(**arguments)
        if len(self.requests) == len(self.responses):
            self.project.git(
                "commit", "--quiet", "--allow-empty", "-m", "Advance during model turn"
            )
        return response
