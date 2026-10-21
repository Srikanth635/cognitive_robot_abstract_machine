"""
Offline provider responses for the real tool-loop and MCP integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from litellm.types.utils import ModelResponse
from typing_extensions import Any

# %% deterministic provider


@dataclass
class ScriptedCompletion:
    """
    Return a finite response script without contacting a provider.
    """

    responses: list[ModelResponse]
    """The responses in their intended request order."""
    requests: list[dict[str, Any]] = field(default_factory=list, repr=False)
    """
    The provider requests captured without credential representations.
    """

    async def __call__(self, **arguments: Any) -> ModelResponse:
        """
        Consume the next scripted response and retain its request.
        """
        response = self.responses[len(self.requests)]
        self.requests.append(arguments)
        return response
