"""Domain exceptions for minimal_llmr."""

from __future__ import annotations


class ActionClassificationFailed(RuntimeError):
    def __init__(self, instruction: str) -> None:
        super().__init__(
            f"Could not classify instruction into a known action class: {instruction!r}"
        )
        self.instruction = instruction


class ParameterDescriptionFailed(RuntimeError):
    def __init__(self, action_name: str) -> None:
        super().__init__(
            f"LLM returned no parameter descriptions for action: {action_name!r}"
        )
        self.action_name = action_name


class UnresolvedRequiredParameters(RuntimeError):
    def __init__(self, action_name: str, params: list[str]) -> None:
        super().__init__(
            f"Required parameters still unresolved for {action_name!r}: {params}"
        )
        self.action_name = action_name
        self.params = params
