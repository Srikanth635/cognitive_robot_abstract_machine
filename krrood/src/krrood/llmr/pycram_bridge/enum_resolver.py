"""LLM-based enum resolver — picks an enum value when the slot-filler didn't specify one.

Called by :class:`~krrood.llmr.pycram_bridge.auto_handler.AutoActionHandler` for
required ENUM fields that were not extracted from the instruction.
"""

from __future__ import annotations

import logging
from typing_extensions import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a robot action parameter resolver.

Given a description of what a parameter means, a snapshot of the current world state,
and a list of allowed values, choose the most appropriate value for the parameter.

Respond with the exact member name (no extra text) and a short reasoning sentence.
"""

_HUMAN_TEMPLATE = """\
## Parameter
Name        : {param_name}
Description : {description}
Allowed values: {members}

## Already-resolved parameters
{known_params}

## World context
{world_context}

Choose the most appropriate value for '{param_name}'.
"""


class _EnumResolverOutput(BaseModel):
    value: str = Field(description="Exact member name from the allowed values list.")
    reasoning: str = Field(description="One sentence explaining the choice.")


class EnumResolver:
    """Resolves a single ENUM field value using the LLM when no value was extracted.

    :param llm: Optional pre-configured LLM with structured output. If ``None``,
                uses the krrood default LLM on first call.
    """

    def __init__(self, llm: Optional[Any] = None) -> None:
        self._llm = llm

    def _get_chain(self) -> Any:
        if self._llm is None:
            from langchain_core.prompts import ChatPromptTemplate
            from krrood.llmr.workflows.llm_configuration import default_llm

            prompt = ChatPromptTemplate.from_messages(
                [("system", _SYSTEM_PROMPT), ("human", _HUMAN_TEMPLATE)]
            )
            self._llm = prompt | default_llm.with_structured_output(
                _EnumResolverOutput, method="function_calling"
            )
        return self._llm

    def resolve(
        self,
        param_name: str,
        description: str,
        members: List[str],
        world_context: str = "",
        known_params: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Return the resolved enum member name, or ``None`` on failure.

        :param param_name: Field name (e.g. ``"arm"``).
        :param description: Field docstring used as LLM context.
        :param members: List of valid enum member names.
        :param world_context: Serialized world state string.
        :param known_params: Already-resolved parameters for context.
        :return: A member name from *members*, or ``None`` on error.
        """
        known_str = (
            "\n".join(f"  {k}: {v}" for k, v in (known_params or {}).items())
            or "None resolved yet."
        )
        try:
            result: _EnumResolverOutput = self._get_chain().invoke(
                {
                    "param_name": param_name,
                    "description": description,
                    "members": ", ".join(members),
                    "known_params": known_str,
                    "world_context": world_context or "World context unavailable.",
                }
            )
            if result.value not in members:
                logger.warning(
                    "EnumResolver returned '%s' which is not in %s — using first member.",
                    result.value, members,
                )
                return members[0] if members else None
            logger.debug(
                "EnumResolver: %s=%s  (%s)", param_name, result.value, result.reasoning
            )
            return result.value
        except Exception as exc:
            logger.error("EnumResolver failed for '%s': %s", param_name, exc)
            return None
