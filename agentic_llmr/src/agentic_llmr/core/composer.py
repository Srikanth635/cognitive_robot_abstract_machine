"""Composer node — final cross-domain synthesis.

Runs once, after the supervisor decides the goal is satisfied. It reasons over ALL
facts gathered by every specialist (not just the last one) plus their conclusions,
and frames the final answer to the user's original goal. This is the only node that
composes across domains, and the only source of the value returned to the caller.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from agentic_llmr.core.state import RobotAgentState

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

_COMPOSER_SYSTEM = """You are the composer for a cognitive robot. The specialists have
finished gathering information. Using ONLY the facts and specialist findings provided,
write the final answer to the user's ORIGINAL goal.

PRINCIPLES
- Reason across domains: combine scene facts and kinematic facts as the goal requires.
- Ground every statement in the provided facts. Never invent a coordinate, measurement,
  or result that is not present.
- If the goal could not be satisfied (e.g. a target is unreachable, no valid grasp, a
  step failed), say so plainly — state the blocker and what would change it. A truthful
  "it cannot be done yet, because…" is a complete answer.
- Be concise and concrete. Do not describe your process or which tools ran.

OUTPUT FORMAT
- For an informational goal: a clear natural-language answer.
- For a goal that commands the robot to ACT (a manipulation designator): output a single
  fenced ```json block with the resolved action designator ({"action_type": ..., "parameters": {...}})
  built from the facts. If it cannot be fully resolved, explain what is missing instead of
  emitting a partial or invented designator.
"""

_FACT_CHANNELS = [
    ("scene_facts", "SCENE FACTS"),
    ("kinematic_facts", "KINEMATIC FACTS"),
    ("action_schema", "ACTION SCHEMA"),
]


def _render_full_facts(state: RobotAgentState) -> str:
    """Render the complete (untruncated) fact record for final synthesis.

    Safe to send in full: the composer runs once, so there is no per-turn growth.
    """
    sections = []
    for key, label in _FACT_CHANNELS:
        facts = state.get(key) or {}
        if not facts:
            continue
        lines = [f"  {k}: {v}" for k, v in facts.items()]
        sections.append(f"{label}:\n" + "\n".join(lines))
    return "\n\n".join(sections) if sections else "(no structured facts were gathered)"


def _render_specialist_findings(state: RobotAgentState) -> str:
    """Pull the specialists' natural-language conclusions from the message log."""
    lines = []
    for msg in state.get("messages", []):
        content = getattr(msg, "content", "")
        if isinstance(content, str) and content.startswith("["):
            lines.append(content)
    return "\n".join(lines) if lines else "(no specialist summaries)"


def make_composer_node(llm: "BaseChatModel"):
    """Return a composer node function bound to the given LLM."""

    def composer(state: RobotAgentState) -> dict:
        content = (
            f"ORIGINAL GOAL: {state.get('instruction', '')}\n"
            f"CONTEXT: {state.get('template_context', '') or '(none)'}\n"
            f"GOAL TYPE: {state.get('query_kind', 'other')}\n\n"
            f"SPECIALIST FINDINGS:\n{_render_specialist_findings(state)}\n\n"
            f"FACTS GATHERED:\n{_render_full_facts(state)}"
        )
        result = llm.invoke([
            SystemMessage(content=_COMPOSER_SYSTEM),
            HumanMessage(content=content),
        ])
        answer = result.content if hasattr(result, "content") else str(result)
        logger.debug("[Composer] final answer framed (%d chars).", len(answer))
        return {"final_response": answer, "messages": [AIMessage(content=answer)]}

    return composer
