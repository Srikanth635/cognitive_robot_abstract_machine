"""Recovery resolver LangGraph node — unchanged logic from original llmr."""

from __future__ import annotations

import logging
from typing_extensions import Any, Dict, Optional

from langgraph.graph.state import END, START, StateGraph

from krrood.llmr.workflows.llm_configuration import default_llm
from krrood.llmr.workflows.prompts.recovery import recovery_prompt
from krrood.llmr.workflows.schemas.recovery import RecoverySchema
from krrood.llmr.workflows.states.all_states import RecoveryState

logger = logging.getLogger(__name__)

_recovery_graph: Optional[Any] = None


def _build_recovery_graph() -> Any:
    global _recovery_graph
    if _recovery_graph is not None:
        return _recovery_graph

    structured_llm = default_llm.with_structured_output(RecoverySchema, method="function_calling")

    def _recovery_node(state: RecoveryState) -> Dict[str, Any]:
        chain = recovery_prompt | structured_llm
        try:
            schema: RecoverySchema = chain.invoke(
                {
                    "world_context": state["world_context"],
                    "original_instruction": state["original_instruction"],
                    "failed_action_description": state["failed_action_description"],
                    "error_message": state["error_message"],
                }
            )
            logger.debug(
                "recovery_node – strategy=%s diagnosis=%s",
                schema.recovery_strategy,
                schema.failure_diagnosis,
            )
            return {"resolved_schema": schema.model_dump(), "error": None}
        except Exception as exc:
            logger.error("recovery_node failed: %s", exc)
            return {"resolved_schema": None, "error": str(exc)}

    builder = StateGraph(RecoveryState)
    builder.add_node("recovery", _recovery_node)
    builder.add_edge(START, "recovery")
    builder.add_edge("recovery", END)
    _recovery_graph = builder.compile()
    return _recovery_graph


def run_recovery_resolver(
    world_context: str,
    original_instruction: str,
    failed_action_description: str,
    error_message: str,
) -> Optional[RecoverySchema]:
    """Run the recovery resolver for a failed action execution."""
    graph = _build_recovery_graph()
    final_state = graph.invoke(
        {
            "world_context": world_context,
            "original_instruction": original_instruction,
            "failed_action_description": failed_action_description,
            "error_message": error_message,
        }
    )

    if final_state.get("error"):
        logger.warning("run_recovery_resolver: LLM node error: %s", final_state["error"])
        return None

    raw = final_state.get("resolved_schema")
    if raw is None:
        return None

    return RecoverySchema.model_validate(raw)
