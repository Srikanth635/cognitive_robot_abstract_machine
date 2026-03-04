"""FrameNet semantic reasoning node for the llmr workflow."""

from __future__ import annotations

from ..llm_configuration import default_llm
from ..pydantics.framenet_pydantics import FrameNetRepresentation
from ..prompts.framenet_prompts import framenet_prompt
from ..states.all_states import ModelReasoningState

_framenet_llm = default_llm.with_structured_output(
    FrameNetRepresentation, method="json_schema"
)


def framenet_node(state: ModelReasoningState) -> dict:
    """LangGraph node: annotate *instruction* with FrameNet frame semantics.

    Args:
        state: Current workflow state containing the natural language instruction.

    Returns:
        State patch with 'framenet_model' set to the JSON-serialised representation.
    """
    instruction: str = state["instruction"]
    chain = framenet_prompt | _framenet_llm
    response: FrameNetRepresentation = chain.invoke({"input_instruction": instruction})
    return {"framenet_model": response.model_dump_json(indent=2, by_alias=True)}
