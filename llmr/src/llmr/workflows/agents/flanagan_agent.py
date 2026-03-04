"""Flanagan motion-phase reasoning agent for the llmr workflow."""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph

from ..llm_configuration import default_llm
from ..pydantics.flanagan_models import (
    ForceDynamics,
    ForceDynamicsMap,
    FailureRecoveryMap,
    FlanaganState,
    GoalStateMap,
    NormalizedPhases,
    ObjectAwarePhasePlanner,
    PhasePreconditionsMap,
    PhaseTiming,
    SensoryFeedbackMap,
    TemporalConstraintsMap,
)
from ..prompts.flanagan_prompts import (
    failure_recovery_prompt,
    force_dynamics_prompt,
    goal_state_generator_prompt,
    phase_normalization_prompt,
    precondition_generator_prompt,
    sensory_feedback_predictor_prompt,
    task_decomposer_prompt,
    temporal_constraints_prompt,
)
from ..states.all_states import ModelReasoningState

flanagan_graph_memory = MemorySaver()

_default_llm = default_llm


# ── LangGraph node functions ───────────────────────────────────────────────────

def task_decomposer_node(state: FlanaganState) -> dict:
    """Object-aware task decomposition with previous action context."""
    instruction = state["instruction"]
    previous_actions = state.get("previous_actions", [])

    chain = task_decomposer_prompt | _default_llm.with_structured_output(
        ObjectAwarePhasePlanner, method="json_mode"
    )
    phase_plan = chain.invoke(
        {"instruction": instruction, "previous_actions": previous_actions}
    )

    phases = [step.phase for step in phase_plan.phases]
    phase_objects = [step.target_object for step in phase_plan.phases]
    phase_descriptions = [step.description or "" for step in phase_plan.phases]
    phases_with_objects = [
        {
            "phase": step.phase,
            "target_object": step.target_object,
            "description": step.description,
        }
        for step in phase_plan.phases
    ]

    return {
        "initial_phases": phases,
        "phase_objects": phase_objects,
        "phase_descriptions": phase_descriptions,
        "phases": phases,
        "phases_with_objects": phases_with_objects,
    }


def phase_normalization_node(state: FlanaganState) -> dict:
    """Normalise raw phase names to the canonical vocabulary."""
    action_phases = state["initial_phases"]
    chain = phase_normalization_prompt | _default_llm.with_structured_output(
        NormalizedPhases, method="json_mode"
    )
    normalized_phases = chain.invoke({"action_phases": action_phases})
    return {"phases": normalized_phases.normalized_phases}


def precondition_generator_node(state: FlanaganState) -> dict:
    """Generate symbolic preconditions for each motion phase."""
    instruction = state["instruction"]
    action_phases = state.get("phases_with_objects", state["phases"])

    chain = precondition_generator_prompt | _default_llm.with_structured_output(
        PhasePreconditionsMap, method="json_mode"
    )
    preconditions_result = chain.invoke(
        {"instruction": instruction, "action_phases": action_phases}
    )
    return {"preconditions": preconditions_result.phase_preconditions}


def force_dynamics_node(state: FlanaganState) -> dict:
    """Generate force dynamics for each motion phase."""
    instruction = state["instruction"]
    action_phases = state.get("phases_with_objects", state["phases"])
    preconditions = state["preconditions"]

    chain = force_dynamics_prompt | _default_llm.with_structured_output(
        ForceDynamicsMap, method="json_mode"
    )
    force_dynamics_result = chain.invoke(
        {
            "instruction": instruction,
            "action_phases": action_phases,
            "preconditions": preconditions,
        }
    )

    force_dynamics_dict = {
        key: value.model_dump() if isinstance(value, ForceDynamics) else value
        for key, value in force_dynamics_result.force_dynamics.items()
    }
    return {"force_dynamics": force_dynamics_dict}


def goal_state_generator_node(state: FlanaganState) -> dict:
    """Generate goal states for each motion phase."""
    instruction = state["instruction"]
    action_phases = state.get("phases_with_objects", state["phases"])
    preconditions = state["preconditions"]
    force_dynamics = state["force_dynamics"]

    chain = goal_state_generator_prompt | _default_llm.with_structured_output(
        GoalStateMap, method="json_mode"
    )
    goal_states_result = chain.invoke(
        {
            "instruction": instruction,
            "action_phases": action_phases,
            "preconditions": preconditions,
            "force_dynamics": force_dynamics,
        }
    )
    return {"goal_states": goal_states_result.goal_states}


def sensory_feedback_predictor_node(state: FlanaganState) -> dict:
    """Predict expected sensory feedback per motion phase."""
    instruction = state["instruction"]
    action_phases = state.get("phases_with_objects", state["phases"])
    preconditions = state["preconditions"]
    force_dynamics = state["force_dynamics"]
    goal_states = state["goal_states"]

    chain = sensory_feedback_predictor_prompt | _default_llm.with_structured_output(
        SensoryFeedbackMap, method="json_mode"
    )
    sensory_feedback_result = chain.invoke(
        {
            "instruction": instruction,
            "action_phases": action_phases,
            "preconditions": preconditions,
            "force_dynamics": force_dynamics,
            "goal_states": goal_states,
        }
    )
    return {"sensory_feedbacks": sensory_feedback_result.sensory_feedback}


def failure_recovery_node(state: FlanaganState) -> dict:
    """Predict failure modes and recovery strategies per phase."""
    instruction = state["instruction"]
    action_phases = state.get("phases_with_objects", state["phases"])
    preconditions = state["preconditions"]
    force_dynamics = state["force_dynamics"]
    goal_states = state["goal_states"]
    sensory_feedback = state["sensory_feedbacks"]

    chain = failure_recovery_prompt | _default_llm.with_structured_output(
        FailureRecoveryMap, method="json_mode"
    )
    failure_recovery_result = chain.invoke(
        {
            "instruction": instruction,
            "action_phases": action_phases,
            "preconditions": preconditions,
            "force_dynamics": force_dynamics,
            "sensory_feedback": sensory_feedback,
            "goal_states": goal_states,
        }
    )
    return {"failure_and_recovery": failure_recovery_result.failure_and_recovery}


def temporal_constraints_node(state: FlanaganState) -> dict:
    """Generate timing constraints for each motion phase."""
    instruction = state["instruction"]
    action_phases = state.get("phases_with_objects", state["phases"])
    preconditions = state["preconditions"]
    force_dynamics = state["force_dynamics"]
    goal_states = state["goal_states"]
    sensory_feedback = state["sensory_feedbacks"]
    failure_and_recovery = state["failure_and_recovery"]

    chain = temporal_constraints_prompt | _default_llm.with_structured_output(
        TemporalConstraintsMap, method="json_mode"
    )
    temporal_constraints_result = chain.invoke(
        {
            "instruction": instruction,
            "action_phases": action_phases,
            "preconditions": preconditions,
            "force_dynamics": force_dynamics,
            "sensory_feedback": sensory_feedback,
            "failure_and_recovery": failure_and_recovery,
            "goal_states": goal_states,
        }
    )

    temporal_constraints_dict = {
        key: value.model_dump() if isinstance(value, PhaseTiming) else value
        for key, value in temporal_constraints_result.temporal_constraints.items()
    }
    return {"temporal_constraints": temporal_constraints_dict}


def composition_node(state: FlanaganState) -> dict:
    """Compose final output with object-aware phase information."""
    phases = state["phases"]
    phase_objects = state.get("phase_objects", [""] * len(phases))
    phase_descriptions = state.get("phase_descriptions", [""] * len(phases))
    preconditions = state["preconditions"]
    goal_states = state["goal_states"]
    force_dynamics = state["force_dynamics"]
    sensory_feedbacks = state.get("sensory_feedbacks", {})
    failure_and_recovery = state.get("failure_and_recovery", {})
    temporal_constraints = state.get("temporal_constraints", {})
    instruction = state["instruction"]

    composed_output: dict = {"instruction": instruction, "phases": []}

    for phase, target_object, description in zip(phases, phase_objects, phase_descriptions):
        phase_key = f"{phase}_{target_object}" if target_object else phase
        phase_capitalised = phase[0].upper() + phase[1:] if phase else phase

        entry = {
            "phase": phase,
            "target_object": target_object,
            "description": description,
            "symbol": (
                f"->[ robot {phase.lower()}s {target_object}]"
                if target_object
                else f"->[ robot performs {phase.lower()}]"
            ),
            "goal_state": goal_states.get(phase_key, goal_states.get(phase_capitalised, {})),
            "preconditions": preconditions.get(
                phase_key, preconditions.get(phase_capitalised, {})
            ),
            "force_dynamics": force_dynamics.get(
                phase_key, force_dynamics.get(phase_capitalised, {})
            ),
            "sensory_feedback": sensory_feedbacks.get(
                phase_key, sensory_feedbacks.get(phase_capitalised.lower(), {})
            ),
            "failure_and_recovery": failure_and_recovery.get(
                phase_key, failure_and_recovery.get(phase_capitalised, {})
            ),
            "temporal_constraints": temporal_constraints.get(
                phase_key, temporal_constraints.get(phase_capitalised, {})
            ),
        }
        composed_output["phases"].append(entry)

    return {"flanagan": composed_output}


# ── Graph assembly ─────────────────────────────────────────────────────────────

flanagan_graph_builder = StateGraph(FlanaganState)
flanagan_graph_builder.add_node("task_decomposer", task_decomposer_node)
flanagan_graph_builder.add_node("phase_normalization", phase_normalization_node)
flanagan_graph_builder.add_node("precondition_generator", precondition_generator_node)
flanagan_graph_builder.add_node("force_dynamics_generator", force_dynamics_node)
flanagan_graph_builder.add_node("goal_state_generator", goal_state_generator_node)
flanagan_graph_builder.add_node("sensory_feedback_predictor", sensory_feedback_predictor_node)
flanagan_graph_builder.add_node("failure_recovery_predictor", failure_recovery_node)
flanagan_graph_builder.add_node("temporal_constraints_predictor", temporal_constraints_node)
flanagan_graph_builder.add_node("composition", composition_node)
flanagan_graph_builder.set_entry_point("task_decomposer")

flanagan_graph_builder.add_edge("task_decomposer", "phase_normalization")
flanagan_graph_builder.add_edge("phase_normalization", "precondition_generator")
flanagan_graph_builder.add_edge("precondition_generator", "force_dynamics_generator")
flanagan_graph_builder.add_edge("force_dynamics_generator", "goal_state_generator")
flanagan_graph_builder.add_edge("goal_state_generator", "sensory_feedback_predictor")
flanagan_graph_builder.add_edge("sensory_feedback_predictor", "failure_recovery_predictor")
flanagan_graph_builder.add_edge("failure_recovery_predictor", "temporal_constraints_predictor")
flanagan_graph_builder.add_edge("temporal_constraints_predictor", "composition")

flanagan_graph = flanagan_graph_builder.compile(checkpointer=flanagan_graph_memory)


def flanagan_agent_node(state: ModelReasoningState) -> dict:
    """LangGraph node: run the full Flanagan motion-phase pipeline.

    Resolves previous atomic instructions from the intent list to provide
    motion-planning context for sequential tasks.
    """
    instruction: str = state["instruction"]
    intents: dict = state["intents"]

    atomic_instructions = [
        intent.get("atomic_instruction")
        for intent in intents
        if intent.get("atomic_instruction")
    ]

    try:
        index = atomic_instructions.index(instruction)
        previous_instructions = atomic_instructions[:index]
        final_flanagan_state = flanagan_graph.invoke(
            {"instruction": instruction, "previous_actions": previous_instructions}
        )
    except ValueError:
        final_flanagan_state = flanagan_graph.invoke(
            {"instruction": instruction, "previous_actions": atomic_instructions}
        )

    return {"flanagan": final_flanagan_state["flanagan"]}


if __name__ == "__main__":
    test_state: dict = {
        "instruction": "pick up the cooking pan from the wooden drawer",
        "intents": [],
    }
    output = flanagan_agent_node(test_state)
    print(output)
