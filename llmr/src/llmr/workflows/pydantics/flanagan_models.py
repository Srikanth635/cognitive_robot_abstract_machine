"""Pydantic and TypedDict models for Flanagan motion-phase reasoning."""

import re
from typing import Any
from typing_extensions import Dict, List, Literal, Optional, TypedDict, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PhaseStep(BaseModel):
    """Individual phase with target object context."""

    model_config = ConfigDict(extra="forbid")

    phase: Literal[
        "Approach", "Grasp", "Lift", "Transport", "Place", "Align", "Cut", "Slice",
        "Tilt", "Pour", "Insert", "Withdraw", "Release", "Reorient", "Stabilize", "Inspect"
    ]
    target_object: str = Field(description="What object/part is being manipulated in this phase")
    description: Optional[str] = Field(default=None, description="Brief description of what's happening")


class ObjectAwarePhasePlanner(BaseModel):
    """Enhanced output with object context for each phase."""

    model_config = ConfigDict(extra="forbid")

    phases: List[PhaseStep]


class NormalizedPhases(BaseModel):
    model_config = ConfigDict(extra="forbid")

    normalized_phases: List[Literal[
        "Approach", "Grasp", "Lift", "Transport", "Place", "Align", "Cut", "Slice",
        "Tilt", "Pour", "Insert", "Withdraw", "Release", "Reorient", "Stabilize", "Inspect"
    ]]


class PhasePreconditionsMap(BaseModel):
    model_config = ConfigDict(extra="allow")

    phase_preconditions: Dict[str, Dict[str, Any]]


class ForceProfile(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str = ""
    expected_range_N: Union[List[float], None] = None
    expected_range_Nm: Union[List[float], None] = None

    @model_validator(mode="before")
    @classmethod
    def coerce_malformed(cls, data):
        """Handle LLM returning varied formats for force_profile.

        Observed LLM outputs:
          - str:  "2-5N linear pull"
          - dict with wrong keys: {"linear_pull": "5-10N"}
          - correct dict: {"type": "grip", "expected_range_N": [5, 10]}
        """
        def _extract_range(text: str) -> Union[List[float], None]:
            nums = re.findall(r"[\d.]+", str(text))
            if len(nums) >= 2:
                return [float(nums[0]), float(nums[1])]
            return None

        if isinstance(data, str):
            return {"type": data, "expected_range_N": _extract_range(data)}

        if isinstance(data, dict) and "type" not in data:
            # e.g. {"linear_pull": "5-10N"} → type="linear_pull", range=[5,10]
            key = next(iter(data), "unknown")
            val = data[key]
            range_N = _extract_range(val) if isinstance(val, str) else None
            return {"type": key, "expected_range_N": range_N}

        return data


class ForceDynamics(BaseModel):
    model_config = ConfigDict(extra="allow")

    contact: bool = False
    motion_type: str = ""
    force_exerted: str = ""
    force_profile: Union[ForceProfile, None] = None


class ForceDynamicsMap(BaseModel):
    model_config = ConfigDict(extra="allow")

    force_dynamics: Dict[str, ForceDynamics]


class GoalStateMap(BaseModel):
    model_config = ConfigDict(extra="allow")

    goal_states: Dict[str, Dict[str, Any]]


class SensoryFeedbackMap(BaseModel):
    model_config = ConfigDict(extra="allow")

    sensory_feedback: Dict[str, Dict[str, Any]]


class FailureRecoveryMap(BaseModel):
    model_config = ConfigDict(extra="allow")

    failure_and_recovery: Dict[str, Dict[str, Union[str, List[str]]]]


class PhaseTiming(BaseModel):
    model_config = ConfigDict(extra="allow")

    max_duration_sec: Union[float, str] = 0.0
    urgency: str = "medium"


class TemporalConstraintsMap(BaseModel):
    model_config = ConfigDict(extra="allow")

    temporal_constraints: Dict[str, PhaseTiming]


class FlanaganState(TypedDict):
    instruction: str
    previous_actions: List[str]
    initial_phases: List[str]
    phases: List[str]
    phase_objects: List[str]
    phase_descriptions: List[str]
    preconditions: Dict[str, dict]
    goal_states: Dict[str, dict]
    force_dynamics: Dict[str, dict]
    sensory_feedbacks: Dict[str, dict]
    failure_and_recovery: Dict[str, dict]
    temporal_constraints: Dict[str, dict]
    flanagan: Dict


class PhasePlanner(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phases: List[Literal[
        "Approach", "Grasp", "Lift", "Transport", "Place", "Align", "Cut", "Slice",
        "Tilt", "Pour", "Insert", "Withdraw", "Release", "Reorient", "Stabilize", "Inspect"
    ]]


class PhaseReasoningOutput(BaseModel):
    """Enhanced output with reasoning chain."""

    model_config = ConfigDict(extra="forbid")

    reasoning: str
    sub_goals: List[str]
    phases: List[Literal[
        "Approach", "Grasp", "Lift", "Transport", "Place", "Align", "Cut", "Slice",
        "Tilt", "Pour", "Insert", "Withdraw", "Release", "Reorient", "Stabilize", "Inspect"
    ]]
