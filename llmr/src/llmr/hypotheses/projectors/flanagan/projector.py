"""Flanagan sidecar projector into the hypothesis graph."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4
from typing_extensions import Any, ClassVar

from llmr.hypotheses.elements import ClaimStatus, GroundingState, HypothesisMeta
from llmr.hypotheses.common.edges import AboutActionEdge, ProducedClaimEdge
from llmr.hypotheses.common.nodes import (
    ActionNode,
    InstructionNode,
    ReasonerRunNode,
)
from llmr.hypotheses.projectors.flanagan.constants import (
    FLANAGAN_PROMPT_VERSION,
    FLANAGAN_REASONER_NAME,
)
from llmr.hypotheses.projectors.flanagan.edges import (
    EvokesMotionPlanEdge,
    HasMotionPhaseEdge,
)
from llmr.hypotheses.projectors.flanagan.nodes import (
    MotionPhaseHypothesisNode,
    MotionPlanHypothesisNode,
)
from llmr.hypotheses.projection import (
    HypothesisProjection,
    HypothesisProjector,
    ProjectionInput,
)


@dataclass
class FlanaganProjector(HypothesisProjector):
    """Project Flanagan motion phases into a typed hypothesis subgraph."""

    REASONER_NAME: ClassVar[str] = FLANAGAN_REASONER_NAME
    PROMPT_VERSION: ClassVar[str] = FLANAGAN_PROMPT_VERSION

    def supports(self, context: ProjectionInput) -> bool:
        return getattr(context.semantics, "motion_phases", None) is not None

    def project(self, context: ProjectionInput) -> HypothesisProjection:
        motion_plan = getattr(context.semantics, "motion_phases", None)
        if motion_plan is None:
            return HypothesisProjection(nodes=[], edges=[])

        run_id = uuid4().hex
        meta = self._make_meta(run_id=run_id, model_name=context.llm_model_name)

        instruction_node = self._build_instruction_node(context)
        action_node = self._build_action_node(context)
        run_node = self._build_reasoner_run_node(context, run_id)
        plan_node = MotionPlanHypothesisNode(
            id=self._node_id(run_id, "plan"),
            meta=meta,
            action_type=context.action_type,
            instruction_text=context.instruction,
            phase_count=len(getattr(motion_plan, "phases", [])),
        )

        nodes = [instruction_node, action_node, run_node, plan_node]
        edges = [
            EvokesMotionPlanEdge(
                id=self._edge_id(run_id, "instruction_to_plan"),
                meta=meta,
                src_id=instruction_node.id,
                dst_id=plan_node.id,
            ),
            AboutActionEdge(
                id=self._edge_id(run_id, "plan_to_action"),
                meta=meta,
                src_id=plan_node.id,
                dst_id=action_node.id,
            ),
            ProducedClaimEdge(
                id=self._edge_id(run_id, "run_to_plan"),
                meta=meta,
                src_id=run_node.id,
                dst_id=plan_node.id,
            ),
        ]

        for index, phase in enumerate(getattr(motion_plan, "phases", [])):
            phase_node = MotionPhaseHypothesisNode(
                id=self._node_id(run_id, f"phase:{index}:{getattr(phase, 'phase', '')}"),
                meta=meta,
                phase_index=index,
                phase_name=str(getattr(phase, "phase", "")),
                target_object=str(getattr(phase, "target_object", "")),
                description=getattr(phase, "description", None),
                symbol=str(getattr(phase, "symbol", "")),
                preconditions=dict(getattr(phase, "preconditions", {}) or {}),
                goal_state=dict(getattr(phase, "goal_state", {}) or {}),
                force_dynamics=dict(getattr(phase, "force_dynamics", {}) or {}),
                sensory_feedback=dict(getattr(phase, "sensory_feedback", {}) or {}),
                failure_and_recovery=dict(
                    getattr(phase, "failure_and_recovery", {}) or {}
                ),
                temporal_constraints=dict(
                    getattr(phase, "temporal_constraints", {}) or {}
                ),
                contact=bool(
                    (getattr(phase, "force_dynamics", {}) or {}).get("contact", False)
                ),
                motion_type=self._optional_str(
                    (getattr(phase, "force_dynamics", {}) or {}).get("motion_type")
                ),
                max_duration_sec=self._optional_float(
                    (getattr(phase, "temporal_constraints", {}) or {}).get(
                        "max_duration_sec"
                    )
                ),
                urgency=self._optional_str(
                    (getattr(phase, "temporal_constraints", {}) or {}).get("urgency")
                ),
                possible_failures=tuple(
                    self._string_list(
                        (getattr(phase, "failure_and_recovery", {}) or {}).get(
                            "possible_failures"
                        )
                    )
                ),
                recovery_strategies=tuple(
                    self._string_list(
                        (getattr(phase, "failure_and_recovery", {}) or {}).get(
                            "recovery_strategies"
                        )
                    )
                ),
            )
            nodes.append(phase_node)
            edges.extend(
                [
                    HasMotionPhaseEdge(
                        id=self._edge_id(run_id, f"plan_to_phase:{index}"),
                        meta=meta,
                        src_id=plan_node.id,
                        dst_id=phase_node.id,
                    ),
                    ProducedClaimEdge(
                        id=self._edge_id(run_id, f"run_to_phase:{index}"),
                        meta=meta,
                        src_id=run_node.id,
                        dst_id=phase_node.id,
                    ),
                ]
            )

        return HypothesisProjection(nodes=nodes, edges=edges)

    def _build_instruction_node(self, context: ProjectionInput) -> InstructionNode:
        text = context.instruction or ""
        normalized_text = self._normalize_instruction_text(text)
        return InstructionNode(
            id=f"instruction:{normalized_text}",
            meta=self._make_meta(
                run_id=None,
                grounding=GroundingState.TEXT_ONLY,
                model_name=context.llm_model_name,
            ),
            text=text,
            normalized_text=normalized_text,
        )

    def _build_action_node(self, context: ProjectionInput) -> ActionNode:
        return ActionNode(
            id=f"action:{id(context.action)}",
            meta=self._make_meta(
                run_id=None,
                grounding=GroundingState.TEXT_ONLY,
                model_name=context.llm_model_name,
            ),
            action_ref=context.action,
            action_type=context.action_type,
        )

    def _build_reasoner_run_node(
        self, context: ProjectionInput, run_id: str
    ) -> ReasonerRunNode:
        return ReasonerRunNode(
            id=self._node_id(run_id, "run"),
            meta=self._make_meta(
                run_id=run_id,
                grounding=GroundingState.TEXT_ONLY,
                model_name=context.llm_model_name,
            ),
            reasoner_name=self.REASONER_NAME,
            run_id=run_id,
            model_name=context.llm_model_name,
            prompt_version=self.PROMPT_VERSION,
            action_type=context.action_type,
            instruction_text=context.instruction,
        )

    def _make_meta(
        self,
        *,
        run_id: str | None,
        status: ClaimStatus = ClaimStatus.HYPOTHESIS,
        grounding: GroundingState = GroundingState.TEXT_ONLY,
        model_name: str | None = None,
    ) -> HypothesisMeta:
        return HypothesisMeta(
            source_reasoner=self.REASONER_NAME,
            status=status,
            grounding=grounding,
            run_id=run_id,
            prompt_version=self.PROMPT_VERSION,
            model_name=model_name,
        )

    def _node_id(self, run_id: str, suffix: str) -> str:
        return f"{self.REASONER_NAME}:{run_id}:node:{suffix}"

    def _edge_id(self, run_id: str, suffix: str) -> str:
        return f"{self.REASONER_NAME}:{run_id}:edge:{suffix}"

    def _normalize_instruction_text(self, text: str) -> str:
        return " ".join(text.split()).strip().lower()

    def _optional_str(self, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _optional_float(self, value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _string_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
        return [str(value)]
