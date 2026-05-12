"""Flanagan builder for projecting reasoner output into sg_model objects."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from typing_extensions import Any, ClassVar, Optional

from llmr.hypotheses.build import BuildInput, BuildResult
from llmr.hypotheses.builders.base import HypothesisBuilder
from llmr.hypotheses.entities.flanagan import PhaseClaim, TemporalInterval
from llmr.hypotheses.graph import HypothesisGraph
from llmr.hypotheses.meta import ClaimStatus, GroundingState, HypothesisMeta

FLANAGAN_REASONER_NAME: str = "flanagan_reasoner"
FLANAGAN_PROMPT_VERSION: str = "flanagan_v1"

# Maps canonical phase name → (entry event type, exit event type).
# Entry event = force-dynamic transition that starts the phase.
# Exit event  = force-dynamic transition that ends the phase.
_PHASE_FORCE_DYNAMIC_EVENTS: dict[str, tuple[str, str]] = {
    "Approach":   ("EffectorNearObject",    "EffectorTouchesObject"),
    "Grasp":      ("EffectorTouchesObject", "GraspAchieved"),
    "Lift":       ("GraspAchieved",         "ObjectLiftOff"),
    "Transport":  ("ObjectLiftOff",         "ObjectAtDestination"),
    "Place":      ("ObjectAtDestination",   "ObjectTouchesSurface"),
    "Release":    ("ObjectStableOnSurface", "EffectorReleasesObject"),
}


@dataclass
class FlanaganBuilder(HypothesisBuilder):
    """Build a Flanagan hypothesis object cluster from reasoner output."""

    REASONER_NAME: ClassVar[str] = FLANAGAN_REASONER_NAME
    PROMPT_VERSION: ClassVar[str] = FLANAGAN_PROMPT_VERSION

    def supports(self, context: BuildInput) -> bool:
        return getattr(context.semantics, "motion_phases", None) is not None

    def build(self, context: BuildInput, graph: HypothesisGraph) -> BuildResult:
        motion_plan = getattr(context.semantics, "motion_phases", None)
        if motion_plan is None:
            return BuildResult(roots=[])

        run_id = uuid4().hex
        meta = self._make_meta(run_id=run_id, model_name=context.llm_model_name)

        instruction = graph.get_or_create_instruction(
            text=context.instruction or "",
            normalized_text=self._normalize_instruction_text(context.instruction or ""),
            meta=self._make_meta(
                run_id=None,
                grounding=GroundingState.TEXT_ONLY,
                model_name=context.llm_model_name,
            ),
        )
        action = graph.get_or_create_action(
            action_ref=context.action,
            action_type=context.action_type,
            meta=self._make_meta(
                run_id=None,
                grounding=GroundingState.TEXT_ONLY,
                model_name=context.llm_model_name,
            ),
        )
        run = graph.create_run(
            entity_id=self._node_id(run_id, "run"),
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
        plan = graph.create_plan_claim(
            entity_id=self._node_id(run_id, "plan"),
            meta=meta,
            action_type=context.action_type,
            instruction_text=context.instruction,
            phase_count=len(getattr(motion_plan, "phases", [])),
        )

        instruction.add_plan(plan)
        action.add_plan_claim(plan)
        run.add_claim(plan)

        cumulative_offset = 0.0
        for index, phase in enumerate(getattr(motion_plan, "phases", [])):
            preconditions = dict(getattr(phase, "preconditions", {}) or {})
            goal_state = dict(getattr(phase, "goal_state", {}) or {})
            failure_and_recovery = dict(
                getattr(phase, "failure_and_recovery", {}) or {}
            )
            temporal_constraints = dict(
                getattr(phase, "temporal_constraints", {}) or {}
            )
            max_dur = self._optional_float(temporal_constraints.get("max_duration_sec"))
            phase_claim = graph.create_phase_claim(
                entity_id=self._node_id(
                    run_id, f"phase:{index}:{getattr(phase, 'phase', '')}"
                ),
                meta=meta,
                phase_index=index,
                phase_name=str(getattr(phase, "phase", "")),
                target_object=str(getattr(phase, "target_object", "")),
                description=getattr(phase, "description", None),
                symbol=str(getattr(phase, "symbol", "")),
                force_dynamics=dict(getattr(phase, "force_dynamics", {}) or {}),
                sensory_feedback=dict(getattr(phase, "sensory_feedback", {}) or {}),
                contact=bool(
                    (getattr(phase, "force_dynamics", {}) or {}).get("contact", False)
                ),
                motion_type=self._optional_str(
                    (getattr(phase, "force_dynamics", {}) or {}).get("motion_type")
                ),
                max_duration_sec=max_dur,
                urgency=self._optional_str(temporal_constraints.get("urgency")),
            )
            phase_claim.temporal_interval = TemporalInterval(
                offset_from_start_sec=cumulative_offset,
                duration_sec=max_dur if max_dur is not None else 0.0,
                is_hypothetical=True,
            )
            cumulative_offset += phase_claim.temporal_interval.duration_sec
            plan.add_phase(phase_claim)
            run.add_claim(phase_claim)
            self._attach_force_dynamic_events(
                graph=graph,
                phase=phase_claim,
                run_id=run_id,
                meta=meta,
            )
            self._attach_preconditions(
                graph=graph,
                phase=phase_claim,
                run_id=run_id,
                meta=meta,
                preconditions=preconditions,
            )
            self._attach_goal_conditions(
                graph=graph,
                phase=phase_claim,
                run_id=run_id,
                meta=meta,
                goal_state=goal_state,
            )
            self._attach_failure_modes(
                graph=graph,
                phase=phase_claim,
                run_id=run_id,
                meta=meta,
                values=failure_and_recovery.get("possible_failures"),
            )
            self._attach_recovery_strategies(
                graph=graph,
                phase=phase_claim,
                run_id=run_id,
                meta=meta,
                values=failure_and_recovery.get("recovery_strategies"),
            )

        return BuildResult(roots=[plan])

    def _make_meta(
        self,
        *,
        run_id: str | None,
        status: ClaimStatus = ClaimStatus.HYPOTHESIS,
        grounding: GroundingState = GroundingState.TEXT_ONLY,
        model_name: Optional[str] = None,
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

    @staticmethod
    def _normalize_instruction_text(text: str) -> str:
        return " ".join(text.split()).strip().lower()

    @staticmethod
    def _optional_str(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _string_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
        return [str(value)]

    def _attach_force_dynamic_events(
        self,
        *,
        graph: HypothesisGraph,
        phase: PhaseClaim,
        run_id: str,
        meta: HypothesisMeta,
    ) -> None:
        entry_type, exit_type = _PHASE_FORCE_DYNAMIC_EVENTS.get(
            phase.phase_name,
            ("EffectorNearObject", "EffectorReleasesObject"),
        )
        offset = (
            phase.temporal_interval.offset_from_start_sec
            if phase.temporal_interval is not None
            else 0.0
        )
        entry = graph.create_force_dynamic_event(
            entity_id=self._node_id(run_id, f"phase:{phase.phase_index}:fde:entry"),
            meta=meta,
            event_type=entry_type,
            agent=None,
            object_ref=phase.target_object,
            expected_offset_sec=offset,
            role="entry",
        )
        exit_offset = offset + (
            phase.temporal_interval.duration_sec
            if phase.temporal_interval is not None
            else 0.0
        )
        exit_ev = graph.create_force_dynamic_event(
            entity_id=self._node_id(run_id, f"phase:{phase.phase_index}:fde:exit"),
            meta=meta,
            event_type=exit_type,
            agent=None,
            object_ref=phase.target_object,
            expected_offset_sec=exit_offset,
            role="exit",
        )
        phase.add_entry_event(entry)
        phase.add_exit_event(exit_ev)
        if phase.run is not None:
            phase.run.add_claim(entry)
            phase.run.add_claim(exit_ev)

    def _attach_preconditions(
        self,
        *,
        graph: HypothesisGraph,
        phase: Any,
        run_id: str,
        meta: HypothesisMeta,
        preconditions: dict[str, Any],
    ) -> None:
        for key, value in preconditions.items():
            for index, name, value_text, source_key in self._normalize_mapping_entry(
                key, value
            ):
                precondition = graph.create_precondition_claim(
                    entity_id=self._node_id(
                        run_id,
                        f"phase:{phase.phase_index}:precondition:{self._slug(source_key)}:{index}",
                    ),
                    meta=meta,
                    predicate_name=name,
                    object_ref=phase.target_object or None,
                    expected_value=self._parse_bool(value_text),
                    source_key=source_key,
                )
                phase.add_precondition(precondition)
                if phase.run is not None:
                    phase.run.add_claim(precondition)

    def _attach_goal_conditions(
        self,
        *,
        graph: HypothesisGraph,
        phase: Any,
        run_id: str,
        meta: HypothesisMeta,
        goal_state: dict[str, Any],
    ) -> None:
        for key, value in goal_state.items():
            for index, name, value_text, source_key in self._normalize_mapping_entry(
                key, value
            ):
                goal_condition = graph.create_goal_condition_claim(
                    entity_id=self._node_id(
                        run_id,
                        f"phase:{phase.phase_index}:goal:{self._slug(source_key)}:{index}",
                    ),
                    meta=meta,
                    predicate_name=name,
                    object_ref=phase.target_object or None,
                    expected_value=self._parse_bool(value_text),
                    source_key=source_key,
                )
                phase.add_goal_condition(goal_condition)
                if phase.run is not None:
                    phase.run.add_claim(goal_condition)

    def _attach_failure_modes(
        self,
        *,
        graph: HypothesisGraph,
        phase: Any,
        run_id: str,
        meta: HypothesisMeta,
        values: Any,
    ) -> None:
        for index, item in enumerate(self._string_list(values)):
            failure_mode = graph.create_failure_mode_claim(
                entity_id=self._node_id(
                    run_id,
                    f"phase:{phase.phase_index}:failure:{index}:{self._slug(item)}",
                ),
                meta=meta,
                name=item,
                value_text=item,
                source_key="possible_failures",
            )
            phase.add_failure_mode(failure_mode)
            if phase.run is not None:
                phase.run.add_claim(failure_mode)

    def _attach_recovery_strategies(
        self,
        *,
        graph: HypothesisGraph,
        phase: Any,
        run_id: str,
        meta: HypothesisMeta,
        values: Any,
    ) -> None:
        for index, item in enumerate(self._string_list(values)):
            recovery_strategy = graph.create_recovery_strategy_claim(
                entity_id=self._node_id(
                    run_id,
                    f"phase:{phase.phase_index}:recovery:{index}:{self._slug(item)}",
                ),
                meta=meta,
                name=item,
                value_text=item,
                source_key="recovery_strategies",
            )
            phase.add_recovery_strategy(recovery_strategy)
            if phase.run is not None:
                phase.run.add_claim(recovery_strategy)

    def _normalize_mapping_entry(
        self, key: str, value: Any
    ) -> list[tuple[int, str, str | None, str]]:
        source_key = str(key)
        if isinstance(value, dict):
            results = []
            for index, (sub_key, sub_value) in enumerate(value.items()):
                results.append(
                    (
                        index,
                        str(sub_key),
                        self._value_text(sub_value),
                        f"{source_key}.{sub_key}",
                    )
                )
            return results
        if isinstance(value, (list, tuple)):
            return [
                (index, source_key, self._value_text(item), source_key)
                for index, item in enumerate(value)
            ]
        return [(0, source_key, self._value_text(value), source_key)]

    @staticmethod
    def _value_text(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _parse_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return True
        return str(value).strip().lower() not in ("false", "no", "0", "none", "")

    @staticmethod
    def _slug(text: str) -> str:
        slug = "".join(
            character.lower() if character.isalnum() else "_"
            for character in text
        ).strip("_")
        while "__" in slug:
            slug = slug.replace("__", "_")
        return slug or "value"
