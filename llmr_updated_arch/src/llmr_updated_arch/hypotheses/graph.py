"""Repository and query-domain provider for sg_model hypothesis entities."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import cast

from typing_extensions import Any, DefaultDict, Dict, Iterator, List, Optional, TypeVar

from llmr_updated_arch.hypotheses.entities.base import ClaimHypothesis, EvidenceHypothesis, Hypothesis
from llmr_updated_arch.hypotheses.entities.common import (
    Action,
    GroundingEvidence,
    Instruction,
    ReasonerRun,
    SlotEvidence,
)
from llmr_updated_arch.hypotheses.entities.flanagan import (
    FailureModeClaim,
    ForceDynamicEvent,
    GoalConditionClaim,
    PhaseClaim,
    PlanClaim,
    PreconditionClaim,
    RecoveryStrategyClaim,
)
from llmr_updated_arch.hypotheses.entities.framenet import FrameClaim, RoleClaim
from llmr_updated_arch.hypotheses.meta import HypothesisMeta

THypothesis = TypeVar("THypothesis", bound=Hypothesis)


def _normalize_instruction_text(text: str) -> str:
    """Normalize instruction text for stable lookup and dedup keys."""

    return " ".join(text.split()).strip().lower()


@dataclass
class HypothesisGraph:
    """Explicit repository of sg_model entities.

    The repository is object-domain oriented: entities are stored directly and
    queried by type, not traversed through graph topology. Specialized indexes
    support common anchor lookups and grounding reverse lookups.
    """

    _entities_by_id: Dict[str, Hypothesis] = field(
        default_factory=dict, init=False, repr=False
    )
    _type_index: DefaultDict[type, List[str]] = field(
        default_factory=lambda: defaultdict(list), init=False, repr=False
    )
    _instruction_ids_by_normalized_text: Dict[str, str] = field(
        default_factory=dict, init=False, repr=False
    )
    _action_ids_by_ref_id: Dict[int, str] = field(
        default_factory=dict, init=False, repr=False
    )
    _grounding_ids_by_symbol_ref_id: DefaultDict[int, List[str]] = field(
        default_factory=lambda: defaultdict(list), init=False, repr=False
    )

    def add(self, entity: THypothesis) -> THypothesis:
        """Register *entity* and return the canonical repository instance."""

        existing = self._entities_by_id.get(entity.id)
        if existing is not None:
            if existing is not entity:
                raise ValueError(f"entity id collision for {entity.id!r}")
            return existing  # type: ignore[return-value]

        self._validate_special_index_keys(entity)
        self._entities_by_id[entity.id] = entity
        self._index_by_mro(entity)
        self._index_special(entity)
        return entity

    def get(self, entity_id: str) -> Optional[Hypothesis]:
        """Return the entity with *entity_id*, if present."""

        return self._entities_by_id.get(entity_id)

    def has(self, entity_id: str) -> bool:
        """Return whether an entity with *entity_id* is registered."""

        return entity_id in self._entities_by_id

    def remove(self, entity_id: str) -> Optional[Hypothesis]:
        """Remove the entity with *entity_id*, if present."""

        entity = self._entities_by_id.pop(entity_id, None)
        if entity is None:
            return None

        self._unindex_by_mro(entity)
        self._unindex_special(entity)
        return entity

    def clear(self) -> None:
        """Remove every entity and reset all repository indexes."""

        self._entities_by_id.clear()
        self._type_index.clear()
        self._instruction_ids_by_normalized_text.clear()
        self._action_ids_by_ref_id.clear()
        self._grounding_ids_by_symbol_ref_id.clear()

    def domain(self, entity_type: type[THypothesis]) -> List[THypothesis]:
        """Return registered entities of *entity_type* preserving insertion order."""

        if not isinstance(entity_type, type) or not issubclass(entity_type, Hypothesis):
            return []
        return [
            self._entities_by_id[entity_id]
            for entity_id in self._type_index.get(entity_type, [])
        ]  # type: ignore[return-value]

    def get_instances_of_type(self, cls: type[THypothesis]) -> List[THypothesis]:
        """Return the repository domain for *cls*.

        The signature matches the instance-provider shape EQL expects.
        """

        return self.domain(cls)

    def nodes_for_run(self, run_id: str) -> List[Hypothesis]:
        """Return all entities tagged with *run_id*."""

        return [entity for entity in self if entity.meta.run_id == run_id]

    def nodes_from_reasoner(self, reasoner_name: str) -> List[Hypothesis]:
        """Return all entities attributed to *reasoner_name*."""

        return [
            entity
            for entity in self
            if entity.meta.source_reasoner == reasoner_name
        ]

    def claims_for_run(self, run_id: str) -> List[ClaimHypothesis]:
        """Return registered claims tagged with *run_id*."""

        return [
            claim
            for claim in self.domain(ClaimHypothesis)
            if claim.meta.run_id == run_id
        ]

    def claims_for_action(self, action_ref: object) -> List[ClaimHypothesis]:
        """Return registered claims directly linked to *action_ref*."""

        matched: List[ClaimHypothesis] = []
        for claim in self.domain(ClaimHypothesis):
            action = getattr(claim, "action", None)
            if action is None:
                continue
            if getattr(action, "action_ref", None) is action_ref:
                matched.append(claim)
        return matched

    def get_instruction(self, text: str) -> Optional[Instruction]:
        """Return the canonical instruction entity for *text*, if present."""

        entity_id = self._instruction_ids_by_normalized_text.get(
            _normalize_instruction_text(text)
        )
        if entity_id is None:
            return None
        return cast(Instruction, self._entities_by_id[entity_id])

    def get_action(self, action_ref: Any) -> Optional[Action]:
        """Return the canonical action entity for *action_ref*, if present."""

        entity_id = self._action_ids_by_ref_id.get(id(action_ref))
        if entity_id is None:
            return None
        return cast(Action, self._entities_by_id[entity_id])

    def groundings_for_symbol(self, symbol_ref: Any) -> List[GroundingEvidence]:
        """Return all grounding evidence entities attached to *symbol_ref*."""

        return [
            cast(GroundingEvidence, self._entities_by_id[entity_id])
            for entity_id in self._grounding_ids_by_symbol_ref_id.get(
                id(symbol_ref), []
            )
        ]

    def get_or_create_instruction(
        self,
        *,
        text: str,
        meta: HypothesisMeta,
        entity_id: Optional[str] = None,
        normalized_text: Optional[str] = None,
    ) -> Instruction:
        """Return the canonical instruction anchor for *text*."""

        normalized = _normalize_instruction_text(
            text if normalized_text is None else normalized_text
        )
        existing = self.get_instruction(normalized)
        if existing is not None:
            return existing
        if entity_id is None:
            entity_id = f"instruction:{normalized}"
        return Instruction(
            id=entity_id,
            meta=meta,
            text=text,
            normalized_text=normalized,
            _register_to=self,
        )

    def get_or_create_action(
        self,
        *,
        action_ref: Any,
        action_type: str,
        meta: HypothesisMeta,
        entity_id: Optional[str] = None,
    ) -> Action:
        """Return the canonical action anchor for *action_ref*."""

        existing = self.get_action(action_ref)
        if existing is not None:
            return existing
        if entity_id is None:
            entity_id = f"action:{id(action_ref)}"
        return Action(
            id=entity_id,
            meta=meta,
            action_ref=action_ref,
            action_type=action_type,
            _register_to=self,
        )

    def create_run(
        self,
        *,
        meta: HypothesisMeta,
        reasoner_name: str,
        run_id: str,
        model_name: Optional[str],
        prompt_version: Optional[str],
        action_type: str,
        instruction_text: Optional[str],
        entity_id: Optional[str] = None,
    ) -> ReasonerRun:
        """Create and register a reasoner-run anchor."""

        if entity_id is None:
            entity_id = f"run:{reasoner_name}:{run_id}"
        return ReasonerRun(
            id=entity_id,
            meta=meta,
            reasoner_name=reasoner_name,
            run_id=run_id,
            model_name=model_name,
            prompt_version=prompt_version,
            action_type=action_type,
            instruction_text=instruction_text,
            _register_to=self,
        )

    def create_slot_evidence(
        self,
        *,
        entity_id: str,
        meta: HypothesisMeta,
        slot_name: str,
        value_ref: Any,
        value_repr: str,
    ) -> SlotEvidence:
        """Create and register slot-alignment evidence."""

        existing = self.get(entity_id)
        if existing is not None:
            if not isinstance(existing, SlotEvidence):
                raise ValueError(f"entity id collision for {entity_id!r}")
            if (
                existing.slot_name != slot_name
                or existing.value_ref is not value_ref
                or existing.value_repr != value_repr
            ):
                raise ValueError(f"slot evidence mismatch for existing id {entity_id!r}")
            return existing
        return SlotEvidence(
            id=entity_id,
            meta=meta,
            slot_name=slot_name,
            value_ref=value_ref,
            value_repr=value_repr,
            _register_to=self,
        )

    def create_grounding_evidence(
        self,
        *,
        entity_id: str,
        meta: HypothesisMeta,
        query_text: str,
        symbol_ref: Any,
        symbol_type: str,
        grounding_method: str,
        ambiguity_note: Optional[str] = None,
    ) -> GroundingEvidence:
        """Create and register symbol-grounding evidence."""

        existing = self.get(entity_id)
        if existing is not None:
            if not isinstance(existing, GroundingEvidence):
                raise ValueError(f"entity id collision for {entity_id!r}")
            if (
                existing.query_text != query_text
                or existing.symbol_ref is not symbol_ref
                or existing.symbol_type != symbol_type
                or existing.grounding_method != grounding_method
                or existing.ambiguity_note != ambiguity_note
            ):
                raise ValueError(
                    f"grounding evidence mismatch for existing id {entity_id!r}"
                )
            return existing
        return GroundingEvidence(
            id=entity_id,
            meta=meta,
            query_text=query_text,
            symbol_ref=symbol_ref,
            symbol_type=symbol_type,
            grounding_method=grounding_method,
            ambiguity_note=ambiguity_note,
            _register_to=self,
        )

    def create_frame_claim(
        self,
        *,
        entity_id: str,
        meta: HypothesisMeta,
        frame: str,
        lexical_unit: str,
        framenet_label: str,
        action_type: str,
        instruction_text: Optional[str],
    ) -> FrameClaim:
        """Create and register a FrameNet frame claim."""

        return FrameClaim(
            id=entity_id,
            meta=meta,
            frame=frame,
            lexical_unit=lexical_unit,
            framenet_label=framenet_label,
            action_type=action_type,
            instruction_text=instruction_text,
            _register_to=self,
        )

    def create_role_claim(
        self,
        *,
        entity_id: str,
        meta: HypothesisMeta,
        role_family: str,
        role_name: str,
        filler_text: str,
        filler_kind: str,
        canonical_text: Optional[str] = None,
    ) -> RoleClaim:
        """Create and register a FrameNet role claim."""

        return RoleClaim(
            id=entity_id,
            meta=meta,
            role_family=role_family,
            role_name=role_name,
            filler_text=filler_text,
            filler_kind=filler_kind,
            canonical_text=canonical_text,
            _register_to=self,
        )

    def create_plan_claim(
        self,
        *,
        entity_id: str,
        meta: HypothesisMeta,
        action_type: str,
        instruction_text: Optional[str],
        phase_count: int,
    ) -> PlanClaim:
        """Create and register a Flanagan plan claim."""

        return PlanClaim(
            id=entity_id,
            meta=meta,
            action_type=action_type,
            instruction_text=instruction_text,
            phase_count=phase_count,
            _register_to=self,
        )

    def create_phase_claim(
        self,
        *,
        entity_id: str,
        meta: HypothesisMeta,
        phase_index: int,
        phase_name: str,
        target_object: str,
        description: Optional[str],
        symbol: str,
        force_dynamics: dict[str, Any],
        sensory_feedback: dict[str, Any],
        contact: bool = False,
        motion_type: Optional[str] = None,
        max_duration_sec: Optional[float] = None,
        urgency: Optional[str] = None,
    ) -> PhaseClaim:
        """Create and register a Flanagan phase claim."""

        return PhaseClaim(
            id=entity_id,
            meta=meta,
            phase_index=phase_index,
            phase_name=phase_name,
            target_object=target_object,
            description=description,
            symbol=symbol,
            force_dynamics=force_dynamics,
            sensory_feedback=sensory_feedback,
            contact=contact,
            motion_type=motion_type,
            max_duration_sec=max_duration_sec,
            urgency=urgency,
            _register_to=self,
        )

    def create_precondition_claim(
        self,
        *,
        entity_id: str,
        meta: HypothesisMeta,
        predicate_name: str,
        subject: Optional[str] = None,
        object_ref: Optional[str] = None,
        expected_value: bool = True,
        predicate_ref: Optional[str] = None,
        source_key: Optional[str] = None,
    ) -> PreconditionClaim:
        """Create and register a Flanagan precondition with structured predicate fields."""

        return PreconditionClaim(
            id=entity_id,
            meta=meta,
            predicate_name=predicate_name,
            subject=subject,
            object_ref=object_ref,
            expected_value=expected_value,
            predicate_ref=predicate_ref,
            source_key=source_key,
            _register_to=self,
        )

    def create_goal_condition_claim(
        self,
        *,
        entity_id: str,
        meta: HypothesisMeta,
        predicate_name: str,
        subject: Optional[str] = None,
        object_ref: Optional[str] = None,
        expected_value: bool = True,
        predicate_ref: Optional[str] = None,
        source_key: Optional[str] = None,
    ) -> GoalConditionClaim:
        """Create and register a Flanagan goal condition with structured predicate fields."""

        return GoalConditionClaim(
            id=entity_id,
            meta=meta,
            predicate_name=predicate_name,
            subject=subject,
            object_ref=object_ref,
            expected_value=expected_value,
            predicate_ref=predicate_ref,
            source_key=source_key,
            _register_to=self,
        )

    def create_failure_mode_claim(
        self,
        *,
        entity_id: str,
        meta: HypothesisMeta,
        name: str,
        value_text: Optional[str] = None,
        source_key: Optional[str] = None,
    ) -> FailureModeClaim:
        """Create and register a queryable Flanagan failure mode."""

        return FailureModeClaim(
            id=entity_id,
            meta=meta,
            name=name,
            value_text=value_text,
            source_key=source_key,
            _register_to=self,
        )

    def create_recovery_strategy_claim(
        self,
        *,
        entity_id: str,
        meta: HypothesisMeta,
        name: str,
        value_text: Optional[str] = None,
        source_key: Optional[str] = None,
    ) -> RecoveryStrategyClaim:
        """Create and register a queryable Flanagan recovery strategy."""

        return RecoveryStrategyClaim(
            id=entity_id,
            meta=meta,
            name=name,
            value_text=value_text,
            source_key=source_key,
            _register_to=self,
        )

    def create_force_dynamic_event(
        self,
        *,
        entity_id: str,
        meta: HypothesisMeta,
        event_type: str,
        agent: Optional[str],
        object_ref: str,
        expected_offset_sec: Optional[float] = None,
        role: str = "entry",
    ) -> ForceDynamicEvent:
        """Create and register a force-dynamic event boundary for a phase."""

        return ForceDynamicEvent(
            id=entity_id,
            meta=meta,
            event_type=event_type,
            agent=agent,
            object_ref=object_ref,
            expected_offset_sec=expected_offset_sec,
            role=role,
            _register_to=self,
        )

    @property
    def entities(self) -> List[Hypothesis]:
        """Return all registered entities preserving insertion order."""

        return list(self._entities_by_id.values())

    @property
    def entity_count(self) -> int:
        return len(self._entities_by_id)

    @property
    def instructions(self) -> List[Instruction]:
        return self.domain(Instruction)

    @property
    def actions(self) -> List[Action]:
        return self.domain(Action)

    @property
    def reasoner_runs(self) -> List[ReasonerRun]:
        return self.domain(ReasonerRun)

    @property
    def evidences(self) -> List[EvidenceHypothesis]:
        return self.domain(EvidenceHypothesis)

    @property
    def frames(self) -> List[FrameClaim]:
        return self.domain(FrameClaim)

    @property
    def roles(self) -> List[RoleClaim]:
        return self.domain(RoleClaim)

    @property
    def plans(self) -> List[PlanClaim]:
        return self.domain(PlanClaim)

    @property
    def phases(self) -> List[PhaseClaim]:
        return self.domain(PhaseClaim)

    @property
    def preconditions(self) -> List[PreconditionClaim]:
        return self.domain(PreconditionClaim)

    @property
    def goal_conditions(self) -> List[GoalConditionClaim]:
        return self.domain(GoalConditionClaim)

    @property
    def failure_modes(self) -> List[FailureModeClaim]:
        return self.domain(FailureModeClaim)

    @property
    def recovery_strategies(self) -> List[RecoveryStrategyClaim]:
        return self.domain(RecoveryStrategyClaim)

    def __iter__(self) -> Iterator[Hypothesis]:
        """Yield all registered entities in insertion order."""

        return iter(self._entities_by_id.values())

    def __len__(self) -> int:
        return len(self._entities_by_id)

    # ------------------------------------------------------------------
    # Internal indexing helpers
    # ------------------------------------------------------------------

    def _validate_special_index_keys(self, entity: Hypothesis) -> None:
        if self._is_instruction_like(entity):
            normalized_text = _normalize_instruction_text(
                str(getattr(entity, "normalized_text"))
            )
            existing_id = self._instruction_ids_by_normalized_text.get(normalized_text)
            if existing_id is not None and existing_id != entity.id:
                raise ValueError(
                    "duplicate instruction anchor for normalized_text "
                    f"{normalized_text!r}; use repository get-or-create semantics"
                )

        if self._is_action_like(entity):
            action_ref_id = id(getattr(entity, "action_ref"))
            existing_id = self._action_ids_by_ref_id.get(action_ref_id)
            if existing_id is not None and existing_id != entity.id:
                raise ValueError(
                    "duplicate action anchor for action_ref "
                    f"{action_ref_id!r}; use repository get-or-create semantics"
                )

    def _index_by_mro(self, entity: Hypothesis) -> None:
        for cls in type(entity).__mro__:
            if cls is object or not issubclass(cls, Hypothesis):
                continue
            self._type_index[cls].append(entity.id)

    def _unindex_by_mro(self, entity: Hypothesis) -> None:
        for cls in type(entity).__mro__:
            if cls is object or not issubclass(cls, Hypothesis):
                continue
            bucket = self._type_index.get(cls)
            if bucket is None:
                continue
            try:
                bucket.remove(entity.id)
            except ValueError:
                continue
            if not bucket:
                self._type_index.pop(cls, None)

    def _index_special(self, entity: Hypothesis) -> None:
        if self._is_instruction_like(entity):
            normalized_text = _normalize_instruction_text(
                str(getattr(entity, "normalized_text"))
            )
            self._instruction_ids_by_normalized_text[normalized_text] = entity.id

        if self._is_action_like(entity):
            action_ref_id = id(getattr(entity, "action_ref"))
            self._action_ids_by_ref_id[action_ref_id] = entity.id

        if self._is_grounding_like(entity):
            symbol_ref_id = id(getattr(entity, "symbol_ref"))
            self._grounding_ids_by_symbol_ref_id[symbol_ref_id].append(entity.id)

    def _unindex_special(self, entity: Hypothesis) -> None:
        if self._is_instruction_like(entity):
            normalized_text = _normalize_instruction_text(
                str(getattr(entity, "normalized_text"))
            )
            self._instruction_ids_by_normalized_text.pop(normalized_text, None)

        if self._is_action_like(entity):
            action_ref_id = id(getattr(entity, "action_ref"))
            self._action_ids_by_ref_id.pop(action_ref_id, None)

        if self._is_grounding_like(entity):
            symbol_ref_id = id(getattr(entity, "symbol_ref"))
            bucket = self._grounding_ids_by_symbol_ref_id.get(symbol_ref_id)
            if bucket is None:
                return
            try:
                bucket.remove(entity.id)
            except ValueError:
                return
            if not bucket:
                self._grounding_ids_by_symbol_ref_id.pop(symbol_ref_id, None)

    @staticmethod
    def _is_instruction_like(entity: Hypothesis) -> bool:
        return hasattr(entity, "text") and hasattr(entity, "normalized_text")

    @staticmethod
    def _is_action_like(entity: Hypothesis) -> bool:
        return hasattr(entity, "action_ref") and hasattr(entity, "action_type")

    @staticmethod
    def _is_grounding_like(entity: Hypothesis) -> bool:
        return hasattr(entity, "symbol_ref") and hasattr(entity, "grounding_method")
