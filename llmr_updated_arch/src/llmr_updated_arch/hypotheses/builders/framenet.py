"""FrameNet builder for projecting reasoner output into sg_model objects."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable
from uuid import uuid4

from typing_extensions import Any, ClassVar, Optional

from llmr_updated_arch.hypotheses.build import BuildInput, BuildResult
from llmr_updated_arch.hypotheses.builders.base import HypothesisBuilder
from llmr_updated_arch.hypotheses.entities.framenet import FrameClaim, RoleClaim
from llmr_updated_arch.hypotheses.graph import HypothesisGraph
from llmr_updated_arch.hypotheses.meta import ClaimStatus, GroundingState, HypothesisMeta

FRAMENET_REASONER_NAME: str = "framenet_reasoner"
FRAMENET_PROMPT_VERSION: str = "framenet_v1"

_FRAMENET_GROUNDABLE_ROLE_NAMES = frozenset(
    {
        "agent",
        "theme",
        "patient",
        "instrument",
        "source",
        "goal",
        "location",
    }
)
_FRAMENET_ENTITY_ROLE_NAMES = frozenset({"agent", "theme", "patient", "instrument"})
_FRAMENET_PLACE_ROLE_NAMES = frozenset({"source", "goal", "location"})
_FRAMENET_MOTION_ROLE_NAMES = frozenset({"direction", "path"})
_FRAMENET_MANNER_ROLE_NAMES = frozenset({"manner", "speed"})
_FRAMENET_STATE_ROLE_NAMES = frozenset({"result"})
_ARTICLES = frozenset({"a", "an", "the"})


@dataclass(frozen=True)
class _ResolvedSlotMatch:
    """One unambiguous resolved-slot alignment candidate."""

    slot_name: str
    value_ref: Any
    value_repr: str


@dataclass
class FrameNetBuilder(HypothesisBuilder):
    """Build a FrameNet hypothesis object cluster from reasoner output."""

    REASONER_NAME: ClassVar[str] = FRAMENET_REASONER_NAME
    PROMPT_VERSION: ClassVar[str] = FRAMENET_PROMPT_VERSION

    def supports(self, context: BuildInput) -> bool:
        return getattr(context.semantics, "frames", None) is not None

    def build(self, context: BuildInput, graph: HypothesisGraph) -> BuildResult:
        frames = getattr(context.semantics, "frames", None)
        if frames is None:
            return BuildResult(roots=[])

        run_id = uuid4().hex
        warnings: list[str] = []

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

        frame_meta = self._make_meta(
            run_id=run_id,
            status=ClaimStatus.HYPOTHESIS,
            grounding=GroundingState.TEXT_ONLY,
            model_name=context.llm_model_name,
        )
        frame = graph.create_frame_claim(
            entity_id=self._node_id(run_id, "frame"),
            meta=frame_meta,
            frame=str(getattr(frames, "frame", "")),
            lexical_unit=str(getattr(frames, "lexical_unit", "")),
            framenet_label=str(getattr(frames, "framenet", "")),
            action_type=context.action_type,
            instruction_text=context.instruction,
        )
        instruction.add_frame(frame)
        action.add_frame_claim(frame)
        run.add_claim(frame)

        for role_family, role_name, filler_text in self._iter_role_entries(frames):
            matches = self._find_resolved_slot_matches(filler_text, context)
            if len(matches) > 1:
                warnings.append(
                    f"ambiguous resolved slot alignment for role '{role_name}' and filler {filler_text!r}"
                )
            matched_slot = matches[0] if len(matches) == 1 else None

            support = self._maybe_slot_support(
                matched_slot=matched_slot,
                run_id=run_id,
                context=context,
                graph=graph,
            )
            grounding = self._maybe_symbol_grounding(
                role_name=role_name,
                filler_text=filler_text,
                matched_slot=matched_slot,
                run_id=run_id,
                context=context,
                graph=graph,
            )

            if grounding is not None:
                role_status = ClaimStatus.SUPPORTED
                role_grounding = GroundingState.SYMBOL_GROUNDED
            elif support is not None:
                role_status = ClaimStatus.SUPPORTED
                role_grounding = GroundingState.SLOT_ALIGNED
            else:
                role_status = ClaimStatus.HYPOTHESIS
                role_grounding = GroundingState.TEXT_ONLY

            role_meta = self._make_meta(
                run_id=run_id,
                status=role_status,
                grounding=role_grounding,
                model_name=context.llm_model_name,
            )
            role = graph.create_role_claim(
                entity_id=self._node_id(run_id, f"role:{role_family}:{role_name}"),
                meta=role_meta,
                role_family=role_family,
                role_name=role_name,
                filler_text=filler_text,
                filler_kind=self._infer_filler_kind(
                    role_family=role_family,
                    role_name=role_name,
                    filler_text=filler_text,
                ),
                canonical_text=self._normalize_text(filler_text),
            )
            frame.add_role(role)
            run.add_claim(role)

            if support is not None:
                role.add_support(support)
            if grounding is not None:
                role.add_grounding(grounding)

        return BuildResult(roots=[frame], warnings=warnings)

    def _iter_role_entries(self, frames: Any) -> Iterable[tuple[str, str, str]]:
        families = (
            ("core", self._to_field_mapping(getattr(frames, "core", None))),
            ("peripheral", self._to_field_mapping(getattr(frames, "peripheral", None))),
        )
        for role_family, fields in families:
            for role_name, value in fields.items():
                if value is None:
                    continue
                if isinstance(value, str):
                    stripped = value.strip()
                    if not stripped or stripped.lower() == "null":
                        continue
                    yield role_family, role_name, stripped
                    continue
                yield role_family, role_name, str(value)

    def _infer_filler_kind(
        self, role_family: str, role_name: str, filler_text: str
    ) -> str:
        del role_family, filler_text
        if role_name in _FRAMENET_ENTITY_ROLE_NAMES:
            return "entity"
        if role_name in _FRAMENET_PLACE_ROLE_NAMES:
            return "place"
        if role_name in _FRAMENET_MOTION_ROLE_NAMES:
            return "motion"
        if role_name in _FRAMENET_MANNER_ROLE_NAMES:
            return "manner"
        if role_name in _FRAMENET_STATE_ROLE_NAMES:
            return "state"
        return "abstract"

    def _find_resolved_slot_matches(
        self, filler_text: str, context: BuildInput
    ) -> list[_ResolvedSlotMatch]:
        normalized_filler = self._normalize_text(filler_text)
        if not normalized_filler:
            return []

        matches: list[_ResolvedSlotMatch] = []
        for slot_name, value in context.resolved_slots.items():
            primary = self._primary_text(value)
            if self._normalize_text(primary) == normalized_filler:
                matches.append(
                    _ResolvedSlotMatch(
                        slot_name=slot_name,
                        value_ref=value,
                        value_repr=primary,
                    )
                )
        return matches

    def _maybe_slot_support(
        self,
        *,
        matched_slot: Optional[_ResolvedSlotMatch],
        run_id: str,
        context: BuildInput,
        graph: HypothesisGraph,
    ):
        del context
        if matched_slot is None:
            return None
        return graph.create_slot_evidence(
            entity_id=self._node_id(run_id, f"slot_support:{matched_slot.slot_name}"),
            meta=self._make_meta(
                run_id=run_id,
                grounding=GroundingState.SLOT_ALIGNED,
            ),
            slot_name=matched_slot.slot_name,
            value_ref=matched_slot.value_ref,
            value_repr=matched_slot.value_repr,
        )

    def _maybe_symbol_grounding(
        self,
        *,
        role_name: str,
        filler_text: str,
        matched_slot: Optional[_ResolvedSlotMatch],
        run_id: str,
        context: BuildInput,
        graph: HypothesisGraph,
    ):
        del context
        if matched_slot is None or role_name not in _FRAMENET_GROUNDABLE_ROLE_NAMES:
            return None
        if not self._is_symbol_like(matched_slot.value_ref):
            return None
        return graph.create_grounding_evidence(
            entity_id=self._node_id(run_id, f"symbol_grounding:{role_name}"),
            meta=self._make_meta(
                run_id=run_id,
                grounding=GroundingState.SYMBOL_GROUNDED,
            ),
            query_text=filler_text,
            symbol_ref=matched_slot.value_ref,
            symbol_type=type(matched_slot.value_ref).__name__,
            grounding_method="resolved_slot_symbol_match",
            ambiguity_note=None,
        )

    def _make_meta(
        self,
        *,
        run_id: Optional[str],
        status: ClaimStatus = ClaimStatus.SUPPORTED,
        grounding: GroundingState,
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
    def _to_field_mapping(value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            return model_dump()
        if isinstance(value, dict):
            return dict(value)
        return {
            name: item
            for name, item in vars(value).items()
            if not name.startswith("_")
        }

    @staticmethod
    def _primary_text(value: Any) -> str:
        """Single canonical string for a slot value.

        This mirrors the current projector's priority order so role fillers and
        resolved slot values normalize to the same text.
        """

        if isinstance(value, type):
            return value.__name__
        if hasattr(value, "name") and not isinstance(value, str):
            return str(value.name)
        from llmr_updated_arch.integrations.krrood.world_reader import symbol_display_name

        display = symbol_display_name(value)
        if display:
            return display
        text = str(value).strip()
        return text if text and "object at 0x" not in text else repr(value)

    @staticmethod
    def _is_symbol_like(value: Any) -> bool:
        primitive_types = (str, bytes, int, float, bool, list, tuple, set, dict)
        return value is not None and not isinstance(value, primitive_types)

    @staticmethod
    def _normalize_instruction_text(text: str) -> str:
        return " ".join(text.split()).strip().lower()

    @staticmethod
    def _normalize_text(text: str) -> str:
        lowered = re.sub(r"[\W_]+", " ", text.lower()).strip()
        tokens = [token for token in lowered.split() if token]
        while tokens and tokens[0] in _ARTICLES:
            tokens.pop(0)
        return " ".join(tokens)
