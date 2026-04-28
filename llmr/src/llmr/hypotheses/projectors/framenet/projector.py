"""FrameNet sidecar projector into the hypothesis graph."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable
from uuid import uuid4
from typing_extensions import Any, ClassVar, Optional

from llmr.hypotheses.elements import ClaimStatus, GroundingState, HypothesisMeta
from llmr.hypotheses.common.edges import (
    AboutActionEdge,
    GroundedByEdge,
    ProducedClaimEdge,
    SupportedByEdge,
)
from llmr.hypotheses.common.nodes import (
    ActionNode,
    InstructionNode,
    ReasonerRunNode,
    SlotBindingEvidenceNode,
    SymbolGroundingEvidenceNode,
)
from llmr.hypotheses.projectors.framenet.edges import EvokesFrameEdge, HasRoleEdge
from llmr.hypotheses.projectors.framenet.nodes import (
    FrameHypothesisNode,
    FrameRoleHypothesisNode,
)
from llmr.hypotheses.projection import (
    HypothesisProjection,
    ProjectionInput,
    HypothesisProjector,
)
from llmr.hypotheses.projectors.framenet.constants import FRAMENET_PROMPT_VERSION, FRAMENET_REASONER_NAME

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
class FrameNetProjector(HypothesisProjector):
    """Project FrameNet sidecar output into a typed hypothesis subgraph."""

    REASONER_NAME: ClassVar[str] = FRAMENET_REASONER_NAME
    PROMPT_VERSION: ClassVar[str] = FRAMENET_PROMPT_VERSION

    def supports(self, context: ProjectionInput) -> bool:
        return getattr(context.semantics, "frames", None) is not None

    def project(self, context: ProjectionInput) -> HypothesisProjection:
        frames = getattr(context.semantics, "frames", None)
        if frames is None:
            return HypothesisProjection(nodes=[], edges=[])

        run_id = uuid4().hex
        warnings: list[str] = []

        instruction_node = self._build_instruction_node(context)
        action_node = self._build_action_node(context)
        run_node = self._build_reasoner_run_node(context, run_id)
        frame_meta = self._make_meta(
            run_id=run_id,
            status=ClaimStatus.HYPOTHESIS,
            grounding=GroundingState.TEXT_ONLY,
            model_name=context.llm_model_name,
        )
        frame_node = self._build_frame_node(context, run_id, frame_meta, frames)

        nodes = [instruction_node, action_node, run_node, frame_node]
        edges = [
            EvokesFrameEdge(
                id=self._edge_id(run_id, "instruction_to_frame"),
                meta=frame_meta,
                src_id=instruction_node.id,
                dst_id=frame_node.id,
            ),
            AboutActionEdge(
                id=self._edge_id(run_id, "frame_to_action"),
                meta=frame_meta,
                src_id=frame_node.id,
                dst_id=action_node.id,
            ),
            ProducedClaimEdge(
                id=self._edge_id(run_id, "run_to_frame"),
                meta=frame_meta,
                src_id=run_node.id,
                dst_id=frame_node.id,
            ),
        ]

        for role_family, role_name, filler_text in self._iter_role_entries(frames):
            matches = self._find_resolved_slot_matches(filler_text, context)
            if len(matches) > 1:
                warnings.append(
                    f"ambiguous resolved slot alignment for role '{role_name}' and filler {filler_text!r}"
                )
            matched_slot = matches[0] if len(matches) == 1 else None

            support_node = self._maybe_slot_support(matched_slot, run_id, context)
            grounding_node = self._maybe_symbol_grounding(
                role_name=role_name,
                filler_text=filler_text,
                matched_slot=matched_slot,
                run_id=run_id,
                context=context,
            )

            if grounding_node is not None:
                role_status = ClaimStatus.SUPPORTED
                role_grounding = GroundingState.SYMBOL_GROUNDED
            elif support_node is not None:
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
            role_node = FrameRoleHypothesisNode(
                id=self._node_id(run_id, f"role:{role_family}:{role_name}"),
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
            nodes.append(role_node)
            edges.extend(
                [
                    HasRoleEdge(
                        id=self._edge_id(
                            run_id, f"frame_to_role:{role_family}:{role_name}"
                        ),
                        meta=role_meta,
                        src_id=frame_node.id,
                        dst_id=role_node.id,
                    ),
                    ProducedClaimEdge(
                        id=self._edge_id(
                            run_id, f"run_to_role:{role_family}:{role_name}"
                        ),
                        meta=role_meta,
                        src_id=run_node.id,
                        dst_id=role_node.id,
                    ),
                ]
            )

            if support_node is not None:
                nodes.append(support_node)
                edges.append(
                    SupportedByEdge(
                        id=self._edge_id(
                            run_id, f"role_supported_by:{role_family}:{role_name}"
                        ),
                        meta=support_node.meta,
                        src_id=role_node.id,
                        dst_id=support_node.id,
                    )
                )
            if grounding_node is not None:
                nodes.append(grounding_node)
                edges.append(
                    GroundedByEdge(
                        id=self._edge_id(
                            run_id, f"role_grounded_by:{role_family}:{role_name}"
                        ),
                        meta=grounding_node.meta,
                        src_id=role_node.id,
                        dst_id=grounding_node.id,
                    )
                )

        return HypothesisProjection(nodes=nodes, edges=edges, warnings=warnings)

    def _build_instruction_node(
        self, context: ProjectionInput
    ) -> InstructionNode:
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

    def _build_frame_node(
        self,
        context: ProjectionInput,
        run_id: str,
        meta: HypothesisMeta,
        frames: Any,
    ) -> FrameHypothesisNode:
        return FrameHypothesisNode(
            id=self._node_id(run_id, "frame"),
            meta=meta,
            frame=str(getattr(frames, "frame", "")),
            lexical_unit=str(getattr(frames, "lexical_unit", "")),
            framenet_label=str(getattr(frames, "framenet", "")),
            action_type=context.action_type,
            instruction_text=context.instruction,
        )

    def _iter_role_entries(
        self, frames: Any
    ) -> Iterable[tuple[str, str, str]]:
        families = (
            ("core", self._to_field_mapping(getattr(frames, "core", None))),
            (
                "peripheral",
                self._to_field_mapping(getattr(frames, "peripheral", None)),
            ),
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
        self, filler_text: str, context: ProjectionInput
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
        matched_slot: Optional[_ResolvedSlotMatch],
        run_id: str,
        context: ProjectionInput,
    ) -> Optional[SlotBindingEvidenceNode]:
        del context
        if matched_slot is None:
            return None
        return SlotBindingEvidenceNode(
            id=self._node_id(run_id, f"slot_support:{matched_slot.slot_name}"),
            meta=self._make_meta(run_id=run_id, grounding=GroundingState.SLOT_ALIGNED),
            slot_name=matched_slot.slot_name,
            value_ref=matched_slot.value_ref,
            value_repr=matched_slot.value_repr,
        )

    def _maybe_symbol_grounding(
        self,
        role_name: str,
        filler_text: str,
        matched_slot: Optional[_ResolvedSlotMatch],
        run_id: str,
        context: ProjectionInput,
    ) -> Optional[SymbolGroundingEvidenceNode]:
        del context
        if matched_slot is None or role_name not in _FRAMENET_GROUNDABLE_ROLE_NAMES:
            return None
        if not self._is_symbol_like(matched_slot.value_ref):
            return None
        return SymbolGroundingEvidenceNode(
            id=self._node_id(run_id, f"symbol_grounding:{role_name}"),
            meta=self._make_meta(run_id=run_id, grounding=GroundingState.SYMBOL_GROUNDED),
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

    def _edge_id(self, run_id: str, suffix: str) -> str:
        return f"{self.REASONER_NAME}:{run_id}:edge:{suffix}"

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
        """Single canonical string for a slot value — mirrors format_slots output.

        Uses the same priority as format_slots so that the LLM's role fillers
        (which were grounded by format_slots) match exactly here.
        """
        if isinstance(value, type):
            return value.__name__
        if hasattr(value, "name") and not isinstance(value, str):
            return str(value.name)
        from llmr.bridge.world_reader import symbol_display_name
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
