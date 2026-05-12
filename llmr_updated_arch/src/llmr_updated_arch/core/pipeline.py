"""Fixed orchestration sequence for action resolution."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from typing_extensions import Any, Callable, Iterable, Optional

from krrood.symbol_graph.symbol_graph import Symbol

from llmr_updated_arch.core.context import ResolutionContext, ResolutionOptions
from llmr_updated_arch.core.contracts import SemanticGenerator
from llmr_updated_arch.core.errors import LLMSlotFillingFailed
from llmr_updated_arch.core.result import ActionResolutionResult
from llmr_updated_arch.generation.slot_semantics import SlotSemanticsGenerator
from llmr_updated_arch.grounding.pipeline import GroundingPipeline
from llmr_updated_arch.integrations.krrood.introspect import ActionFieldIntrospector
from llmr_updated_arch.integrations.krrood.match_reader import MatchSnapshot, snapshot_match
from llmr_updated_arch.integrations.krrood.world_reader import render_world_context
from llmr_updated_arch.materialization.action import ActionMaterializer
from llmr_updated_arch.schemas import ActionClassificationResult, SemanticBundle

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from llmr_updated_arch.generation import Reasoner
    from llmr_updated_arch.hypotheses import BuildOrchestrator, BuildResult

logger = logging.getLogger(__name__)


class _UnresolvedSentinel:
    def __repr__(self) -> str:
        return "<UNRESOLVED>"


UNRESOLVED = _UnresolvedSentinel()


@dataclass
class ActionResolutionPipeline:
    """Resolve a KRROOD ``Match`` through generation, grounding, and materialization."""

    llm: "BaseChatModel"
    symbol_type: type = Symbol
    instruction: Optional[str] = None
    world_context_provider: Optional[Callable[[], str]] = None
    strict_required: bool = False
    classification: Optional[ActionClassificationResult] = None
    semantic_generators: list[SemanticGenerator] = field(
        default_factory=lambda: [SlotSemanticsGenerator()]
    )
    grounding_pipeline: GroundingPipeline = field(default_factory=GroundingPipeline)
    reasoners: list["Reasoner"] = field(default_factory=list)
    sg_model_orchestrator: Optional["BuildOrchestrator"] = None
    introspector: ActionFieldIntrospector = field(default_factory=ActionFieldIntrospector)

    def resolve(self, match: Any, *, instruction: Optional[str] = None) -> ActionResolutionResult:
        """Resolve *match* and return the concrete action plus sidecars."""

        context = self._prepare_context(match, instruction=instruction)
        match_snapshot = context.match_snapshot

        if match_snapshot.free_slots:
            self._generate_semantics(context)
            if context.semantic_bundle.slot_filling is None:
                raise LLMSlotFillingFailed(action_name=match_snapshot.action_name)
            grounded_slots = self.grounding_pipeline.ground(context, UNRESOLVED)
        else:
            grounded_slots = {}

        self._run_post_grounding_reasoners(context)
        action = ActionMaterializer(self.introspector).materialize(
            match_snapshot,
            strict_required=context.options.strict_required,
        )

        result = ActionResolutionResult(
            action=action,
            match_snapshot=match_snapshot,
            semantic_bundle=context.semantic_bundle,
            grounded_slots=grounded_slots,
            world_context=context.world_context,
        )
        result.projection_result = self._project(result)
        return result

    def _prepare_context(
        self,
        match: Any,
        *,
        instruction: Optional[str],
    ) -> ResolutionContext:
        match_snapshot = snapshot_match(match, self.introspector, unresolved=UNRESOLVED)
        resolved_instruction = self.instruction if instruction is None else instruction
        options = ResolutionOptions(
            strict_required=self.strict_required,
            world_context_provider=self.world_context_provider,
            classification=self.classification,
            reasoners=list(self.reasoners),
            sg_model_orchestrator=self.sg_model_orchestrator,
        )
        semantic_bundle = SemanticBundle(
            action_type=match_snapshot.action_name,
            instruction=resolved_instruction,
            classification=self.classification,
        )
        world_context = self._build_world_context(options)
        return ResolutionContext(
            instruction=resolved_instruction,
            match_snapshot=match_snapshot,
            world_context=world_context,
            llm=self.llm,
            symbol_type=self.symbol_type,
            semantic_bundle=semantic_bundle,
            options=options,
        )

    def _generate_semantics(self, context: ResolutionContext) -> None:
        for generator in self.semantic_generators:
            artifact = generator.generate(context)
            if artifact is None:
                continue
            context.semantic_bundle.put_artifact(
                artifact.generator_name,
                artifact.artifact_type,
                artifact.value,
            )
            if hasattr(context.semantic_bundle, artifact.artifact_type):
                setattr(context.semantic_bundle, artifact.artifact_type, artifact.value)

    def _run_post_grounding_reasoners(self, context: ResolutionContext) -> None:
        for reasoner in context.options.reasoners:
            try:
                reasoner.annotate(
                    context.semantic_bundle,
                    context.match_snapshot,
                    context.world_context,
                )
            except Exception as exc:
                logger.warning(
                    "ActionResolutionPipeline: reasoner %r raised %s - annotation skipped.",
                    reasoner,
                    exc,
                )

    def _build_world_context(self, options: ResolutionOptions) -> str:
        if options.world_context_provider is not None:
            try:
                return options.world_context_provider()
            except Exception as exc:
                logger.warning(
                    "ActionResolutionPipeline: world_context_provider raised %s; falling back.",
                    exc,
                )
        return render_world_context(self.symbol_type)

    def _project(self, result: ActionResolutionResult) -> Optional["BuildResult"]:
        from llmr_updated_arch.hypotheses import BuildInput, BuildOrchestrator

        orchestrator = self.sg_model_orchestrator
        if orchestrator is None:
            orchestrator = BuildOrchestrator.with_default_builders()
            self.sg_model_orchestrator = orchestrator

        resolved_slots = {
            slot.prompt_name: slot.value
            for slot in result.match_snapshot.slots
            if not slot.is_free
        }
        try:
            build_result = orchestrator.build(
                BuildInput(
                    instruction=result.semantic_bundle.instruction,
                    action=result.action,
                    action_type=result.match_snapshot.action_name,
                    semantics=result.semantic_bundle,
                    match_data=result.match_snapshot,
                    resolved_slots=resolved_slots,
                    world_context=result.world_context,
                    symbol_type=self.symbol_type,
                    llm_model_name=self._infer_model_name(),
                )
            )
            for warning in build_result.warnings:
                logger.warning("ActionResolutionPipeline: projection warning - %s", warning)
            return build_result
        except Exception as exc:
            logger.warning(
                "ActionResolutionPipeline: hypothesis projection raised %s - skipped.",
                exc,
            )
            return None

    def _infer_model_name(self) -> Optional[str]:
        for attr_name in ("model_name", "model", "model_id"):
            value = getattr(self.llm, attr_name, None)
            if isinstance(value, str) and value:
                return value
        return type(self.llm).__name__ or None
