"""ActionResolutionPipeline — orchestrates Generation and Grounding."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TYPE_CHECKING

from minimal_llmr.bridge.template import (
    ActionTemplate,
    bind_parameter,
    missing_required_parameters,
    snapshot_match,
)
from minimal_llmr.bridge.world import render_world_context
from minimal_llmr.core.errors import ParameterDescriptionFailed, UnresolvedRequiredParameters
from minimal_llmr.core.schemas import ParameterInterpretations
from minimal_llmr.generation.describer import describe_parameters
from minimal_llmr.grounding.binder import bind_one
from minimal_llmr.grounding.referent import ReferentResolver

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

_UNRESOLVED = object()


@dataclass
class ActionResolutionResult:
    """Resolved executable action plus all sidecar state."""

    action: Any
    template: ActionTemplate
    interpretations: Optional[ParameterInterpretations]
    grounded_params: dict[str, Any]
    world_context: str


@dataclass
class ActionResolutionPipeline:
    """Resolve a KRROOD Match through Generation and Grounding to a concrete action."""

    llm: "BaseChatModel"
    instruction: Optional[str] = None
    strict_required: bool = False
    world_context_provider: Optional[Callable[[], str]] = None

    def resolve(self, match: Any, *, instruction: Optional[str] = None) -> ActionResolutionResult:
        template = snapshot_match(match)
        resolved_instruction = instruction if instruction is not None else self.instruction
        world_context = self._build_world_context()

        interpretations: Optional[ParameterInterpretations] = None
        grounded_params: dict[str, Any] = {}

        if template.free_parameters:
            interpretations = describe_parameters(
                instruction=resolved_instruction,
                action_cls=template.action_type,
                free_param_names=template.free_parameter_names,
                fixed_params=template.fixed_bindings,
                optional_defaults=template.optional_default_bindings or None,
                world_context=world_context,
                llm=self.llm,
            )
            if interpretations is None:
                raise ParameterDescriptionFailed(action_name=template.action_name)

            grounded_params = self._ground(template, interpretations)

        action = self._materialize(template)
        return ActionResolutionResult(
            action=action,
            template=template,
            interpretations=interpretations,
            grounded_params=grounded_params,
            world_context=world_context,
        )

    def _ground(
        self,
        template: ActionTemplate,
        interpretations: ParameterInterpretations,
    ) -> dict[str, Any]:
        resolver = ReferentResolver()
        grounded: dict[str, Any] = {}
        for param in template.free_parameters:
            value = bind_one(
                param=param,
                interpretations=interpretations,
                resolver=resolver,
                unresolved=_UNRESOLVED,
            )
            if value is _UNRESOLVED:
                logger.debug("'%s' remains unresolved", param.prompt_name)
                continue
            if bind_parameter(param, value):
                grounded[param.prompt_name] = value
        return grounded

    def _materialize(self, template: ActionTemplate) -> Any:
        if self.strict_required:
            unresolved = missing_required_parameters(template)
            if unresolved:
                raise UnresolvedRequiredParameters(
                    action_name=template.action_name, params=unresolved
                )
        template._expression._update_kwargs_from_literal_values()
        return template._expression.construct_instance()

    def _build_world_context(self) -> str:
        if self.world_context_provider is not None:
            try:
                return self.world_context_provider()
            except Exception as exc:
                logger.warning(
                    "world_context_provider raised %s; falling back to SymbolGraph", exc
                )
        return render_world_context()
