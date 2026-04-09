"""krrood.llmr.pycram_bridge — pycram integration layer.
"""

from __future__ import annotations

import dataclasses
import logging
from typing_extensions import Any, Dict, List, Optional, Type

from krrood.llmr.pipeline.action_pipeline import ActionPipeline
from krrood.llmr.pipeline.dispatcher import ActionDispatcher, ActionSpec
from krrood.llmr.pipeline.entity_grounder import EntityGrounder
from krrood.llmr.pycram_bridge.action_registry import PycramActionRegistry
from krrood.llmr.pycram_bridge.auto_handler import AutoActionHandler
from krrood.llmr.pycram_bridge.introspector import ActionSchema, PycramIntrospector

logger = logging.getLogger(__name__)


class PycramBridge:
    """One-stop entry point for the krrood.llmr ↔ pycram integration.

    1. Discovers all pycram action classes (or accepts a manual list).
    2. Introspects each into an :class:`ActionSchema`.
    3. Registers an :class:`AutoActionHandler` for each action type.
    4. Provides :meth:`make_pipeline` to build a fully-wired
       :class:`ActionPipeline`.
    5. Provides :meth:`to_pycram_action` to convert an :class:`ActionSpec`
       into a concrete pycram action instance ready for ``.perform()``.

    :param groundable_type: Symbol subclass for SymbolGraph grounding (e.g. Body).
    :param context: Caller context dict forwarded to handlers.  Must contain:

        - ``manipulators`` (Dict[str, Manipulator]): arm-name → Manipulator.
          E.g. ``{"LEFT": robot.left_arm.manipulator, "RIGHT": ...}``.

        Optionally:
        - ``world_context`` (str): pre-built world context string (otherwise
          serialised from SymbolGraph automatically by the pipeline).

    :param packages: Extra pycram action packages to scan beyond the default
        ``pycram.robot_plans.actions.core``.
    :param action_classes: Explicit action classes to register (instead of / in
        addition to auto-discovery).
    """

    def __init__(
        self,
        groundable_type: type,
        context: Optional[Dict[str, Any]] = None,
        packages: Optional[List[str]] = None,
        action_classes: Optional[List[type]] = None,
    ) -> None:
        self._groundable_type = groundable_type
        self._context = context or {}
        self._registry = PycramActionRegistry()

        # Discover from default package
        self._registry.discover()

        # Scan additional packages
        for pkg in (packages or []):
            self._registry.discover(package=pkg)

        # Register any explicitly provided classes
        for cls in (action_classes or []):
            self._registry.register(cls)

        # Wire AutoActionHandler into ActionDispatcher for each discovered action
        self._grounder = EntityGrounder(groundable_type)
        self._handler_classes: Dict[str, type] = {}  # populated by _register_auto_handlers
        self._register_auto_handlers()

        logger.info(
            "PycramBridge ready — %d action(s): %s",
            len(self._registry),
            self._registry.action_types(),
        )

    # ── Pipeline factory ───────────────────────────────────────────────────────

    def make_pipeline(
        self,
        world_context_provider: Optional[Any] = None,
    ) -> ActionPipeline:
        """Return a fully-wired :class:`ActionPipeline` using introspection-driven slot-filling.

        :param world_context_provider: Optional ``Callable[[], str]`` that returns
            the world context string.  Defaults to SymbolGraph serialisation.
        """
        return ActionPipeline(
            groundable_type=self._groundable_type,
            context=self._context,
            world_context_provider=world_context_provider,
            action_types=self._registry.action_types_dict(),
            action_schemas=list(self._registry.schemas().values()),
        )

    # ── pycram action conversion ───────────────────────────────────────────────

    def to_pycram_action(self, spec: ActionSpec) -> Any:
        """Convert an :class:`ActionSpec` to a concrete pycram action instance.

        Merges ``spec.parameters`` and ``spec.grounded_entities`` and calls the
        discovered action class constructor.

        :param spec: Output of :class:`AutoActionHandler.execute`.
        :return: A pycram ``ActionDescription`` subclass instance.
        :raises KeyError: If the action type is not registered.
        :raises TypeError: If the merged kwargs don't match the constructor.
        """
        schema = self._registry.get(spec.action_type)
        if schema is None:
            raise KeyError(
                f"PycramBridge: no registered action for type '{spec.action_type}'. "
                f"Available: {self._registry.action_types()}"
            )

        # Merge resolved params + grounded entities into constructor kwargs
        kwargs: Dict[str, Any] = {}
        kwargs.update(spec.parameters)
        kwargs.update(spec.grounded_entities)

        # Drop any MISSING sentinels
        kwargs = {k: v for k, v in kwargs.items() if v is not dataclasses.MISSING}

        logger.debug(
            "PycramBridge.to_pycram_action: %s(%s)",
            spec.action_type,
            {k: type(v).__name__ for k, v in kwargs.items()},
        )
        return schema.action_cls(**kwargs)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _register_auto_handlers(self) -> None:
        """Register one AutoActionHandler per discovered action schema."""
        for action_type, schema in self._registry.schemas().items():
            # Each handler needs a closure over its schema
            def _make_handler_cls(s: ActionSchema) -> type:
                @dataclasses.dataclass
                class _Handler(AutoActionHandler):
                    pass
                _Handler.__name__ = f"Auto_{s.action_type}Handler"

                # Override __post_init__ to inject the schema
                original_post_init = _Handler.__post_init__

                def _post_init(self_h):
                    self_h.action_schema = s
                    original_post_init(self_h)

                _Handler.__post_init__ = _post_init
                return _Handler

            handler_cls = _make_handler_cls(schema)
            ActionDispatcher.register(action_type, handler_cls)
            self._handler_classes[action_type] = handler_cls
            logger.debug("Registered AutoActionHandler for '%s'.", action_type)

    # ── Convenience properties ─────────────────────────────────────────────────

    @property
    def registry(self) -> PycramActionRegistry:
        """The underlying action registry."""
        return self._registry

    @property
    def action_types(self) -> List[str]:
        """Registered action type strings."""
        return self._registry.action_types()

    @property
    def handlers(self) -> Dict[str, type]:
        """Auto-generated handler classes keyed by action type string.

        Each value is the dynamically created ``Auto_<ActionType>Handler``
        dataclass subclass of :class:`~krrood.llmr.pycram_bridge.auto_handler.AutoActionHandler`
        that was registered with :class:`~krrood.llmr.pipeline.dispatcher.ActionDispatcher`.
        """
        return dict(self._handler_classes)
