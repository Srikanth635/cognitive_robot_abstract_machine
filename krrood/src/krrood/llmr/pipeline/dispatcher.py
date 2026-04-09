"""Generic action dispatcher — registry pattern, zero robot/framework dependencies.

The dispatcher maps action type strings to :class:`ActionHandler` registered subclasses.
It produces :class:`ActionSpec` objects — generic
dicts carrying action type + resolved parameters — instead of concrete pycram
or robot actions.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing_extensions import Any, Dict, Optional, Type

from krrood.llmr.pipeline.entity_grounder import EntityGrounder
from krrood.llmr.workflows.schemas.common import ActionSlotSchema

logger = logging.getLogger(__name__)


# ── Generic action output ──────────────────────────────────────────────────────


@dataclass
class ActionSpec:
    """Generic, framework-agnostic output of an action handler.

    The caller's integration layer converts this into the concrete action type
    (pycram ``PickUpAction``, ROS action, simulation command, etc.).

    :param action_type: String identifier matching the registered handler key.
    :param parameters: Fully resolved parameters as a plain dict.
    :param grounded_entities: Map of role name → grounded Symbol instance(s).
        E.g. ``{"object": body_instance, "target": surface_instance}``.
    """

    action_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    grounded_entities: Dict[str, Any] = field(default_factory=dict)


# ── ActionHandler ABC ──────────────────────────────────────────────────────────


@dataclass
class ActionHandler(ABC):
    """Base class for action-specific execution handlers.

    Subclasses implement :meth:`execute` to perform entity grounding,
    optional LLM-driven parameter resolution, and return an :class:`ActionSpec`.

    :param grounder: :class:`EntityGrounder` configured with the caller's
        groundable type.  Injected by :class:`ActionDispatcher`.
    :param context: Arbitrary caller-supplied context dict (manipulator,
        execution context, arm info, etc.).  ``ActionHandler`` treats this
        as opaque — subclasses extract what they need.
    """

    grounder: EntityGrounder
    context: Dict[str, Any] = field(default_factory=dict)

    @abstractmethod
    def execute(self, schema: ActionSlotSchema) -> ActionSpec:
        """Ground entities, resolve parameters, return an :class:`ActionSpec`."""


# ── ActionDispatcher ───────────────────────────────────────────────────────────


class ActionDispatcher:
    """Routes a typed slot schema to the correct :class:`ActionHandler`.

    Maintains a class-level registry of ``action_type_string → handler class``.
    Instantiated with a shared :class:`EntityGrounder` and optional context dict
    that are passed to every handler.

    Usage::

        class MyPickUpHandler(ActionHandler):
            def execute(self, schema):
                obj = self.grounder.ground(schema.object_description)
                ...
                return ActionSpec("pick_up", parameters={...}, grounded_entities={...})

        ActionDispatcher.register("PickUpAction", MyPickUpHandler)

        dispatcher = ActionDispatcher(grounder=EntityGrounder(Body), context={...})
        spec = dispatcher.dispatch(schema)
    """

    _registry: Dict[str, Type[ActionHandler]] = {}

    @classmethod
    def register(cls, action_type: str, handler_class: Type[ActionHandler]) -> None:
        """Register *handler_class* for *action_type* string."""
        cls._registry[action_type] = handler_class
        logger.debug("ActionDispatcher: registered handler for '%s'.", action_type)

    def __init__(
        self,
        grounder: EntityGrounder,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._grounder = grounder
        self._context = context or {}
        self._handlers: Dict[str, ActionHandler] = {
            action_type: handler_cls(self._grounder, self._context)
            for action_type, handler_cls in self._registry.items()
        }

    def dispatch(self, schema: ActionSlotSchema) -> ActionSpec:
        """Execute the handler for *schema.action_type* and return an :class:`ActionSpec`."""
        action_type = schema.action_type
        handler = self._handlers.get(action_type)
        if handler is None:
            raise KeyError(
                f"No handler registered for action type '{action_type}'. "
                f"Registered types: {list(self._handlers)}. "
                "Implement an ActionHandler subclass and call ActionDispatcher.register()."
            )
        logger.info("Dispatching to '%s' handler.", action_type)
        return handler.execute(schema)
