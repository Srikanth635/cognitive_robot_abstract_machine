"""PycramActionRegistry — discovers and registers pycram action classes.

Scans ``pycram.robot_plans.actions.core`` for dataclass subclasses of
``ActionDescription`` and introspects each one into an :class:`ActionSchema`.

Usage::

    registry = PycramActionRegistry()
    registry.discover()

    schema = registry.get("PickUpAction")
    action_types_dict = registry.action_types_dict()  # for slot-filler prompt
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import logging
import pkgutil
from typing_extensions import Dict, List, Optional

from krrood.llmr.pycram_bridge.introspector import ActionSchema, PycramIntrospector, introspect

logger = logging.getLogger(__name__)


class PycramActionRegistry:
    """Registry mapping action type strings to :class:`ActionSchema` objects.

    :param introspector: Optional custom introspector; uses the module singleton by default.
    """

    def __init__(self, introspector: Optional[PycramIntrospector] = None) -> None:
        self._introspector = introspector or PycramIntrospector()
        self._schemas: Dict[str, ActionSchema] = {}

    # ── Discovery ──────────────────────────────────────────────────────────────

    def discover(self, package: str = "pycram.robot_plans.actions.core") -> "PycramActionRegistry":
        """Scan *package* for pycram action classes and register them all.

        :param package: Dotted module path to scan (defaults to pycram core actions).
        :return: ``self`` for chaining.
        """
        try:
            from pycram.robot_plans.actions.base import ActionDescription
            pkg = importlib.import_module(package)
        except ImportError as exc:
            logger.warning("Cannot discover pycram actions — pycram not installed: %s", exc)
            return self

        prefix = package + "."
        for _, modname, _ in pkgutil.iter_modules(pkg.__path__, prefix):
            try:
                mod = importlib.import_module(modname)
            except Exception as exc:
                logger.debug("Skipping module %s: %s", modname, exc)
                continue

            for name, obj in inspect.getmembers(mod, inspect.isclass):
                if (
                    dataclasses.is_dataclass(obj)
                    and issubclass(obj, ActionDescription)
                    and obj is not ActionDescription
                    and obj.__module__ == mod.__name__  # defined here, not imported
                ):
                    self._register_cls(obj)

        logger.info(
            "PycramActionRegistry: discovered %d action(s): %s",
            len(self._schemas),
            list(self._schemas.keys()),
        )
        return self

    # ── Manual registration ────────────────────────────────────────────────────

    def register(self, action_cls: type) -> "PycramActionRegistry":
        """Manually register a single action class.

        :return: ``self`` for chaining.
        """
        self._register_cls(action_cls)
        return self

    def _register_cls(self, action_cls: type) -> None:
        name = action_cls.__name__
        try:
            schema = self._introspector.introspect(action_cls)
            self._schemas[name] = schema
            logger.debug("Registered '%s' (%d fields)", name, len(schema.fields))
        except Exception as exc:
            logger.warning("Failed to introspect '%s': %s", name, exc)

    # ── Lookups ────────────────────────────────────────────────────────────────

    def get(self, action_type: str) -> Optional[ActionSchema]:
        """Return :class:`ActionSchema` for *action_type*, or ``None``."""
        return self._schemas.get(action_type)

    def action_types(self) -> List[str]:
        """Return all registered action type names."""
        return list(self._schemas.keys())

    def action_types_dict(self) -> Dict[str, str]:
        """Return ``{action_type: docstring}`` for use in the slot-filler prompt."""
        return {name: schema.docstring for name, schema in self._schemas.items()}

    def schemas(self) -> Dict[str, ActionSchema]:
        """Return all registered schemas."""
        return dict(self._schemas)

    def __contains__(self, action_type: str) -> bool:
        return action_type in self._schemas

    def __len__(self) -> int:
        return len(self._schemas)
