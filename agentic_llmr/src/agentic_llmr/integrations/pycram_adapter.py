"""PyCRAM adapter boundary for action discovery."""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import logging
import pkgutil
from typing import Dict

logger = logging.getLogger(__name__)

_ACTION_CACHE: Dict[str, type] | None = None

def discover_action_classes() -> Dict[str, type]:
    """Return all concrete PyCRAM action classes rooted at ActionDescription.
    
    Loads every module under ``pycram.robot_plans.actions`` once so that
    Python registers all subclasses, then uses krrood's recursive_subclasses
    to collect them. The result is cached after the first successful run.
    """
    global _ACTION_CACHE
    if _ACTION_CACHE is not None:
        return _ACTION_CACHE

    from krrood.utils import recursive_subclasses

    try:
        _pkg = importlib.import_module("pycram.robot_plans.actions")
    except ImportError:
        logger.warning("discover_action_classes: pycram.robot_plans.actions not found.")
        return {}

    for _, modname, _ in pkgutil.walk_packages(
        _pkg.__path__, prefix=_pkg.__name__ + "."
    ):
        try:
            importlib.import_module(modname)
        except Exception as exc:
            logger.debug("discover_action_classes: skipping %s: %s", modname, exc)

    from pycram.robot_plans.actions.base import ActionDescription

    _ACTION_CACHE = {
        cls.__name__: cls
        for cls in recursive_subclasses(ActionDescription)
        if dataclasses.is_dataclass(cls) and not inspect.isabstract(cls)
    }
    
    return _ACTION_CACHE
