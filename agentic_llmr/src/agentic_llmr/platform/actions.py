"""Actions — PyCRAM action class discovery and LLM-facing schema documentation.

Merged from: integrations/pycram_adapter.py + resolution/schema_docs.py

Responsibilities:
  - Discover all concrete PyCRAM action classes at runtime (once, then cached)
  - Build human-readable parameter documentation strings for those classes
    so the LLM agent knows what fields each action requires
"""

from __future__ import annotations

import ast
import dataclasses
import functools
import importlib
import inspect
import logging
import pkgutil
import textwrap
from dataclasses import MISSING, is_dataclass
from enum import Enum
from typing import Any, Dict, List

from krrood.class_diagrams.class_diagram import ClassDiagram
from krrood.utils import own_dataclass_fields

logger = logging.getLogger(__name__)


# ── PyCRAM action class discovery ─────────────────────────────────────────────

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


# ── Schema documentation builder ──────────────────────────────────────────────


@functools.lru_cache(maxsize=None)
def _extract_field_docstrings(cls: type) -> Dict[str, str]:
    """Extract attribute-level docstrings via AST (string literal after annotation)."""
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(cls)))
    except Exception:
        return {}

    docs: Dict[str, str] = {}
    class_body: list[ast.stmt] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_body = node.body
            break

    for i, node in enumerate(class_body):
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        if i + 1 < len(class_body):
            nxt = class_body[i + 1]
            if (
                isinstance(nxt, ast.Expr)
                and isinstance(nxt.value, ast.Constant)
                and isinstance(nxt.value.value, str)
            ):
                docs[node.target.id] = nxt.value.value.strip()
    return docs


@functools.lru_cache(maxsize=None)
def _build_recursive_fields(cls: type, indent: int = 2) -> List[str]:
    """Recursively documents the fields of a dataclass for the LLM schema."""
    lines = []
    prefix = " " * indent

    own_names = {f.name for f in own_dataclass_fields(cls)}
    cd = ClassDiagram([cls])
    wc = cd.get_wrapped_class(cls)
    docstrings = _extract_field_docstrings(cls)
    wf_by_name = {wf.name: wf for wf in wc.fields if wf.name in own_names}

    for name, wf in wf_by_name.items():
        if wf.field.default is not MISSING or wf.field.default_factory is not MISSING:
            continue

        # Auto-injected by hydrator — skip any Manipulator subclass (e.g. ParallelGripper)
        if "Manipulator" in {c.__name__ for c in getattr(wf.type_endpoint, "__mro__", [])}:
            continue

        doc = docstrings.get(name, "")
        doc_str = f" — {doc}" if doc else ""

        if wf.is_enum:
            members = list(wf.type_endpoint.__members__)
            type_label = f"Enum: {' | '.join(members)}"
            lines.append(f"{prefix}- {name} ({type_label}){doc_str}")
        elif is_dataclass(wf.type_endpoint):
            type_label = f"Dict[{wf.type_endpoint.__name__}]"
            lines.append(f"{prefix}- {name} ({type_label}){doc_str}. Required keys:")
            lines.extend(_build_recursive_fields(wf.type_endpoint, indent + 4))
        else:
            type_label = getattr(wf.type_endpoint, "__name__", str(wf.type_endpoint))
            lines.append(f"{prefix}- {name} ({type_label}){doc_str}")

    return lines


@functools.lru_cache(maxsize=None)
def build_action_documentation(action_cls: type) -> str:
    """Build a rich string describing the required parameters and docstrings for a PyCRAM action class."""
    lines: list[str] = []
    lines.append(f"Action Class: {action_cls.__name__}")

    action_doc = " ".join((inspect.getdoc(action_cls) or "").split())
    if action_doc:
        lines.append(f"Description: {action_doc}")

    lines.append("\nRequired Free Parameters:")

    try:
        field_lines = _build_recursive_fields(action_cls, indent=2)
        if not field_lines:
            lines.append("  (None found or all optional)")
        else:
            lines.extend(field_lines)
    except Exception as exc:
        lines.append(f"  (Failed to introspect parameters: {exc})")

    return "\n".join(lines)
