"""Schema documentation — builds LLM-facing parameter descriptions from PyCRAM action classes."""

import ast
import functools
import inspect
import textwrap
from typing import Any, Dict, List
from krrood.class_diagrams.class_diagram import ClassDiagram
from krrood.utils import own_dataclass_fields
from dataclasses import MISSING, is_dataclass
from enum import Enum


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
        if "Manipulator" in {cls.__name__ for cls in getattr(wf.type_endpoint, "__mro__", [])}:
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
