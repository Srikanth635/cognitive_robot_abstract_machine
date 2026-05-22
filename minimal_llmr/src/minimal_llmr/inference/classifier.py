"""Task 1: Inference — classify a NL instruction into a known action class."""

from __future__ import annotations

import dataclasses
import inspect
import logging
from typing import TYPE_CHECKING, Optional

from krrood.class_diagrams.class_diagram import ClassDiagram
from krrood.utils import own_dataclass_fields, recursive_subclasses



from minimal_llmr.core.schemas import ActionClassification

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

_CLASSIFIER_SYSTEM = """\
You are a robot action classifier.

Given a natural-language instruction, identify which robot action class it
corresponds to from the list of available action classes below.

Available action classes and schema summaries:
{action_classes}

Return the EXACT Python class name (e.g. "PickUpAction" not "pick up action").
Return structured JSON.
"""


def build_action_registry(action_base_class: type) -> dict[str, type]:
    """Discover all concrete action dataclass subclasses of *action_base_class*.

    Uses KRROOD's recursive_subclasses() — no PyCRAM module path needed.
    Only classes directly decorated with @dataclass are included (not bare subclasses
    that merely inherit __dataclass_fields__ from a parent dataclass).
    """
    result: dict[str, type] = {}
    for cls in recursive_subclasses(action_base_class):
        if "__dataclass_fields__" not in cls.__dict__:
            continue
        if getattr(cls, "__abstractmethods__", frozenset()):
            continue
        result[cls.__name__] = cls
    return result


def infer_action_class(
    instruction: str,
    llm: "BaseChatModel",
    action_registry: dict[str, type],
) -> Optional[ActionClassification]:
    """Classify *instruction* to an action class via one structured LLM call.

    :param instruction:     Natural-language instruction.
    :param llm:             LangChain-compatible chat model.
    :param action_registry: {class_name: class} map built by build_action_registry().
    :returns: ActionClassification on success; None on LLM failure.
    """
    if not action_registry:
        logger.warning("infer_action_class: action_registry is empty")
        return None

    system = _CLASSIFIER_SYSTEM.format(action_classes=build_action_catalog(action_registry))
    structured_llm = llm.with_structured_output(ActionClassification)
    try:
        return structured_llm.invoke([
            {"role": "system", "content": system},
            {"role": "user", "content": instruction},
        ])
    except Exception:
        logger.exception("infer_action_class: LLM call failed")
        return None


def build_action_catalog(action_registry: dict[str, type]) -> str:
    """Build the classifier prompt catalog using WrappedField directly."""
    lines: list[str] = []

    for name in sorted(action_registry):
        action_cls = action_registry[name]
        doc = " ".join((inspect.getdoc(action_cls) or "").split())
        if len(doc) > 180:
            doc = f"{doc[:177]}..."

        header = f"  - {name}"
        if doc:
            header += f": {doc}"
        lines.append(header)

        try:
            own_names = {f.name for f in own_dataclass_fields(action_cls)}
            cd = ClassDiagram([action_cls])
            wc = cd.get_wrapped_class(action_cls)
            summaries = []
            for wf in (w for w in wc.fields if w.name in own_names):
                optional_label = "optional" if wf.is_optional else "required"
                type_label = getattr(wf.type_endpoint, "__name__", str(wf.type_endpoint))
                summaries.append(f"{wf.name}:{type_label}({optional_label})")
            if summaries:
                lines.append(f"    fields: {', '.join(summaries)}")
        except Exception:
            pass

    return "\n".join(lines)
