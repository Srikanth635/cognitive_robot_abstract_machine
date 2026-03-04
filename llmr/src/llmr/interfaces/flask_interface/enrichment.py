"""
enrichment.py

Hybrid enrichment mapper for ontology-based action structures.

Responsibilities
----------------
- Take:
    1) ontology ActionSequence (ontology_refs.py)
    2) workflow-derived enriched_action_core_attributes (structured dicts)

- Produce:
    - ontology ActionSequence (unchanged)
    - enrichment_sidecar with action-scoped, entity-scoped attributes

Design principles
-----------------
- Programmatic mapping FIRST
- LLM fallback ONLY when confidence is low
- No mutation of ontology models
- Action-dependent attribute schemas supported
"""

from __future__ import annotations
import anthropic
from typing import Dict, Any, List, Optional
from copy import deepcopy
import json
from pydantic import ValidationError
from llmr.interfaces.flask_interface.ontology_refs import ActionSequence
from openai import OpenAI
from langchain_ollama import ChatOllama
gclient = OpenAI()
cclient = anthropic.Anthropic()

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def normalize_label(text: str) -> str:
    """
    Normalize labels for matching.
    Example: "Milk Bottle" -> "milk_bottle"
    """
    return (
        text.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def index_entities(action_sequence: ActionSequence) -> Dict[str, str]:
    """
    Build label -> entity_id index from ontology spine.
    """
    index: Dict[str, str] = {}

    for action in action_sequence.actions:
        for ra in action.has_participant:
            label = ra.participant.label
            if label:
                index[normalize_label(label)] = ra.participant.id

    return index


# -----------------------------------------------------------------------------
# Confidence evaluation
# -----------------------------------------------------------------------------

def entity_match_confidence(label: str, entity_index: Dict[str, str]) -> float:
    """
    Simple confidence score for entity matching.
    """
    norm = normalize_label(label)
    if norm in entity_index:
        return 0.99
    # weak heuristic fallback
    for k in entity_index:
        if norm in k or k in norm:
            return 0.6
    return 0.0


# -----------------------------------------------------------------------------
# LLM fallback stub (pluggable)
# -----------------------------------------------------------------------------

def resolve_entity_with_llm(
    label: str,
    entity_index: Dict[str, str],
    llm_resolver: Optional[Any] = None,
) -> Optional[str]:
    """
    Resolve entity label to entity_id using LLM (fallback).

    llm_resolver is expected to be a callable:
        (label: str, candidates: Dict[label, id]) -> entity_id | None
    """
    if llm_resolver is None:
        return None

    candidates = [
        {"label": k, "id": v}
        for k, v in entity_index.items()
    ]

    return llm_resolver(label, candidates)


# -----------------------------------------------------------------------------
# Core enrichment logic
# -----------------------------------------------------------------------------

def coerce_workflow_list(items: List[Any]) -> List[Dict[str, Any]]:
    """
    Accepts either:
      - List[dict]
      - List[str] where each string is a JSON object
    Returns List[dict].
    """
    out: List[Dict[str, Any]] = []
    for i, item in enumerate(items):
        if isinstance(item, dict):
            out.append(item)
        elif isinstance(item, str):
            s = item.strip()
            try:
                out.append(json.loads(s))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"enriched_action_core_attributes[{i}] is a string but not valid JSON."
                ) from e
        else:
            raise TypeError(
                f"enriched_action_core_attributes[{i}] must be dict or JSON string, got {type(item)}"
            )
    return out


def enrich_action_sequence(
    action_sequence: ActionSequence,
    enriched_action_core_attributes: List[Dict[str, Any]],
    llm_resolver: Optional[Any] = None,
    confidence_threshold: float = 0.7,
) -> Dict[str, Any]:
    """
    Main enrichment entry point.

    Returns a dict with:
      - ontology_spine (unchanged)
      - enrichment_sidecar
    """
    enriched_action_core_attributes = coerce_workflow_list(enriched_action_core_attributes)
    entity_index = index_entities(action_sequence)

    enrichment_sidecar: Dict[str, Any] = {
        "version": "1.0",
        "source": "workflow_enriched_action_core_attributes",
        "per_action": {},
        "mapping_report": {
            "entity_alignment": [],
            "llm_fallback_used": False,
        },
    }

    # Align workflow actions to ontology actions by index
    for idx, action in enumerate(action_sequence.actions):
        action_id = action.id
        workflow_attrs = enriched_action_core_attributes[idx]

        action_block = {
            "task": action.executes_task[0].ontology_class
            if action.executes_task else None,
            "action_attributes": {},
            "entity_attributes": {},
        }

        # --------------------------
        # Action-level attributes
        # --------------------------
        if "action_verb" in workflow_attrs:
            action_block["action_attributes"]["action_verb"] = workflow_attrs["action_verb"]

        # --------------------------
        # Entity-level attributes
        # --------------------------
        for key, value in workflow_attrs.items():
            if not key.endswith("_props"):
                continue

            ref_key = key[:-6]  # remove "_props"

            if ref_key not in workflow_attrs:
                continue

            referenced_label = workflow_attrs[ref_key]
            if not isinstance(referenced_label, str):
                continue

            referenced_label_norm = normalize_label(referenced_label)

            confidence = entity_match_confidence(referenced_label, entity_index)

            if confidence < confidence_threshold:
                resolved_id = resolve_entity_with_llm(
                    referenced_label, entity_index, llm_resolver
                )
                enrichment_sidecar["mapping_report"]["llm_fallback_used"] |= bool(resolved_id)
            else:
                resolved_id = entity_index.get(referenced_label_norm)

            if not resolved_id:
                continue

            action_block["entity_attributes"][resolved_id] = deepcopy(value)
            action_block["entity_attributes"][resolved_id]["_provenance"] = [
                {"from": "workflow", "field": key}
            ]

            enrichment_sidecar["mapping_report"]["entity_alignment"].append(
                {
                    "workflow_name": referenced_label,
                    "mapped_to_entity_id": resolved_id,
                    "confidence": round(confidence, 2),
                }
            )

        enrichment_sidecar["per_action"][action_id] = action_block

    return {
        "ontology_spine": action_sequence,
        "enrichment_sidecar": enrichment_sidecar,
    }


def extract_action_sequence(instruction: str, SYSTEM_PROMPT: str, SEQUENCE_SCHEMA: str, max_retries: int = 2) -> ActionSequence:
    input_payload = [
        {"role": "assistant", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps({"instruction": instruction, "target_schema": SEQUENCE_SCHEMA}, indent=2)},
    ]

    for attempt in range(max_retries + 1):

        try:
            response = gclient.beta.chat.completions.parse(
                model="gpt-4.1-mini",
                messages=input_payload,
                response_format=ActionSequence
            )
            print("Used GPT model")
            raw = response.choices[0].message.parsed

        except Exception as e:
            print("Exception Raised - ",e)
            print("Turning to Ollama QWen3")
            llm = ChatOllama(model="qwen3:8b", temperature=0.1).with_structured_output(ActionSequence, method="json_schema")
            response = llm.invoke(input_payload)
            raw = response

        try:
            return ActionSequence.model_validate(raw)

        except ValidationError as e:
            if attempt >= max_retries:
                raise

            input_payload.append({"role": "assistant", "content": json.dumps(raw, indent=2)})
            input_payload.append({"role": "user", "content": f"Validation failed. Fix the JSON.\nError:\n{e}"})


    raise RuntimeError("Unreachable")


# -----------------------------------------------------------------------------
# Example usage (no LLM)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    pass
