"""
Turning a hypothesis record into data and back.

ONE WIRE FORMAT, TWO USES
-------------------------
The shape here is the same flat fragment graph a model emits during generation:
a list of fragments, each with an id and a type, cross-referenced by id. Reusing
it means a stored hypothesis and a generated one have the same structure, so a
round-trip failure shows up as a decode failure in either direction rather than
in only one -- and there is one shape to reason about instead of two.

SELF-DESCRIBING VALUES
----------------------
Unlike generation, deserialisation here has no schema to decode against: a store
may hold hypotheses generated under module versions that are no longer loaded,
and asking it to derive a schema first would make reading old records depend on
the current model. So values carry their own kind: ``$ref`` for a fragment,
``$enum`` for a value-partition member, ``$concept`` for a foreign term.

RECORD-LOCAL IDENTITY
---------------------
``local_id`` is preserved on load. It is scoped by the record identifier, so
restoring it cannot create a global collision and lets stored citations continue
to address the same fragment after a round trip.
"""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from typing import Any, Mapping, Optional

from llm_reasoner.clients.base import Exchange
from llm_reasoner.knowledge.hypothesis.provenance import Provenance
from llm_reasoner.knowledge.hypothesis.record import (
    Assumption,
    HypothesisRecord,
    ValidationReport,
)
from llm_reasoner.knowledge.hypothesis.scope import KnowledgeScope
from llm_reasoner.knowledge.hypothesis.status import Defeater, HypothesisStatus
from llm_reasoner.exceptions import LLMReasonerError
from llm_reasoner.knowledge.ontological.decorators import (
    REGISTRY,
    ConceptReference,
    OntologyRegistry,
    property_meta,
)

FORMAT_VERSION = 1


class SerializationError(LLMReasonerError):
    """
    A record could not be written or read.
    """


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------


def _encode_value(value: Any, ids: Mapping[int, str]) -> Any:
    # Enum is tested BEFORE the primitives. A StrEnum member is also a str and
    # an IntEnum member is also an int, so the obvious ordering silently
    # degrades every value-partition member to a bare string on the way out --
    # and the record still round-trips, still loads, and only fails when
    # something compares it against the enum.
    if isinstance(value, Enum):
        return {"$enum": f"{type(value).__name__}.{value.name}"}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, ConceptReference):
        return {
            "$concept": value.iri,
            "label": value.label,
            "ancestors": sorted(value.ancestors),
        }
    identity = ids.get(id(value))
    if identity is not None:
        return {"$ref": identity}
    raise SerializationError(
        f"{type(value).__name__} is neither a primitive, an enum member, a "
        "concept reference, nor a fragment of this hypothesis"
    )


def _decode_value(value: Any, instances: Mapping[str, Any], registry) -> Any:
    if not isinstance(value, dict):
        return value
    if "$ref" in value:
        target = instances.get(value["$ref"])
        if target is None:
            raise SerializationError(f"dangling reference: {value['$ref']}")
        return target
    if "$enum" in value:
        type_name, _, member_name = value["$enum"].partition(".")
        meta = registry.by_name(type_name)
        if meta is None or not issubclass(meta.cls, Enum):
            raise SerializationError(f"unknown enumeration: {type_name}")
        try:
            return meta.cls[member_name]
        except KeyError as error:
            raise SerializationError(
                f"{type_name} has no member {member_name}"
            ) from error
    if "$concept" in value:
        return ConceptReference(
            iri=value["$concept"],
            label=value.get("label"),
            ancestors=frozenset(value.get("ancestors", ())),
        )
    raise SerializationError(f"unrecognised encoded value: {sorted(value)}")


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def encode(record: HypothesisRecord) -> dict[str, Any]:
    """
    Turn a record into plain data.
    """
    fragments = record.scope.fragments
    ids = {id(f): f.local_id for f in fragments}

    encoded: list[dict[str, Any]] = []
    for fragment in fragments:
        meta = REGISTRY.meta_for(type(fragment))
        entry: dict[str, Any] = {
            "id": fragment.local_id,
            "type": meta.local_name if meta else type(fragment).__name__,
        }
        if getattr(fragment, "label", ""):
            entry["label"] = fragment.label
        for name in property_meta(type(fragment)):
            value = getattr(fragment, name, None)
            if value is None or value == [] or value is False:
                continue
            if isinstance(value, (list, tuple, set, frozenset)):
                entry[name] = [_encode_value(v, ids) for v in value]
            else:
                entry[name] = _encode_value(value, ids)
        encoded.append(entry)

    provenance = record.provenance
    return {
        "format": FORMAT_VERSION,
        "record_id": record.record_id,
        "status": str(record.status),
        "action_type": {
            "iri": record.action_type.iri,
            "label": record.action_type.label,
            "ancestors": sorted(record.action_type.ancestors),
        },
        "root_id": record.root_id,
        "supersedes": record.supersedes,
        "superseded_by": record.superseded_by,
        "provenance": {
            "generator": provenance.generator,
            "prompt_version": provenance.prompt_version,
            "module_versions": dict(provenance.module_versions),
            "generated_at": provenance.generated_at.isoformat(),
            "degraded": provenance.degraded,
            "exchanges": [
                {
                    "system": e.system,
                    "user": e.user,
                    "raw_response": e.raw_response,
                    "client_identity": e.client_identity,
                    "attempt": e.attempt,
                }
                for e in provenance.exchanges
            ],
        },
        "validation": {
            "checks_run": record.validation.checks_run,
            "errors": list(record.validation.errors),
            "warnings": list(record.validation.warnings),
            "repair_attempts": record.validation.repair_attempts,
        },
        "assumptions": [
            {
                "fragment_id": a.fragment_id,
                "description": a.description,
                "on_roles": list(a.on_roles),
            }
            for a in record.assumptions
        ],
        "defeaters": [
            {
                "reason": d.reason,
                "source": d.source,
                "fragment_id": d.fragment_id,
                "recorded_at": d.recorded_at.isoformat(),
            }
            for d in record.defeaters
        ],
        "fragments": encoded,
    }


def decode(
    data: Mapping[str, Any], registry: OntologyRegistry = REGISTRY
) -> HypothesisRecord:
    """
    Rebuild a record from plain data.

    Two passes, the same as everywhere else fragments are built: instantiate by type,
    then resolve references, because fragments legitimately point at each other in both
    directions.
    """
    if data.get("format") != FORMAT_VERSION:
        raise SerializationError(f"unsupported format version: {data.get('format')!r}")

    entries = data.get("fragments", [])
    instances: dict[str, Any] = {}
    for entry in entries:
        meta = registry.by_name(entry["type"])
        if meta is None:
            raise SerializationError(
                f"{entry['type']} is not declared by the loaded modules. The "
                "record was written under a model this process does not have."
            )
        instance = meta.cls(local_id=entry["id"])
        if "label" in entry:
            instance.label = entry["label"]
        instances[entry["id"]] = instance

    for entry in entries:
        instance = instances[entry["id"]]
        for key, value in entry.items():
            if key in ("id", "type", "label"):
                continue
            if isinstance(value, list):
                decoded = [_decode_value(v, instances, registry) for v in value]
                current = getattr(instance, key, None)
                if isinstance(current, list):
                    current.extend(decoded)
                else:
                    setattr(instance, key, decoded)
            else:
                setattr(instance, key, _decode_value(value, instances, registry))

    scope = KnowledgeScope(label=data.get("record_id", ""))
    scope.admit_all(instances.values())

    provenance_data = data["provenance"]
    provenance = Provenance(
        generator=provenance_data["generator"],
        prompt_version=provenance_data["prompt_version"],
        module_versions=dict(provenance_data.get("module_versions", {})),
        generated_at=datetime.fromisoformat(provenance_data["generated_at"]),
        degraded=provenance_data.get("degraded", False),
        exchanges=tuple(Exchange(**e) for e in provenance_data.get("exchanges", ())),
    )
    validation_data = data.get("validation", {})
    action = data["action_type"]

    root_id = data.get("root_id")
    root = instances.get(root_id) if root_id else None

    record = HypothesisRecord(
        action_type=ConceptReference(
            iri=action["iri"],
            label=action.get("label"),
            ancestors=frozenset(action.get("ancestors", ())),
        ),
        scope=scope,
        provenance=provenance,
        root_id=getattr(root, "local_id", None),
        record_id=data.get("record_id", ""),
        status=HypothesisStatus(data.get("status", "candidate")),
        validation=ValidationReport(
            checks_run=validation_data.get("checks_run", 0),
            errors=tuple(validation_data.get("errors", ())),
            warnings=tuple(validation_data.get("warnings", ())),
            repair_attempts=validation_data.get("repair_attempts", 0),
        ),
        assumptions=tuple(
            Assumption(
                fragment_id=(
                    instances[a["fragment_id"]].local_id
                    if a["fragment_id"] in instances
                    else a["fragment_id"]
                ),
                description=a["description"],
                on_roles=tuple(
                    instances[r].local_id if r in instances else r
                    for r in a.get("on_roles", ())
                ),
            )
            for a in data.get("assumptions", ())
        ),
        defeaters=tuple(
            Defeater(
                reason=d["reason"],
                source=d["source"],
                fragment_id=d.get("fragment_id"),
                recorded_at=datetime.fromisoformat(d["recorded_at"]),
            )
            for d in data.get("defeaters", ())
        ),
        supersedes=data.get("supersedes"),
        superseded_by=data.get("superseded_by"),
    )
    return record


def dumps(record: HypothesisRecord, indent: Optional[int] = None) -> str:
    return json.dumps(encode(record), indent=indent)


def loads(text: str, registry: OntologyRegistry = REGISTRY) -> HypothesisRecord:
    return decode(json.loads(text), registry=registry)


__all__ = [
    "FORMAT_VERSION",
    "SerializationError",
    "decode",
    "dumps",
    "encode",
    "loads",
]
