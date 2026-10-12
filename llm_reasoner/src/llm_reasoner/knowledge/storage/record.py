"""
The record of an admitted hypothesis: what was asked, of which model, under which
versions of the ontology, how many answers it took, and what was found.

The hypothesis itself is stored as Turtle beside its record, since its fragments refer to
one another in cycles that JSON cannot express; the record round-trips through krrood's
JSON serializer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from uuid import uuid4

from krrood.adapters.json_serializer import SubclassJSONSerializer
from typing_extensions import Any, Dict, Self

from llm_reasoner.clients.base import Exchange
from llm_reasoner.knowledge.ontologies import PredictiveManipulationOntologyFile

VERSION_INFO = re.compile(r'owl:versionInfo\s+"([^"]+)"')
"""
Locates the version a module file states for its ontology.
"""


def current_module_versions() -> dict[str, str]:
    """
    :return: The version of every ontology module, by module file name.
    """
    return {
        module_file.value: VERSION_INFO.search(module_file.path.read_text()).group(1)
        for module_file in PredictiveManipulationOntologyFile
    }


def qualified_name(python_type: type) -> str:
    """
    :return: The module and name of a class, which identify it across processes.
    """
    return f"{python_type.__module__}.{python_type.__qualname__}"


class ConsistencyVerdict(StrEnum):
    """
    What the open-world check said about an admitted hypothesis.
    """

    CONSISTENT = "consistent"
    """
    A reasoner found the hypothesis consistent with the ontology.
    """

    NOT_CHECKED = "not_checked"
    """
    No reasoner was asked.
    """


class RecordField(StrEnum):
    """
    The keys of a record's JSON form.
    """

    RECORD_ID = "record_id"
    """
    The record identifier.
    """

    CREATED_AT = "created_at"
    """
    When the hypothesis was admitted.
    """

    INSTRUCTION = "instruction"
    """
    The action put to the model.
    """

    ROOT_TYPE = "root_type"
    """
    The type that was generated.
    """

    FIXED_VALUES = "fixed_values"
    """
    The values the query fixed.
    """

    CLIENT_IDENTITY = "client_identity"
    """
    The model that answered.
    """

    EXCHANGES = "exchanges"
    """
    Every exchange of the generation.
    """

    WARNINGS = "warnings"
    """
    The warnings the hypothesis was admitted with.
    """

    CONSISTENCY = "consistency"
    """
    What the open-world check said.
    """

    MODULE_VERSIONS = "module_versions"
    """
    The ontology module versions.
    """


@dataclass
class HypothesisRecord(SubclassJSONSerializer):
    """
    The provenance of one admitted hypothesis.
    """

    instruction: str
    """
    The action the hypothesis describes, as it was put to the model.
    """

    root_type: str
    """
    The qualified name of the type that was generated.
    """

    fixed_values: dict[str, str]
    """
    The values the query fixed, by field name, as they were stated to the model.
    """

    client_identity: str
    """
    The model that generated the hypothesis.
    """

    exchanges: list[Exchange]
    """
    Every exchange of the generation, repairs included.
    """

    warnings: list[str]
    """
    The rule violations of warning severity the hypothesis was admitted with.
    """

    consistency: ConsistencyVerdict
    """
    What the open-world check said.
    """

    module_versions: dict[str, str]
    """
    The version of every ontology module the hypothesis was generated under.
    """

    record_id: str = field(default_factory=lambda: uuid4().hex)
    """
    The identifier of the record, and of the hypothesis it describes.
    """

    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    """
    When the hypothesis was admitted, as an ISO 8601 timestamp.
    """

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            RecordField.RECORD_ID: self.record_id,
            RecordField.CREATED_AT: self.created_at,
            RecordField.INSTRUCTION: self.instruction,
            RecordField.ROOT_TYPE: self.root_type,
            RecordField.FIXED_VALUES: self.fixed_values,
            RecordField.CLIENT_IDENTITY: self.client_identity,
            RecordField.EXCHANGES: [vars(exchange) for exchange in self.exchanges],
            RecordField.WARNINGS: self.warnings,
            RecordField.CONSISTENCY: self.consistency.value,
            RecordField.MODULE_VERSIONS: self.module_versions,
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            record_id=data[RecordField.RECORD_ID],
            created_at=data[RecordField.CREATED_AT],
            instruction=data[RecordField.INSTRUCTION],
            root_type=data[RecordField.ROOT_TYPE],
            fixed_values=data[RecordField.FIXED_VALUES],
            client_identity=data[RecordField.CLIENT_IDENTITY],
            exchanges=[
                Exchange(**exchange) for exchange in data[RecordField.EXCHANGES]
            ],
            warnings=data[RecordField.WARNINGS],
            consistency=ConsistencyVerdict(data[RecordField.CONSISTENCY]),
            module_versions=data[RecordField.MODULE_VERSIONS],
        )

    def answers(self, root_type: str, fixed_values: dict[str, str]) -> bool:
        """
        :return: Whether the record answers a request for the root type with the fixed
            values under the current versions of the ontology; the instruction it was
            generated for is provenance and plays no part.
        """
        return (
            self.root_type == root_type
            and self.fixed_values == fixed_values
            and self.module_versions == current_module_versions()
        )
