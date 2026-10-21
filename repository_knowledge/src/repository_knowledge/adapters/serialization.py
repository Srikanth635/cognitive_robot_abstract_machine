"""
Round-trip source evidence using KRROOD's explicit serializer contracts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, is_dataclass, replace
from enum import Enum, StrEnum

from krrood.adapters.json_field import JSONField
from krrood.adapters.json_serializer import (
    DataclassJSONSerializer,
    EnumJSONSerializer,
    SubclassJSONSerializer,
)
from typing_extensions import TYPE_CHECKING

from repository_knowledge.exceptions import InvalidSerializedType
from repository_knowledge.clients import CanonicalStatus, SessionReport, SessionState
from repository_knowledge.model_trials import ModelTrial, QueryObservation, QueryOrigin
from repository_knowledge.publication import (
    AdmissionResult,
    ExtractionFingerprint,
    PublicationMode,
    SourceIdentity,
    ArtifactDownload,
    ArtifactReceipt,
    ConceptDigest,
    ConsumptionResult,
    ConsumptionState,
    ControlEvidence,
    IdentityKind,
    Publication,
    PublicationPolicy,
    RepositoryPermission,
    ReviewDecision,
    ReviewEvidence,
    ReviewState,
    TrustIssue,
    TrustReport,
    VerificationBinding,
)
from repository_knowledge.assistance import (
    ChangeKind,
    ConsumerCount,
    ChangeReport,
    ContextItem,
    ContextOmission,
    ContextRole,
    ContractChange,
    DiscoveryReport,
    FixtureRecord,
    OmissionReason,
    SourceCitation,
    TaskContext,
    TaskProfile,
    TestCatalog,
    TestRecord,
)
from repository_knowledge.concepts import (
    Concept,
    ConceptEvidence,
    ConceptProvenance,
    KnowledgeBundle,
    ConceptValidation,
    BundleValidation,
    ConceptKind,
    Lifecycle,
    EvidenceState,
    KnowledgeIssue,
    ConceptPage,
    ContentComparison,
)
from repository_knowledge.model import (
    AnalysisIdentity,
    Diagnostic,
    DiagnosticKind,
    FileExtraction,
    FreshnessReport,
    ImportBinding,
    IndexData,
    IndexManifest,
    IndexVersion,
    QueryPage,
    ReferenceRecord,
    RefreshReport,
    RelationshipKind,
    Resolution,
    SourceExcerpt,
    SourceKind,
    SourceLocation,
    SourceRecord,
    SourceAnalysis,
    SourceScope,
    SymbolKind,
    SymbolRecord,
    TraversalDirection,
)

if TYPE_CHECKING:
    from typing_extensions import Any, Self

# %% nested evidence boundary


@dataclass
class MappingEntry:
    """
    A portable dictionary entry carried by the shared dataclass serializer.
    """

    key: object
    """The dictionary key preserved as data."""
    value: object
    """
    The corresponding dictionary value.
    """


@dataclass
class EvidenceMapping:
    """
    Preserve nested dictionary values without runtime serializer discovery.
    """

    entries: list[MappingEntry]
    """The dictionary entries in their original order."""


class EnvelopeField(StrEnum):
    """
    The field carrying the original typed evidence in a boundary envelope.
    """

    EVIDENCE = "evidence"


@dataclass
class EvidenceEnvelope(SubclassJSONSerializer):
    """
    Carry nested source records through the explicit subclass serializer path.

    The enclosed record keeps the schema produced by :class:`DataclassJSONSerializer`.
    """

    evidence: object
    """
    The source dataclass or enum represented by this envelope.
    """

    def to_json(self, **kwargs: Any) -> dict[str, Any]:
        if isinstance(self.evidence, Enum):
            payload = EnumJSONSerializer.to_json(self.evidence)
        else:
            record = EvidenceFields(wrap_records=True).transform_fields(self.evidence)
            payload = DataclassJSONSerializer.to_json(record)
        return {
            JSONField.TYPE: f"{type(self).__module__}.{type(self).__name__}",
            EnvelopeField.EVIDENCE: payload,
        }

    @classmethod
    def _from_json(cls, data: dict[str, Any], **kwargs: Any) -> Self:
        payload = data[EnvelopeField.EVIDENCE]
        record_class = EvidenceType.resolve(payload[JSONField.TYPE])
        if issubclass(record_class, Enum):
            evidence = EnumJSONSerializer.from_json(payload, clazz=record_class)
        else:
            evidence = DataclassJSONSerializer.from_json(payload, clazz=record_class)
        return cls(EvidenceFields(wrap_records=False).transform(evidence))


class EvidenceType(Enum):
    """
    The fixed record vocabulary accepted by the source-index boundary.
    """

    ENVELOPE = EvidenceEnvelope
    MAPPING = EvidenceMapping
    MAPPING_ENTRY = MappingEntry
    DIAGNOSTIC = Diagnostic
    DIAGNOSTIC_KIND = DiagnosticKind
    EXTRACTION = FileExtraction
    FRESHNESS = FreshnessReport
    IMPORT_BINDING = ImportBinding
    INDEX = IndexData
    MANIFEST = IndexManifest
    FORMAT = IndexVersion
    QUERY = QueryPage
    REFERENCE = ReferenceRecord
    RELATIONSHIP = RelationshipKind
    RESOLUTION = Resolution
    EXCERPT = SourceExcerpt
    SOURCE_KIND = SourceKind
    LOCATION = SourceLocation
    SOURCE = SourceRecord
    SCOPE = SourceScope
    SYMBOL_KIND = SymbolKind
    SYMBOL = SymbolRecord
    DIRECTION = TraversalDirection
    ANALYSIS_IDENTITY = AnalysisIdentity
    ANALYSIS = SourceAnalysis
    REFRESH = RefreshReport
    CONCEPT = Concept
    CONCEPT_EVIDENCE = ConceptEvidence
    CONCEPT_PROVENANCE = ConceptProvenance
    BUNDLE = KnowledgeBundle
    CONCEPT_VALIDATION = ConceptValidation
    BUNDLE_VALIDATION = BundleValidation
    CONCEPT_KIND = ConceptKind
    LIFECYCLE = Lifecycle
    EVIDENCE_STATE = EvidenceState
    KNOWLEDGE_ISSUE = KnowledgeIssue
    CONCEPT_PAGE = ConceptPage
    CONTENT_COMPARISON = ContentComparison
    ARTIFACT_DOWNLOAD = ArtifactDownload
    ADMISSION_RESULT = AdmissionResult
    EXTRACTION_FINGERPRINT = ExtractionFingerprint
    ARTIFACT_RECEIPT = ArtifactReceipt
    CONCEPT_DIGEST = ConceptDigest
    CONSUMPTION_RESULT = ConsumptionResult
    CONSUMPTION_STATE = ConsumptionState
    CONTROL_EVIDENCE = ControlEvidence
    IDENTITY_KIND = IdentityKind
    PUBLICATION = Publication
    PUBLICATION_MODE = PublicationMode
    SOURCE_IDENTITY = SourceIdentity
    PUBLICATION_POLICY = PublicationPolicy
    REPOSITORY_PERMISSION = RepositoryPermission
    REVIEW_DECISION = ReviewDecision
    REVIEW_EVIDENCE = ReviewEvidence
    REVIEW_STATE = ReviewState
    TRUST_ISSUE = TrustIssue
    TRUST_REPORT = TrustReport
    VERIFICATION_BINDING = VerificationBinding
    CHANGE_KIND = ChangeKind
    CONSUMER_COUNT = ConsumerCount
    CHANGE_REPORT = ChangeReport
    CONTEXT_ITEM = ContextItem
    CONTEXT_OMISSION = ContextOmission
    CONTEXT_ROLE = ContextRole
    CONTRACT_CHANGE = ContractChange
    DISCOVERY_REPORT = DiscoveryReport
    FIXTURE_RECORD = FixtureRecord
    OMISSION_REASON = OmissionReason
    SOURCE_CITATION = SourceCitation
    TASK_CONTEXT = TaskContext
    TASK_PROFILE = TaskProfile
    TEST_CATALOG = TestCatalog
    TEST_RECORD = TestRecord
    SESSION_REPORT = SessionReport
    SESSION_STATE = SessionState
    CANONICAL_STATUS = CanonicalStatus
    MODEL_TRIAL = ModelTrial
    QUERY_OBSERVATION = QueryObservation
    QUERY_ORIGIN = QueryOrigin

    @classmethod
    def resolve(cls, serialized_type: str) -> type:
        """
        Resolve only explicitly imported evidence classes.
        """
        for member in cls:
            record_class = member.value
            if f"{record_class.__module__}.{record_class.__name__}" == serialized_type:
                return record_class
        raise InvalidSerializedType(serialized_type)


@dataclass
class EvidenceFields:
    """
    Preserve nested record values across the serialization boundary.
    """

    wrap_records: bool
    """
    Whether nested records become envelopes or envelopes become records.
    """

    def transform_fields(self, record: object) -> object:
        """
        Rebuild a dataclass with its nested values adapted at the boundary.
        """
        return replace(
            record,
            **{
                attribute_name: self.transform(attribute_value)
                for attribute_name, attribute_value in vars(record).items()
            },
        )

    def transform(self, value: object) -> object:
        """
        Adapt nested dataclasses and enums while preserving collection shape.
        """
        if self.wrap_records and (is_dataclass(value) or isinstance(value, Enum)):
            return EvidenceEnvelope(value)
        if isinstance(value, EvidenceEnvelope):
            return self.transform(value.evidence)
        if isinstance(value, EvidenceMapping):
            entries = [self.transform(entry) for entry in value.entries]
            return {entry.key: entry.value for entry in entries}
        if self.wrap_records and isinstance(value, dict):
            return EvidenceEnvelope(
                EvidenceMapping(
                    [MappingEntry(key, item) for key, item in value.items()]
                )
            )
        if is_dataclass(value):
            return self.transform_fields(value)
        if isinstance(value, list):
            return [self.transform(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.transform(item) for item in value)
        if isinstance(value, dict):
            return {
                self.transform(key): self.transform(item) for key, item in value.items()
            }
        return value


# %% owned evidence encoding


@dataclass
class RecordCodec:
    """
    Encode evidence with KRROOD and validate types before deserialization.
    """

    def encode(self, record: object) -> str:
        """
        Encode an owned dataclass using the shared field serializer.
        """
        EvidenceType.resolve(f"{type(record).__module__}.{type(record).__name__}")
        prepared = EvidenceFields(wrap_records=True).transform_fields(record)
        return json.dumps(
            DataclassJSONSerializer.to_json(prepared), sort_keys=True, ensure_ascii=True
        )

    def decode(self, payload: str, record_class: type) -> Any:
        """
        Read evidence only after validating all serialized identities.
        """
        data = json.loads(payload)
        expected = f"{record_class.__module__}.{record_class.__name__}"
        if not isinstance(data, dict) or data.get(JSONField.TYPE) != expected:
            raise InvalidSerializedType(
                str(data.get(JSONField.TYPE))
                if isinstance(data, dict)
                else type(data).__name__
            )
        pending: list[object] = [data]
        allowed_collections = {
            f"{collection.__module__}.{collection.__name__}"
            for collection in (list, tuple)
        }
        while pending:
            current = pending.pop()
            if isinstance(current, dict):
                serialized_type = current.get(JSONField.TYPE)
                if serialized_type is not None:
                    EvidenceType.resolve(serialized_type)
                collection_type = current.get(JSONField.COLLECTION_TYPE)
                if (
                    collection_type is not None
                    and collection_type not in allowed_collections
                ):
                    raise InvalidSerializedType(str(collection_type))
                if current.get(JSONField.IS_CLASS):
                    raise InvalidSerializedType(str(serialized_type))
                pending.extend(current.values())
            elif isinstance(current, list):
                pending.extend(current)
        decoded = DataclassJSONSerializer.from_json(data, clazz=record_class)
        return EvidenceFields(wrap_records=False).transform(decoded)
