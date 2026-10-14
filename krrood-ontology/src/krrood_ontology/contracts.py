"""Versioned ontology-service records independent of agent, reasoner and storage APIs."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

CONTRACT_VERSION = "1.0"


class ContractError(ValueError):
    """A malformed service record, with a stable code and record location."""

    def __init__(self, code: str, message: str, location: str = ""):
        self.code, self.location = code, location
        super().__init__(f"{code}: {message}" + (f" [{location}]" if location else ""))


def _text(value, location):
    if not isinstance(value, str) or not value.strip():
        raise ContractError("text_required", "Expected nonempty text", location)


def _iri(value, location):
    _text(value, location)
    if not re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", value) or any(
        c.isspace() or c in '<>"{}|^`\\' or ord(c) < 32 for c in value
    ):
        raise ContractError(
            "absolute_iri",
            "Expected an absolute IRI without forbidden characters",
            location,
        )


def _id(value, location):
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,255}", value
    ):
        raise ContractError(
            "identifier", "Expected a stable local identifier", location
        )


def _sha(value, location):
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
        raise ContractError(
            "sha256", "Expected a lowercase SHA-256 fingerprint", location
        )


def _tuple(value, kind, location):
    if not isinstance(value, tuple) or any(not isinstance(v, kind) for v in value):
        raise ContractError(
            "immutable_tuple", "Expected an immutable tuple of typed records", location
        )


def _version(value):
    if value != CONTRACT_VERSION:
        raise ContractError(
            "contract_version", "Unsupported contract version", "/contract_version"
        )


@dataclass(frozen=True)
class CandidateContext:
    schema_id: str
    schema_sha256: str
    base_revision: str

    def __post_init__(self):
        _iri(self.schema_id, "/schema_id")
        _sha(self.schema_sha256, "/schema_sha256")
        _text(self.base_revision, "/base_revision")

    @classmethod
    def from_manifest(
        cls, manifest: Path | str, *, schema_id: str, base_revision: str
    ) -> CandidateContext:
        """Bind the context to the compiler's verified RDF closure bytes."""
        manifest = Path(manifest).resolve()
        data = json.loads(manifest.read_text(encoding="utf-8"))
        try:
            closure = (manifest.parent / data["runtime_ontology"]).resolve()
            expected = data["closure_sha256"]
        except (KeyError, TypeError) as error:
            raise ContractError("manifest", "Missing closure provenance") from error
        if not closure.is_relative_to(manifest.parent) or not closure.is_file():
            raise ContractError(
                "manifest_path", "Closure must be a file inside the generated output"
            )
        actual = hashlib.sha256(closure.read_bytes()).hexdigest()
        if actual != expected:
            raise ContractError(
                "schema_changed", "Closure fingerprint does not match the manifest"
            )
        return cls(schema_id, actual, base_revision)


@dataclass(frozen=True)
class EntityReference:
    local_id: str | None = None
    iri: str | None = None

    def __post_init__(self):
        if (self.local_id is None) == (self.iri is None):
            raise ContractError(
                "entity_reference", "Choose exactly one local ID or existing IRI"
            )
        if self.local_id is not None:
            _id(self.local_id, "/local_id")
        if self.iri is not None:
            _iri(self.iri, "/iri")


@dataclass(frozen=True)
class LiteralValue:
    lexical: str
    datatype_iri: str | None = None
    language: str | None = None

    def __post_init__(self):
        if not isinstance(self.lexical, str):
            raise ContractError("literal_lexical", "Literal lexical form must be text")
        if self.datatype_iri is not None and self.language is not None:
            raise ContractError(
                "literal_metadata", "Choose datatype or language metadata"
            )
        if self.datatype_iri is not None:
            _iri(self.datatype_iri, "/datatype_iri")
        if self.language is not None:
            if not isinstance(self.language, str) or not re.fullmatch(
                r"[A-Za-z]+(?:-[A-Za-z0-9]+)*", self.language
            ):
                raise ContractError("literal_language", "Malformed language tag")


@dataclass(frozen=True)
class CreateEntity:
    request_id: str
    local_id: str
    class_iris: tuple[str, ...]
    iri: str | None = None
    kind: str = field(default="create_entity", init=False)

    def __post_init__(self):
        _id(self.request_id, "/request_id")
        _id(self.local_id, "/local_id")
        _tuple(self.class_iris, str, "/class_iris")
        if not self.class_iris or len(set(self.class_iris)) != len(self.class_iris):
            raise ContractError(
                "class_iris", "At least one unique class IRI is required"
            )
        for iri in self.class_iris:
            _iri(iri, "/class_iris")
        if self.iri is not None:
            _iri(self.iri, "/iri")


@dataclass(frozen=True)
class ObjectAssertion:
    request_id: str
    subject: EntityReference
    property_iri: str
    target: EntityReference
    kind: str = field(default="object_assertion", init=False)

    def __post_init__(self):
        _id(self.request_id, "/request_id")
        _iri(self.property_iri, "/property_iri")
        if not isinstance(self.subject, EntityReference) or not isinstance(
            self.target, EntityReference
        ):
            raise ContractError(
                "entity_reference", "Object assertions require typed references"
            )


@dataclass(frozen=True)
class LiteralAssertion:
    request_id: str
    subject: EntityReference
    property_iri: str
    value: LiteralValue
    kind: str = field(default="literal_assertion", init=False)

    def __post_init__(self):
        _id(self.request_id, "/request_id")
        _iri(self.property_iri, "/property_iri")
        if not isinstance(self.subject, EntityReference) or not isinstance(
            self.value, LiteralValue
        ):
            raise ContractError(
                "literal_assertion",
                "Literal assertions require typed references and values",
            )


Operation = CreateEntity | ObjectAssertion | LiteralAssertion


@dataclass(frozen=True)
class Proposal:
    operations: tuple[Operation, ...]
    contract_version: str = CONTRACT_VERSION

    def __post_init__(self):
        _version(self.contract_version)
        _tuple(
            self.operations,
            (CreateEntity, ObjectAssertion, LiteralAssertion),
            "/operations",
        )
        identifiers = [v.request_id for v in self.operations]
        local_ids = [v.local_id for v in self.operations if isinstance(v, CreateEntity)]
        if len(set(identifiers)) != len(identifiers):
            raise ContractError("duplicate_request", "Request IDs must be unique")
        if len(set(local_ids)) != len(local_ids):
            raise ContractError("duplicate_entity", "Created local IDs must be unique")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CandidateToken:
    session_id: str
    revision: int
    digest: str

    def __post_init__(self):
        _id(self.session_id, "/session_id")
        _sha(self.digest, "/digest")
        if type(self.revision) is not int or self.revision < 1:
            raise ContractError("revision", "Revision must be a positive integer")


@dataclass(frozen=True)
class CandidateSnapshot:
    token: CandidateToken
    context: CandidateContext
    proposal: Proposal

    @classmethod
    def build(
        cls,
        session_id: str,
        revision: int,
        context: CandidateContext,
        proposal: Proposal,
    ) -> CandidateSnapshot:
        """Fingerprint immutable proposal content together with its authority context."""
        if not isinstance(context, CandidateContext) or not isinstance(
            proposal, Proposal
        ):
            raise ContractError("snapshot", "Expected typed context and proposal")
        content = json.dumps(
            {"context": asdict(context), "proposal": proposal.to_dict()},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        token = CandidateToken(
            session_id, revision, hashlib.sha256(content.encode("utf-8")).hexdigest()
        )
        return cls(token, context, proposal)


class CheckStage(str, Enum):
    PREFLIGHT = "preflight"
    CONSTRUCTION = "construction"
    LOCAL_VALIDATION = "local_validation"
    REASONER = "reasoner"


class CheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"
    ERROR = "error"


class FeedbackCategory(str, Enum):
    REQUEST = "request"
    LOCAL_VALIDATION = "local_validation"
    REASONER = "reasoner"
    UNSUPPORTED = "unsupported"
    EXECUTION = "execution"
    LIFECYCLE = "lifecycle"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class Feedback:
    category: FeedbackCategory
    severity: Severity
    code: str
    message: str
    location: str = ""
    entity: EntityReference | None = None
    property_iri: str | None = None
    axiom_id: str | None = None

    def __post_init__(self):
        if not isinstance(self.category, FeedbackCategory) or not isinstance(
            self.severity, Severity
        ):
            raise ContractError(
                "feedback", "Use explicit feedback category and severity"
            )
        _id(self.code, "/code")
        _text(self.message, "/message")
        if not isinstance(self.location, str) or (
            self.location and not self.location.startswith("/")
        ):
            raise ContractError("location", "Location must be empty or a JSON pointer")
        if self.entity is not None and not isinstance(self.entity, EntityReference):
            raise ContractError("entity_reference", "Expected typed reference")
        if self.property_iri is not None:
            _iri(self.property_iri, "/property_iri")
        if self.axiom_id is not None:
            _text(self.axiom_id, "/axiom_id")


@dataclass(frozen=True)
class StageResult:
    stage: CheckStage
    status: CheckStatus
    profile: str
    feedback: tuple[Feedback, ...] = ()

    def __post_init__(self):
        if not isinstance(self.stage, CheckStage) or not isinstance(
            self.status, CheckStatus
        ):
            raise ContractError("check_result", "Use explicit stage and status")
        _text(self.profile, "/profile")
        _tuple(self.feedback, Feedback, "/feedback")
        if self.status is CheckStatus.PASS and any(
            f.severity is Severity.ERROR for f in self.feedback
        ):
            raise ContractError(
                "contradictory_result", "A passing stage cannot contain error feedback"
            )


@dataclass(frozen=True)
class CheckReport:
    token: CandidateToken
    results: tuple[StageResult, ...]
    feedback: tuple[Feedback, ...] = ()
    contract_version: str = CONTRACT_VERSION

    def __post_init__(self):
        _version(self.contract_version)
        _tuple(self.results, StageResult, "/results")
        _tuple(self.feedback, Feedback, "/feedback")
        if not isinstance(self.token, CandidateToken):
            raise ContractError("review_token", "Expected typed candidate token")
        if len({v.stage for v in self.results}) != len(self.results):
            raise ContractError("duplicate_stage", "Report stages must be unique")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class StageRequirement:
    stage: CheckStage
    profile: str

    def __post_init__(self):
        if not isinstance(self.stage, CheckStage):
            raise ContractError("check_stage", "Expected explicit stage")
        _text(self.profile, "/profile")


@dataclass(frozen=True)
class AcceptancePolicy:
    required_checks: tuple[StageRequirement, ...] = field(
        default_factory=lambda: tuple(
            StageRequirement(stage, "configured_ontology_profile")
            for stage in CheckStage
        )
    )

    def __post_init__(self):
        _tuple(self.required_checks, StageRequirement, "/required_checks")
        if not self.required_checks or len(
            {r.stage for r in self.required_checks}
        ) != len(self.required_checks):
            raise ContractError(
                "acceptance_policy", "Require at least one check, with unique stages"
            )

    def blockers(self, report: CheckReport) -> tuple[Feedback, ...]:
        blockers = [f for f in report.feedback if f.severity is Severity.ERROR]
        results = {r.stage: r for r in report.results}
        for requirement in self.required_checks:
            result = results.get(requirement.stage)
            if result is None:
                code, message = (
                    "missing_check",
                    f"Missing required {requirement.stage.value} check",
                )
            elif result.profile != requirement.profile:
                code, message = (
                    "profile_mismatch",
                    f"{requirement.stage.value} profile differs from configured policy",
                )
            elif result.status is not CheckStatus.PASS:
                code, message = (
                    "nonpassing_check",
                    f"{requirement.stage.value} result is {result.status.value}",
                )
            else:
                continue
            blockers.append(
                Feedback(FeedbackCategory.LIFECYCLE, Severity.ERROR, code, message)
            )
        for result in report.results:
            if result.status in {CheckStatus.FAIL, CheckStatus.ERROR}:
                blockers.extend(result.feedback)
                if result.stage not in {r.stage for r in self.required_checks}:
                    blockers.append(
                        Feedback(
                            FeedbackCategory.LIFECYCLE,
                            Severity.ERROR,
                            "additional_failure",
                            f"Additional {result.stage.value} check failed",
                        )
                    )
        return tuple(blockers)


@dataclass(frozen=True)
class AcceptanceRequest:
    token: CandidateToken
    context: CandidateContext
    required_checks: tuple[StageRequirement, ...]
    contract_version: str = CONTRACT_VERSION

    def __post_init__(self):
        _version(self.contract_version)
        if not isinstance(self.token, CandidateToken) or not isinstance(
            self.context, CandidateContext
        ):
            raise ContractError(
                "acceptance_request", "Expected typed token and context"
            )
        AcceptancePolicy(self.required_checks)

    def to_dict(self) -> dict:
        return asdict(self)
