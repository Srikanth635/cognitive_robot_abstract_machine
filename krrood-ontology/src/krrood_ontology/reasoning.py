"""Explicit, isolated ontology reasoner backends and candidate-checker integration."""

from __future__ import annotations

import math
import pickle
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol, runtime_checkable

from rdflib import OWL

from krrood_ontology.construction import (
    CandidateConstructor,
    ConstructionArtifact,
    _dump_worker_record,
)
from krrood_ontology.contracts import (
    CandidateSnapshot,
    CheckReport,
    CheckStage,
    CheckStatus,
    ContractError,
    EntityReference,
    Feedback,
    FeedbackCategory,
    Severity,
    StageResult,
)
from krrood_ontology.preparation import PreparedCandidate


class ReasonerKind(str, Enum):
    OWL_RL = "owlrl"
    HERMIT = "hermit"


@dataclass(frozen=True)
class ReasonerCapabilities:
    """Declare the exact semantics behind a reasoner-stage result."""

    profile: str
    engine: str
    semantics: str
    can_certify_consistency: bool
    checks_owl2_dl_profile: bool
    materializes_inferences: bool = False
    provides_explanations: bool = False

    def __post_init__(self):
        if any(
            not isinstance(v, str) or not v.strip()
            for v in (self.profile, self.engine, self.semantics)
        ) or any(
            type(v) is not bool
            for v in (
                self.can_certify_consistency,
                self.checks_owl2_dl_profile,
                self.materializes_inferences,
                self.provides_explanations,
            )
        ):
            raise ContractError(
                "reasoner_capabilities",
                "Expected explicit semantics and boolean capability flags",
            )


OWL_RL_CAPABILITIES = ReasonerCapabilities(
    "owlrl_diagnostics_v1",
    "owlrl",
    "OWL RL/RDF rule diagnostics; no completeness certification",
    False,
    False,
)
HERMIT_CAPABILITIES = ReasonerCapabilities(
    "hermit_owl2_dl_consistency_v1",
    "HermiT/OWLAPI",
    "OWL 2 DL direct semantics of OWLAPI-loaded axioms; open world; no unique-name assumption",
    True,
    True,
)
UNAVAILABLE_CAPABILITIES = ReasonerCapabilities(
    "not_available",
    "none",
    "No reasoning executed",
    False,
    False,
)


@dataclass(frozen=True)
class ProfileFinding:
    """An OWLAPI profile finding, with its own functional-syntax axiom identity."""

    code: str
    axiom_text: str
    axiom_id: str | None

    def __post_init__(self):
        Feedback(
            FeedbackCategory.UNSUPPORTED,
            Severity.ERROR,
            self.code,
            "OWLAPI profile finding",
        )
        if not isinstance(self.axiom_text, str):
            raise ContractError(
                "profile_finding", "Expected functional-syntax axiom text"
            )
        if self.axiom_id is not None:
            EntityReference(iri=self.axiom_id)


@dataclass(frozen=True)
class ReasonerEvidence:
    """Immutable revision-bound outcome; asserted candidate RDF remains unchanged."""

    prepared: PreparedCandidate
    capabilities: ReasonerCapabilities
    status: CheckStatus
    consistent: bool | None
    feedback: tuple[Feedback, ...] = ()
    engine_version: str = "not_available"
    engine_sha256: str | None = None
    worker_pid: int | None = None
    axioms_checked: int | None = None
    profile_findings: tuple[ProfileFinding, ...] = ()
    unparsed_triples: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.prepared, PreparedCandidate) or not isinstance(
            self.capabilities, ReasonerCapabilities
        ):
            raise ContractError(
                "reasoner_evidence", "Expected typed candidate and capabilities"
            )
        StageResult(
            CheckStage.REASONER, self.status, self.capabilities.profile, self.feedback
        )
        if not isinstance(self.engine_version, str) or not self.engine_version:
            raise ContractError("reasoner_evidence", "Expected engine provenance text")
        if (
            self.consistent is not None
            and type(self.consistent) is not bool
            or self.status is CheckStatus.PASS
            and (
                self.consistent is not True
                or not self.capabilities.can_certify_consistency
            )
            or self.status is CheckStatus.FAIL
            and self.consistent is not False
            or self.status not in {CheckStatus.PASS, CheckStatus.FAIL}
            and self.consistent is not None
            or not isinstance(self.profile_findings, tuple)
            or any(not isinstance(f, ProfileFinding) for f in self.profile_findings)
            or not isinstance(self.unparsed_triples, tuple)
            or any(not isinstance(t, str) for t in self.unparsed_triples)
            or self.status is CheckStatus.PASS
            and (self.profile_findings or self.unparsed_triples)
            or self.worker_pid is not None
            and (type(self.worker_pid) is not int or self.worker_pid <= 0)
            or self.axioms_checked is not None
            and (type(self.axioms_checked) is not int or self.axioms_checked < 0)
        ):
            raise ContractError(
                "reasoner_evidence",
                "Status, consistency and immutable evidence must agree",
            )

    def graph(self):
        return self.prepared.graph()

    def require_current(self, snapshot: CandidateSnapshot) -> None:
        self.prepared.require_current(snapshot)


@runtime_checkable
class ReasonerBackend(Protocol):
    """Replaceable trusted backend over an immutable prepared candidate."""

    @property
    def capabilities(self) -> ReasonerCapabilities: ...

    def evaluate(self, prepared: PreparedCandidate) -> ReasonerEvidence: ...


def _finding(category, severity, code, message):
    return (Feedback(category, severity, code, message),)


@dataclass(frozen=True)
class UnavailableReasoner:
    """Keep reasoner approval blocked until the server explicitly selects a backend."""

    capabilities: ReasonerCapabilities = UNAVAILABLE_CAPABILITIES

    def evaluate(self, prepared: PreparedCandidate) -> ReasonerEvidence:
        return ReasonerEvidence(
            prepared,
            self.capabilities,
            CheckStatus.UNSUPPORTED,
            None,
            _finding(
                FeedbackCategory.UNSUPPORTED,
                Severity.WARNING,
                "reasoner_not_configured",
                "No ontology reasoner backend is configured",
            ),
        )


@dataclass(frozen=True)
class _ReasonerJob:
    prepared: PreparedCandidate
    kind: ReasonerKind
    max_heap_mb: int
    jvm_path: str | None


@dataclass(frozen=True)
class IsolatedReasoner:
    """Run an existing engine inside a fresh worker, with no persistent JVM or RDF store.

    The HermiT worker embeds Java through JPype, so timeout kills and waits for
    the entire worker/JVM process. No separate Java child remains running.
    """

    kind: ReasonerKind
    timeout_seconds: float = 90
    max_heap_mb: int = 512
    jvm_path: str | None = None

    def __post_init__(self):
        if not isinstance(self.kind, ReasonerKind):
            raise ValueError("Use an explicit ReasonerKind")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        if type(self.max_heap_mb) is not int or self.max_heap_mb < 1:
            raise ValueError("max_heap_mb must be a positive integer")
        if self.jvm_path is not None and (
            not isinstance(self.jvm_path, str) or not self.jvm_path
        ):
            raise ValueError("jvm_path must be an explicit shared-library path or None")

    @property
    def capabilities(self) -> ReasonerCapabilities:
        return (
            HERMIT_CAPABILITIES
            if self.kind is ReasonerKind.HERMIT
            else OWL_RL_CAPABILITIES
        )

    def evaluate(self, prepared: PreparedCandidate) -> ReasonerEvidence:
        if not isinstance(prepared, PreparedCandidate):
            raise TypeError("Reasoning requires a prepared candidate")
        if any(prepared.graph().triples((None, OWL.imports, None))):
            return ReasonerEvidence(
                prepared,
                self.capabilities,
                CheckStatus.UNSUPPORTED,
                None,
                _finding(
                    FeedbackCategory.UNSUPPORTED,
                    Severity.ERROR,
                    "reasoner_unresolved_import",
                    "Reasoner input must contain the resolved local closure, without owl:imports",
                ),
            )
        try:
            evidence = self._run_worker(prepared)
            if (
                not isinstance(evidence, ReasonerEvidence)
                or evidence.prepared != prepared
                or evidence.capabilities != self.capabilities
            ):
                raise ContractError(
                    "reasoner_protocol", "Worker returned foreign reasoner evidence"
                )
            evidence.__post_init__()
            return evidence
        except subprocess.TimeoutExpired:
            code, message = (
                "reasoner_timeout",
                "Ontology reasoning exceeded its worker deadline",
            )
        except ContractError as error:
            code, message = error.code, str(error)
        except Exception as error:
            code, message = (
                "reasoner_execution",
                f"Reasoner execution failed ({type(error).__name__})",
            )
        return ReasonerEvidence(
            prepared,
            self.capabilities,
            CheckStatus.ERROR,
            None,
            _finding(FeedbackCategory.EXECUTION, Severity.ERROR, code, message),
        )

    def _run_worker(self, prepared: PreparedCandidate) -> ReasonerEvidence:
        job = _ReasonerJob(prepared, self.kind, self.max_heap_mb, self.jvm_path)
        with TemporaryDirectory(prefix="krrood-reasoner-") as temporary:
            request, response = (
                Path(temporary) / "request.pickle",
                Path(temporary) / "response.pickle",
            )
            request.write_bytes(_dump_worker_record(job))
            completed = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    str(Path(__file__).with_name("reasoning_worker.py")),
                    str(Path(__file__).parents[1]),
                    str(request),
                    str(response),
                ],
                capture_output=True,
                timeout=self.timeout_seconds,
            )
            if completed.returncode or not response.is_file():
                raise ContractError(
                    "reasoner_worker", "Reasoner worker exited without valid evidence"
                )
            return pickle.loads(response.read_bytes())


@dataclass(frozen=True)
class ReasoningResult:
    """Construction evidence and ontology review for the same candidate revision."""

    report: CheckReport
    construction: ConstructionArtifact | None
    evidence: ReasonerEvidence | None

    def __post_init__(self):
        if not isinstance(self.report, CheckReport):
            raise ContractError("reasoning_result", "Expected a typed review report")
        if (
            self.construction is not None
            and self.construction.prepared.token != self.report.token
        ):
            raise ContractError(
                "reasoning_result", "Construction belongs to another revision"
            )
        if self.evidence is not None and (
            self.construction is None
            or self.evidence.prepared != self.construction.prepared
            or not any(
                r.stage is CheckStage.REASONER
                and r.status is self.evidence.status
                and r.profile == self.evidence.capabilities.profile
                for r in self.report.results
            )
        ):
            raise ContractError(
                "reasoning_result",
                "Reasoner evidence must belong to its construction and report",
            )


@dataclass(frozen=True)
class CandidateReasoner:
    """Construct a candidate once, then review its exact preserved RDF with a backend."""

    constructor: CandidateConstructor
    backend: ReasonerBackend = field(default_factory=UnavailableReasoner)

    def __post_init__(self):
        if not isinstance(self.constructor, CandidateConstructor) or not isinstance(
            self.backend, ReasonerBackend
        ):
            raise TypeError("Expected a constructor and reasoner backend")

    def check(self, snapshot: CandidateSnapshot) -> CheckReport:
        return self.review(snapshot).report

    def review(self, snapshot: CandidateSnapshot) -> ReasoningResult:
        construction = self.constructor.construct(snapshot)
        evidence = None
        if construction.artifact is None:
            result = StageResult(
                CheckStage.REASONER,
                CheckStatus.UNKNOWN,
                self.backend.capabilities.profile,
                _finding(
                    FeedbackCategory.LIFECYCLE,
                    Severity.INFO,
                    "construction_required",
                    "Reasoning requires passing candidate construction",
                ),
            )
        else:
            try:
                value = self.backend.evaluate(construction.artifact.prepared)
                if (
                    not isinstance(value, ReasonerEvidence)
                    or value.prepared != construction.artifact.prepared
                    or value.capabilities != self.backend.capabilities
                ):
                    raise ContractError(
                        "reasoner_protocol",
                        "Backend returned evidence for another candidate or profile",
                    )
                value.__post_init__()
                result = StageResult(
                    CheckStage.REASONER,
                    value.status,
                    value.capabilities.profile,
                    value.feedback,
                )
                evidence = value
            except ContractError as error:
                result = StageResult(
                    CheckStage.REASONER,
                    CheckStatus.ERROR,
                    self.backend.capabilities.profile,
                    _finding(
                        FeedbackCategory.EXECUTION,
                        Severity.ERROR,
                        error.code,
                        str(error),
                    ),
                )
            except Exception as error:
                result = StageResult(
                    CheckStage.REASONER,
                    CheckStatus.ERROR,
                    self.backend.capabilities.profile,
                    _finding(
                        FeedbackCategory.EXECUTION,
                        Severity.ERROR,
                        "reasoner_execution",
                        f"Reasoner backend failed ({type(error).__name__})",
                    ),
                )
        report = CheckReport(
            snapshot.token,
            tuple(
                r
                for r in construction.report.results
                if r.stage is not CheckStage.REASONER
            )
            + (result,),
            construction.report.feedback,
        )
        return ReasoningResult(report, construction.artifact, evidence)
