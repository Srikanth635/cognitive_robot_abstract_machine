"""Construct generated objects in an ephemeral native graph owned by a worker."""

from __future__ import annotations

import copyreg
import hashlib
import io
import json
import math
import pickle
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

from rdflib import Literal

from krrood_ontology.config import BootstrapOptions
from krrood_ontology.contracts import (
    CandidateContext,
    CandidateSnapshot,
    CheckReport,
    CheckStage,
    CheckStatus,
    ContractError,
    EntityReference,
    Feedback,
    FeedbackCategory,
    LiteralValue,
    Severity,
    StageResult,
)
from krrood_ontology.preparation import CandidatePreparer, PreparedCandidate
from krrood_ontology.relation_runtime import RelationFact
from krrood_ontology.runtime import GeneratedModel
from krrood_ontology.verification import PackageVerifier

CONSTRUCTION_PROFILE = "isolated_generated_objects_v1"


def _restore_literal(lexical, language, datatype):
    return Literal(lexical, lang=language, datatype=datatype, normalize=False)


def _reduce_literal(term):
    return _restore_literal, (str(term), term.language, term.datatype)


def _dump_worker_record(record) -> bytes:
    """Use private pickle IPC while preserving RDFLib literal lexical spelling.

    RDFLib's default pickle reduction reconstructs literals with normalization.
    A per-pickler hook avoids changing RDFLib's process-wide normalization flag.
    """
    stream = io.BytesIO()
    pickler = pickle.Pickler(stream)
    pickler.dispatch_table = {**copyreg.dispatch_table, Literal: _reduce_literal}
    pickler.dump(record)
    return stream.getvalue()


@dataclass(frozen=True)
class ObjectFieldView:
    """Native field values projected to resolved references or converted RDF literals."""

    property_iri: str
    kind: str
    container: str
    values: tuple[EntityReference | LiteralValue, ...]

    def __post_init__(self):
        EntityReference(iri=self.property_iri)
        expected = EntityReference if self.kind == "object" else LiteralValue
        if (
            self.kind not in {"object", "datatype"}
            or self.container not in {"set", "list"}
            or not isinstance(self.values, tuple)
            or any(not isinstance(v, expected) for v in self.values)
            or (self.kind == "object" and any(v.iri is None for v in self.values))
        ):
            raise ContractError(
                "object_field_view", "Expected immutable typed field values"
            )


@dataclass(frozen=True)
class ObjectView:
    """Read-only evidence of a concrete object; identity index is local to this artifact."""

    iri: str
    class_iri: str
    python_type: str
    identity_index: int
    asserted_types: tuple[str, ...]
    fields: tuple[ObjectFieldView, ...]

    def __post_init__(self):
        EntityReference(iri=self.iri)
        EntityReference(iri=self.class_iri)
        if (
            not isinstance(self.asserted_types, tuple)
            or self.class_iri not in self.asserted_types
            or not isinstance(self.fields, tuple)
            or any(not isinstance(v, ObjectFieldView) for v in self.fields)
            or len({v.property_iri for v in self.fields}) != len(self.fields)
            or type(self.identity_index) is not int
            or self.identity_index < 0
            or not isinstance(self.python_type, str)
            or not self.python_type
        ):
            raise ContractError(
                "object_view", "Expected a concrete immutable object view"
            )
        for iri in self.asserted_types:
            EntityReference(iri=iri)

    def field(self, property_iri: str) -> ObjectFieldView:
        return {v.property_iri: v for v in self.fields}[property_iri]


@dataclass(frozen=True)
class ConstructionArtifact:
    """Verified RDF and native-object evidence after the worker has exited.

    No live Symbol, descriptor, or mutable native collection crosses this boundary.
    Relations record KRROOD's first-insertion inference flags, not proof history.
    """

    prepared: PreparedCandidate
    objects: tuple[ObjectView, ...]
    relations: tuple[RelationFact, ...]
    worker_pid: int
    native_symbols_created: int
    reflexive_seeds: int
    roundtrip_equal: bool
    generated_sha256: str

    def __post_init__(self):
        if (
            not isinstance(self.prepared, PreparedCandidate)
            or not isinstance(self.objects, tuple)
            or any(not isinstance(v, ObjectView) for v in self.objects)
            or not isinstance(self.relations, tuple)
            or any(not isinstance(v, RelationFact) for v in self.relations)
            or self.roundtrip_equal is not True
            or type(self.worker_pid) is not int
            or self.worker_pid <= 0
            or any(
                type(n) is not int or n < 0
                for n in (self.native_symbols_created, self.reflexive_seeds)
            )
        ):
            raise ContractError(
                "construction_artifact",
                "Expected verified immutable construction evidence",
            )
        iris = {v.iri for v in self.objects}
        if (
            len(iris) != len(self.objects)
            or len({v.identity_index for v in self.objects}) != len(self.objects)
            or any(
                v.iri not in iris
                for obj in self.objects
                for f in obj.fields
                if f.kind == "object"
                for v in f.values
            )
            or any(
                f.subject_iri not in iris or f.object_iri not in iris
                for f in self.relations
            )
        ):
            raise ContractError(
                "construction_artifact",
                "Object identities and references must stay inside the registry",
            )

    def object_for_iri(self, iri: str) -> ObjectView:
        return {v.iri: v for v in self.objects}[iri]

    def object_for_local_id(self, local_id: str) -> ObjectView:
        return self.object_for_iri(self.prepared.iri_for(local_id))

    def graph(self):
        return self.prepared.graph()

    def require_current(self, snapshot: CandidateSnapshot) -> None:
        self.prepared.require_current(snapshot)


@dataclass(frozen=True)
class ConstructionResult:
    report: CheckReport
    artifact: ConstructionArtifact | None

    def __post_init__(self):
        if not isinstance(self.report, CheckReport) or (
            self.artifact is not None
            and (
                not isinstance(self.artifact, ConstructionArtifact)
                or self.artifact.prepared.token != self.report.token
                or not any(
                    r.stage is CheckStage.CONSTRUCTION and r.status is CheckStatus.PASS
                    for r in self.report.results
                )
            )
        ):
            raise ContractError(
                "construction_result",
                "Artifact must belong to its passing construction report",
            )


@dataclass(frozen=True)
class _ConstructionJob:
    prepared: PreparedCandidate
    directory: Path
    package_name: str
    generated_sha256: str
    model_sha256: str
    seed_reflexivity: bool


@dataclass(frozen=True)
class _WorkerResult:
    status: CheckStatus
    feedback: tuple[Feedback, ...] = ()
    artifact: ConstructionArtifact | None = None


def _model_digest(model: GeneratedModel) -> str:
    """Fingerprint registry metadata captured by preflight and reimported by the worker."""
    content = {
        "classes": [
            (
                iri,
                c.__module__,
                c.__qualname__,
                [getattr(parent, "__ontology_iri__", None) for parent in c.__mro__],
            )
            for iri, c in sorted(model.classes.items())
        ],
        "properties": [asdict(v) for _, v in sorted(model.properties.items())],
        "bindings": [asdict(v) for v in model.bindings],
        "binding_policy": model.binding_policy,
        "annotations": sorted(model.annotation_properties),
        "entity_base": (
            (model.entity_base.__module__, model.entity_base.__qualname__)
            if model.entity_base
            else None
        ),
    }
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def _artifact_digest(directory: Path, package_name: str) -> str:
    """Pin exact generated code and closure bytes; never fetch ontology imports."""
    manifest = directory / "manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    package = directory / package_name
    paths = [manifest, directory / data["runtime_ontology"], *package.rglob("*.py")]
    if not (package / "__init__.py").is_file():
        raise ContractError("generated_artifact", "Missing generated package")
    records = []
    for path in sorted(paths):
        resolved = path.resolve()
        if not resolved.is_relative_to(directory) or not resolved.is_file():
            raise ContractError(
                "generated_artifact", "Generated files must stay inside their directory"
            )
        records.append(
            (
                str(path.relative_to(directory)),
                hashlib.sha256(resolved.read_bytes()).hexdigest(),
            )
        )
    return hashlib.sha256(json.dumps(records).encode()).hexdigest()


def _feedback(code: str, message: str) -> tuple[Feedback, ...]:
    return (Feedback(FeedbackCategory.EXECUTION, Severity.ERROR, code, message),)


@dataclass(frozen=True)
class CandidateConstructor:
    """Use a fresh process per construction, without retaining or clearing parent Symbols.

    Generated modules are trusted server code. Process isolation owns the native
    graph lifecycle; it is not a sandbox for untrusted Python or filesystem access.
    """

    preparer: CandidatePreparer
    directory: Path
    package_name: str
    timeout_seconds: float = 45
    seed_reflexivity: bool = False
    _generated_sha256: str = field(init=False, repr=False)
    _model_sha256: str = field(init=False, repr=False)

    def __post_init__(self):
        options = BootstrapOptions(self.directory, self.package_name)
        object.__setattr__(self, "directory", options.output_directory)
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        if type(self.seed_reflexivity) is not bool:
            raise ValueError("seed_reflexivity must be a boolean")
        self._require_context()
        fingerprint = _artifact_digest(self.directory, self.package_name)
        PackageVerifier().verify(self.directory, self.package_name, preflight=True)
        if _artifact_digest(self.directory, self.package_name) != fingerprint:
            raise ContractError(
                "generated_artifact_changed",
                "Generated files changed during verification",
            )
        object.__setattr__(self, "_generated_sha256", fingerprint)
        object.__setattr__(self, "_model_sha256", _model_digest(self.preparer.model))

    def _require_context(self):
        context = CandidateContext.from_manifest(
            self.directory / "manifest.json",
            schema_id=self.preparer.context.schema_id,
            base_revision=self.preparer.context.base_revision,
        )
        if context != self.preparer.context:
            raise ContractError(
                "context_mismatch",
                "Constructor artifact differs from the captured schema",
            )

    def check(self, snapshot: CandidateSnapshot) -> CheckReport:
        return self.construct(snapshot).report

    def construct(self, snapshot: CandidateSnapshot) -> ConstructionResult:
        prepared = self.preparer.prepare(snapshot)
        artifact = None
        if prepared.candidate is None:
            result = StageResult(
                CheckStage.CONSTRUCTION,
                CheckStatus.UNKNOWN,
                CONSTRUCTION_PROFILE,
                (
                    Feedback(
                        FeedbackCategory.LIFECYCLE,
                        Severity.INFO,
                        "preflight_required",
                        "Construction requires a passing preflight",
                    ),
                ),
            )
        else:
            try:
                if (
                    _artifact_digest(self.directory, self.package_name)
                    != self._generated_sha256
                ):
                    raise ContractError(
                        "generated_artifact_changed",
                        "Generated files changed since constructor setup",
                    )
                worker = self._run_worker(prepared.candidate)
                artifact = worker.artifact
                result = StageResult(
                    CheckStage.CONSTRUCTION,
                    worker.status,
                    CONSTRUCTION_PROFILE,
                    worker.feedback,
                )
            except subprocess.TimeoutExpired:
                result = StageResult(
                    CheckStage.CONSTRUCTION,
                    CheckStatus.ERROR,
                    CONSTRUCTION_PROFILE,
                    _feedback(
                        "construction_timeout",
                        "Candidate construction exceeded its worker deadline",
                    ),
                )
            except ContractError as error:
                result = StageResult(
                    CheckStage.CONSTRUCTION,
                    CheckStatus.ERROR,
                    CONSTRUCTION_PROFILE,
                    _feedback(error.code, str(error)),
                )
            except Exception as error:
                result = StageResult(
                    CheckStage.CONSTRUCTION,
                    CheckStatus.ERROR,
                    CONSTRUCTION_PROFILE,
                    _feedback(
                        "construction_execution",
                        f"Construction execution failed ({type(error).__name__})",
                    ),
                )
        report = CheckReport(
            snapshot.token,
            tuple(
                result if r.stage is CheckStage.CONSTRUCTION else r
                for r in prepared.report.results
            ),
        )
        return ConstructionResult(report, artifact)

    def _run_worker(self, prepared: PreparedCandidate) -> _WorkerResult:
        job = _ConstructionJob(
            prepared,
            self.directory,
            self.package_name,
            self._generated_sha256,
            self._model_sha256,
            self.seed_reflexivity,
        )
        with TemporaryDirectory(prefix="krrood-candidate-") as temporary:
            request, response = (
                Path(temporary) / "request.pickle",
                Path(temporary) / "response.pickle",
            )
            # Private trusted IPC only. These files are never agent input or a persistence API.
            request.write_bytes(_dump_worker_record(job))
            completed = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    str(Path(__file__).with_name("construction_worker.py")),
                    str(Path(__file__).parents[1]),
                    str(request),
                    str(response),
                ],
                capture_output=True,
                timeout=self.timeout_seconds,
            )
            if completed.returncode or not response.is_file():
                raise ContractError(
                    "construction_worker",
                    "Construction worker exited without a valid response",
                )
            value = pickle.loads(response.read_bytes())
        if not isinstance(value, _WorkerResult):
            raise ContractError(
                "construction_protocol", "Invalid construction worker response"
            )
        StageResult(
            CheckStage.CONSTRUCTION, value.status, CONSTRUCTION_PROFILE, value.feedback
        )
        if (value.status is CheckStatus.PASS) != (value.artifact is not None):
            raise ContractError(
                "construction_protocol", "Worker artifact and status disagree"
            )
        if value.artifact is not None:
            if (
                not isinstance(value.artifact, ConstructionArtifact)
                or value.artifact.prepared != prepared
                or value.artifact.generated_sha256 != self._generated_sha256
            ):
                raise ContractError(
                    "construction_protocol",
                    "Worker artifact belongs to another construction",
                )
            value.artifact.__post_init__()
        return value
