"""Revision-bound candidate RDF preparation using the existing individual planner."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from rdflib import OWL, RDF, RDFS, BNode, Graph, Literal, URIRef

from krrood_ontology.contracts import (
    CandidateContext,
    CandidateSnapshot,
    CandidateToken,
    CheckReport,
    CheckStage,
    CheckStatus,
    ContractError,
    CreateEntity,
    EntityReference,
    Feedback,
    FeedbackCategory,
    LiteralAssertion,
    ObjectAssertion,
    Severity,
    StageResult,
)
from krrood_ontology.individuals import IndividualPlanner
from krrood_ontology.runtime import GeneratedModel, IndividualLoadingError

PREFLIGHT_PROFILE = "asserted_named_individuals_v1"


def _graph(triples):
    graph = Graph()
    for triple in triples:
        graph.add(triple)
    return graph


@dataclass(frozen=True)
class PreparedCandidate:
    """Immutable RDF terms; graph access always returns a separate RDFLib graph.

    Generated IRIs are provisional and stable within a session across revisions.
    This record grants neither acceptance nor permission to construct native objects.
    """

    token: CandidateToken
    context: CandidateContext
    local_iris: tuple[tuple[str, str], ...]
    triples: frozenset
    additions: frozenset

    def __post_init__(self):
        if not isinstance(self.token, CandidateToken) or not isinstance(
            self.context, CandidateContext
        ):
            raise ContractError(
                "prepared_candidate", "Expected typed candidate token and context"
            )
        if not isinstance(self.triples, frozenset) or not isinstance(
            self.additions, frozenset
        ):
            raise ContractError(
                "prepared_candidate", "RDF snapshots must be immutable frozensets"
            )
        if not self.additions <= self.triples:
            raise ContractError(
                "prepared_candidate", "Additions must be part of the candidate graph"
            )
        if not isinstance(self.local_iris, tuple) or any(
            not isinstance(pair, tuple) or len(pair) != 2 for pair in self.local_iris
        ):
            raise ContractError(
                "prepared_candidate", "IRI assignments must be immutable pairs"
            )
        for local_id, iri in self.local_iris:
            EntityReference(local_id=local_id)
            EntityReference(iri=iri)
        if len({pair[0] for pair in self.local_iris}) != len(self.local_iris) or len(
            {pair[1] for pair in self.local_iris}
        ) != len(self.local_iris):
            raise ContractError(
                "prepared_candidate", "Local IDs and assigned IRIs must be unique"
            )

    def graph(self) -> Graph:
        return _graph(self.triples)

    def delta_graph(self) -> Graph:
        return _graph(self.additions)

    def iri_for(self, local_id: str) -> str:
        return dict(self.local_iris)[local_id]

    def require_current(self, snapshot: CandidateSnapshot) -> None:
        if self.token != snapshot.token or self.context != snapshot.context:
            raise ContractError(
                "stale_candidate", "Prepared RDF belongs to another candidate revision"
            )


@dataclass(frozen=True)
class PreparationResult:
    report: CheckReport
    candidate: PreparedCandidate | None

    def __post_init__(self):
        if not isinstance(self.report, CheckReport) or (
            self.candidate is not None
            and (
                not isinstance(self.candidate, PreparedCandidate)
                or self.candidate.token != self.report.token
                or not any(
                    r.stage is CheckStage.PREFLIGHT and r.status is CheckStatus.PASS
                    for r in self.report.results
                )
            )
        ):
            raise ContractError(
                "preparation_result",
                "Prepared RDF must belong to its passing preflight report",
            )


class _RequestError(ValueError):
    def __init__(self, code, message, location="", entity=None, property_iri=None):
        self.feedback = Feedback(
            FeedbackCategory.REQUEST,
            Severity.ERROR,
            code,
            message,
            location,
            entity=entity,
            property_iri=property_iri,
        )


class CandidatePreparer:
    """A trusted schema/model adapter, with a captured accepted RDF baseline.

    Recreate the adapter when the accepted revision changes. No native KRROOD
    symbol or relation is constructed. All semantic field checks are delegated
    to IndividualPlanner, and do not amount to OWL consistency checking.
    """

    def __init__(
        self,
        model: GeneratedModel,
        context: CandidateContext,
        schema_bytes: bytes,
        *,
        schema_format: str = "turtle",
        accepted_graph: Graph | None = None,
    ):
        if hashlib.sha256(schema_bytes).hexdigest() != context.schema_sha256:
            raise ContractError(
                "schema_changed", "Schema bytes differ from the authority context"
            )
        schema = Graph().parse(
            data=schema_bytes, format=schema_format, publicID=context.schema_id
        )
        # Reject pairing a registry with a closure that does not declare its terms.
        if any(
            (URIRef(iri), RDF.type, OWL.Class) not in schema
            and (URIRef(iri), RDF.type, RDFS.Class) not in schema
            for iri in model.classes
        ) or any(
            mapping.kind not in {"object", "datatype"}
            or (
                URIRef(iri),
                RDF.type,
                (
                    OWL.ObjectProperty
                    if mapping.kind == "object"
                    else OWL.DatatypeProperty
                ),
            )
            not in schema
            for iri, mapping in model.properties.items()
        ):
            raise ContractError(
                "model_schema_mismatch",
                "Generated model terms or property kinds differ from the schema closure",
            )
        # Freeze registry containers. Generated class definitions remain trusted code.
        self._model = GeneratedModel(
            MappingProxyType(dict(model.classes)),
            MappingProxyType(dict(model.properties)),
            frozenset(model.annotation_properties),
            tuple(model.bindings),
            model.binding_policy,
            model.entity_base,
        )
        self._context = context
        baseline = set(schema)
        if accepted_graph is not None:
            schema_types = IndividualPlanner(self._model).schema_types - {
                OWL.AllDifferent,
                RDF.List,
                OWL.NegativePropertyAssertion,
            }
            if any(
                kind in schema_types for kind in accepted_graph.objects(None, RDF.type)
            ):
                raise ContractError(
                    "accepted_schema",
                    "accepted_graph must contain additional A-Box facts, not a second ontology schema",
                )
            # RDF merge: separately supplied blank nodes have their own identity scope.
            blanks = {
                node: BNode()
                for node in accepted_graph.all_nodes()
                if isinstance(node, BNode)
            }
            baseline.update(
                tuple(blanks.get(term, term) for term in triple)
                for triple in accepted_graph
            )
        self._baseline = frozenset(baseline)
        self._occupied = frozenset(
            str(term)
            for triple in baseline
            for term in triple
            if isinstance(term, URIRef)
        )
        self._existing = frozenset(
            str(s)
            for s, p, o in baseline
            if p == RDF.type and str(o) in self._model.classes and isinstance(s, URIRef)
        )

    @property
    def context(self):
        return self._context

    @property
    def model(self) -> GeneratedModel:
        """Return the captured registry used by preflight."""
        return self._model

    @classmethod
    def from_manifest(
        cls,
        model: GeneratedModel,
        manifest: Path | str,
        *,
        schema_id: str,
        base_revision: str,
        accepted_graph: Graph | None = None,
    ):
        manifest = Path(manifest).resolve()
        context = CandidateContext.from_manifest(
            manifest,
            schema_id=schema_id,
            base_revision=base_revision,
        )
        data = json.loads(manifest.read_text(encoding="utf-8"))
        closure = (manifest.parent / data["runtime_ontology"]).resolve()
        # Recheck the path and digest after reading metadata; no import/network fetching.
        if not closure.is_relative_to(manifest.parent) or not closure.is_file():
            raise ContractError(
                "manifest_path", "Closure must be inside the generated output"
            )
        return cls(model, context, closure.read_bytes(), accepted_graph=accepted_graph)

    def check(self, snapshot: CandidateSnapshot) -> CheckReport:
        """CandidateChecker integration; no shared mutable last-result cache."""
        return self.prepare(snapshot).report

    def prepare(self, snapshot: CandidateSnapshot) -> PreparationResult:
        origins = {}
        try:
            rebuilt = CandidateSnapshot.build(
                snapshot.token.session_id,
                snapshot.token.revision,
                snapshot.context,
                snapshot.proposal,
            )
            if rebuilt.token != snapshot.token:
                raise _RequestError(
                    "candidate_digest", "Candidate token does not match its contents"
                )
            if snapshot.context != self._context:
                raise _RequestError(
                    "context_mismatch",
                    "Schema or accepted revision differs from this adapter",
                    "/context",
                )
            candidate = self._prepare(snapshot, origins)
            result = StageResult(
                CheckStage.PREFLIGHT, CheckStatus.PASS, PREFLIGHT_PROFILE
            )
        except _RequestError as error:
            candidate = None
            result = StageResult(
                CheckStage.PREFLIGHT,
                CheckStatus.FAIL,
                PREFLIGHT_PROFILE,
                (error.feedback,),
            )
        except IndividualLoadingError as error:
            candidate = None
            related = origins.get(error.iri, [])
            creation_error = error.code in {
                "unknown_type",
                "unrelated_types",
                "missing_type",
                "punning_unsupported",
            }
            related = [v for v in related if v[4] == creation_error]
            unique = {v[:4] for v in related}
            origin = next(iter(unique)) if len(unique) == 1 else None
            unsupported = error.code in {
                "unrelated_types",
                "punning_unsupported",
                "anonymous_individual",
                "identity_or_import_unsupported",
                "anonymous_annotation",
            } or (
                error.code == "unknown_property" and error.iri in self._model.properties
            )
            feedback = Feedback(
                (
                    FeedbackCategory.UNSUPPORTED
                    if unsupported
                    else FeedbackCategory.REQUEST
                ),
                Severity.ERROR,
                error.code,
                (f"Request {origin[0]}: " if origin else "") + str(error),
                origin[1] if origin else "",
                entity=origin[2] if origin else None,
                property_iri=origin[3] if origin else None,
            )
            result = StageResult(
                CheckStage.PREFLIGHT,
                CheckStatus.UNSUPPORTED if unsupported else CheckStatus.FAIL,
                PREFLIGHT_PROFILE,
                (feedback,),
            )
        remaining = tuple(
            StageResult(
                stage,
                CheckStatus.UNSUPPORTED,
                "not_available",
                (
                    Feedback(
                        FeedbackCategory.UNSUPPORTED,
                        Severity.WARNING,
                        "phase_not_implemented",
                        f"{stage.value} is not executed by candidate RDF preparation",
                    ),
                ),
            )
            for stage in CheckStage
            if stage != CheckStage.PREFLIGHT
        )
        return PreparationResult(
            CheckReport(snapshot.token, (result,) + remaining), candidate
        )

    def _prepare(self, snapshot, origins):
        operations = snapshot.proposal.operations
        local = {}
        allocated = set()
        triples = set()

        def origin(index, operation, entity=None, prop=None):
            return (
                operation.request_id,
                f"/operations/{index}",
                entity,
                prop,
                isinstance(operation, CreateEntity),
            )

        # Allocate all entities first so assertions may precede their creation requests.
        for index, operation in enumerate(operations):
            if not isinstance(operation, CreateEntity):
                continue
            ref = EntityReference(local_id=operation.local_id)
            for kind in operation.class_iris:
                if kind not in self._model.classes:
                    raise _RequestError(
                        "unknown_type",
                        "Class is absent from the generated model",
                        f"/operations/{index}/class_iris",
                        ref,
                    )
            iri = operation.iri or (
                f"urn:krrood-ontology:candidate:{snapshot.token.session_id}:{operation.local_id}"
            )
            if iri in self._occupied or iri in allocated:
                raise _RequestError(
                    "iri_collision",
                    "Created entity IRI is already in use",
                    f"/operations/{index}/iri",
                    ref,
                )
            local[operation.local_id] = iri
            allocated.add(iri)
            origins.setdefault(iri, []).append(origin(index, operation, ref))
            for kind in operation.class_iris:
                triples.add((URIRef(iri), RDF.type, URIRef(kind)))

        def resolve(ref, index, field):
            if ref.local_id is not None:
                if ref.local_id in local:
                    return local[ref.local_id]
            elif ref.iri in self._existing:
                return ref.iri
            raise _RequestError(
                "unresolved_reference",
                "Reference must identify a proposed local entity or explicitly typed baseline individual",
                f"/operations/{index}/{field}",
                ref,
            )

        for index, operation in enumerate(operations):
            if isinstance(operation, CreateEntity):
                continue
            mapping = self._model.properties.get(operation.property_iri)
            if mapping is None:
                raise _RequestError(
                    "unknown_property",
                    "Property is absent from the generated model",
                    f"/operations/{index}/property_iri",
                    operation.subject,
                    operation.property_iri,
                )
            expected = (
                "object" if isinstance(operation, ObjectAssertion) else "datatype"
            )
            if mapping.kind != expected:
                raise _RequestError(
                    "property_kind",
                    "Assertion kind differs from the generated property kind",
                    f"/operations/{index}/property_iri",
                    operation.subject,
                    operation.property_iri,
                )
            subject = resolve(operation.subject, index, "subject")
            info = origin(index, operation, operation.subject, operation.property_iri)
            origins.setdefault(subject, []).append(info)
            origins.setdefault(operation.property_iri, []).append(info)
            if isinstance(operation, ObjectAssertion):
                target = resolve(operation.target, index, "target")
                origins.setdefault(target, []).append(info)
                value = URIRef(target)
            elif isinstance(operation, LiteralAssertion):
                value = Literal(
                    operation.value.lexical,
                    datatype=(
                        URIRef(operation.value.datatype_iri)
                        if operation.value.datatype_iri
                        else None
                    ),
                    lang=operation.value.language,
                    normalize=False,
                )
            triples.add((URIRef(subject), URIRef(operation.property_iri), value))
        merged = self._baseline | frozenset(triples)
        # Existing reusable preflight handles types, domains/ranges, local binding scope,
        # explicit identity axioms and RDFLib literal conversion without native objects.
        IndividualPlanner(self._model, identity_policy="distinct").plan_graph(
            _graph(merged)
        )
        return PreparedCandidate(
            snapshot.token,
            snapshot.context,
            tuple(sorted(local.items())),
            merged,
            frozenset(triples) - self._baseline,
        )
