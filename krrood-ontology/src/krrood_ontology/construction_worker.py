"""Execute trusted native materialization and return immutable evidence on process exit."""

import importlib
import os
import pickle
import sys
from pathlib import Path


def execute(job):
    from krrood.symbol_graph.symbol_graph import SymbolGraph
    from rdflib import Literal, URIRef

    from krrood_ontology.construction import (
        ConstructionArtifact,
        ObjectFieldView,
        ObjectView,
        _artifact_digest,
        _model_digest,
    )
    from krrood_ontology.contracts import (
        CandidateContext,
        ContractError,
        EntityReference,
        LiteralValue,
    )
    from krrood_ontology.export import IndividualExporter
    from krrood_ontology.individuals import IndividualLoader
    from krrood_ontology.relation_runtime import ModelRelations

    if _artifact_digest(job.directory, job.package_name) != job.generated_sha256:
        raise ContractError(
            "generated_artifact_changed", "Generated files changed before worker import"
        )
    context = CandidateContext.from_manifest(
        job.directory / "manifest.json",
        schema_id=job.prepared.context.schema_id,
        base_revision=job.prepared.context.base_revision,
    )
    if context != job.prepared.context:
        raise ContractError(
            "context_mismatch", "Worker schema differs from candidate authority"
        )
    sys.path.insert(0, str(job.directory))
    package = importlib.import_module(job.package_name)
    if (
        Path(package.__file__).resolve()
        != (job.directory / job.package_name / "__init__.py").resolve()
    ):
        raise ContractError(
            "generated_artifact", "Worker imported another generated package"
        )
    model = package.MODEL
    if _model_digest(model) != job.model_sha256:
        raise ContractError(
            "model_registry_changed", "Worker registry differs from candidate preflight"
        )
    native = SymbolGraph()
    before = len(native.wrapped_instances)
    loaded = IndividualLoader(model).load_graph(
        job.prepared.graph(), identity_policy="distinct"
    )
    seeds = 0
    if job.seed_reflexivity:
        helper = getattr(package, "model_relations", None)
        if helper is None:
            raise ContractError(
                "reflexivity_unavailable",
                "Generated model has no scoped reflexivity helper",
            )
        seeds = helper.initialize_reflexivity(loaded).added
    current = IndividualExporter(model).export(
        loaded, mode="current", include_inferred=False
    )
    if frozenset(current.graph) != job.prepared.triples:
        raise ContractError(
            "construction_projection",
            "Constructed fields do not preserve asserted candidate RDF",
        )
    identities = {id(obj): iri for iri, obj in loaded.by_iri.items()}
    if len(identities) != len(loaded.by_iri):
        raise ContractError(
            "construction_identity", "Different IRIs shared one native object"
        )
    objects = []
    for index, (iri, obj) in enumerate(sorted(loaded.by_iri.items())):
        concrete = type(obj)
        fields = []
        for prop_iri in sorted(model.properties):
            mapping = model.mapping_for(
                prop_iri, concrete.__ontology_iri__, require_applicable=True
            )
            if mapping is None:
                continue
            values = []
            for value in tuple(getattr(obj, mapping.name)):
                if mapping.kind == "object":
                    target = identities.get(id(value))
                    if target is None:
                        raise ContractError(
                            "construction_identity",
                            "Native reference leaves the candidate registry",
                        )
                    values.append(EntityReference(iri=target))
                else:
                    term = Literal(
                        value,
                        datatype=(
                            URIRef(mapping.range_iri) if mapping.range_iri else None
                        ),
                    )
                    values.append(
                        LiteralValue(
                            str(term),
                            str(term.datatype) if term.datatype else None,
                            term.language,
                        )
                    )
            values.sort(
                key=lambda v: (
                    v.iri
                    if isinstance(v, EntityReference)
                    else (v.lexical, v.datatype_iri or "", v.language or "")
                )
            )
            fields.append(
                ObjectFieldView(
                    prop_iri, mapping.kind, mapping.container, tuple(values)
                )
            )
        objects.append(
            ObjectView(
                iri,
                concrete.__ontology_iri__,
                f"{concrete.__module__}.{concrete.__qualname__}",
                index,
                loaded.asserted_types[iri],
                tuple(fields),
            )
        )
    if _artifact_digest(job.directory, job.package_name) != job.generated_sha256:
        raise ContractError(
            "generated_artifact_changed", "Generated files changed during construction"
        )
    return ConstructionArtifact(
        job.prepared,
        tuple(objects),
        ModelRelations(model, {}).snapshot(loaded),
        os.getpid(),
        len(native.wrapped_instances) - before,
        seeds,
        True,
        job.generated_sha256,
    )


def main():
    runtime, request, response = sys.argv[1:]
    sys.path.insert(0, runtime)
    from krrood_ontology.construction import (
        _ConstructionJob,
        _dump_worker_record,
        _feedback,
        _WorkerResult,
    )
    from krrood_ontology.contracts import (
        CheckStatus,
        ContractError,
        Feedback,
        FeedbackCategory,
        Severity,
    )
    from krrood_ontology.export import IndividualExportError
    from krrood_ontology.runtime import IndividualLoadingError

    try:
        job = pickle.loads(Path(request).read_bytes())
        if not isinstance(job, _ConstructionJob):
            raise ContractError(
                "construction_protocol", "Expected a trusted construction job"
            )
        result = _WorkerResult(CheckStatus.PASS, artifact=execute(job))
    except IndividualExportError as error:
        result = _WorkerResult(
            CheckStatus.UNSUPPORTED,
            (
                Feedback(
                    FeedbackCategory.UNSUPPORTED,
                    Severity.ERROR,
                    "projection_unsupported",
                    str(error),
                ),
            ),
        )
    except IndividualLoadingError as error:
        result = _WorkerResult(CheckStatus.FAIL, _feedback(error.code, str(error)))
    except ContractError as error:
        result = _WorkerResult(CheckStatus.ERROR, _feedback(error.code, str(error)))
    except Exception as error:
        result = _WorkerResult(
            CheckStatus.ERROR,
            _feedback(
                "construction_failed",
                f"Native construction failed ({type(error).__name__})",
            ),
        )
    Path(response).write_bytes(_dump_worker_record(result))


if __name__ == "__main__":
    main()
