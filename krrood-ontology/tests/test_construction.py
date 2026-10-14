"""Revision-bound worker construction through real KRROOD generated models."""

import importlib
import os
import shutil
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
from krrood.symbol_graph.symbol_graph import SymbolGraph
from rdflib import XSD

from krrood_ontology import CompilationOptions, OntologyCompiler, OntologyInput
from krrood_ontology.construction import CandidateConstructor
from krrood_ontology.contracts import (
    CheckStage,
    CheckStatus,
    ContractError,
    CreateEntity,
    EntityReference,
    LiteralAssertion,
    LiteralValue,
    ObjectAssertion,
    Proposal,
)
from krrood_ontology.preparation import CandidatePreparer
from krrood_ontology.sessions import AcceptanceBlockedError, OntologySystem

NS = "urn:construction:"


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    root = tmp_path_factory.mktemp("construction") / "out"
    OntologyCompiler().compile(
        OntologyInput(Path(__file__).parent / "fixtures/construction.ttl"),
        CompilationOptions(
            root,
            "construction_model",
            structural_only=True,
            binding_policy="local-usage",
            scoped_semantics=True,
        ),
    )
    sys.path.insert(0, str(root))
    module = importlib.import_module("construction_model")
    yield root, module
    sys.path.remove(str(root))


def adapter(root, model, **kwargs):
    preparer = CandidatePreparer.from_manifest(
        model, root / "manifest.json", schema_id="urn:test:schema", base_revision="r0"
    )
    return CandidateConstructor(preparer, root, "construction_model", **kwargs)


@pytest.fixture(scope="module")
def constructor(generated):
    return adapter(generated[0], generated[1].MODEL)


def proposal():
    return Proposal(
        (
            CreateEntity("create", "a", (NS + "Entity", NS + "Special")),
            ObjectAssertion(
                "link",
                EntityReference(local_id="a"),
                NS + "connected",
                EntityReference(iri=NS + "one"),
            ),
        )
    )


def construct(constructor, value=None):
    session = OntologySystem(
        constructor.preparer.context, checker=constructor
    ).begin_candidate(value or proposal())
    return session, constructor.construct(session.snapshot)


def stage(result):
    return next(r for r in result.report.results if r.stage is CheckStage.CONSTRUCTION)


def test_constructs_actual_classes_shared_references_and_native_inference(constructor):
    session, result = construct(constructor)
    assert stage(result).status is CheckStatus.PASS
    artifact = result.artifact
    obj = artifact.object_for_local_id("a")
    assert obj.class_iri == NS + "Special"
    assert obj.python_type == "construction_model.classes.Special"
    assert obj.asserted_types == (NS + "Entity", NS + "Special")
    assert artifact.worker_pid != os.getpid()
    assert artifact.native_symbols_created == len(artifact.objects) == 3
    assert obj.field(NS + "connected").values == (EntityReference(iri=NS + "one"),)
    assert any(
        f.subject_iri == NS + "one" and f.object_iri == obj.iri and f.inferred
        for f in artifact.relations
    )
    assert artifact.roundtrip_equal is True
    assert frozenset(artifact.graph()) == artifact.prepared.triples
    artifact.require_current(session.snapshot)


def test_native_literal_conversion_and_false_zero_distinction(constructor):
    operations = proposal().operations + (
        LiteralAssertion(
            "decimal",
            EntityReference(local_id="a"),
            NS + "amount",
            LiteralValue("001.2300", str(XSD.decimal)),
        ),
        LiteralAssertion(
            "false",
            EntityReference(local_id="a"),
            NS + "values",
            LiteralValue("false", str(XSD.boolean)),
        ),
        LiteralAssertion(
            "zero",
            EntityReference(local_id="a"),
            NS + "values",
            LiteralValue("0", str(XSD.integer)),
        ),
    )
    _, result = construct(constructor, Proposal(operations))
    assert stage(result).status is CheckStatus.PASS
    obj = result.artifact.object_for_local_id("a")
    assert obj.field(NS + "amount").values == (
        LiteralValue("1.2300", str(XSD.decimal)),
    )
    assert set(obj.field(NS + "values").values) == {
        LiteralValue("false", str(XSD.boolean)),
        LiteralValue("0", str(XSD.integer)),
    }


def test_parent_native_graph_and_source_unchanged(constructor, generated):
    native = SymbolGraph()
    nodes, edges = tuple(native.wrapped_instances), tuple(native.relations())
    closure = generated[0] / "source/closure.ttl"
    content = closure.read_bytes()
    construct(constructor)
    assert tuple(native.wrapped_instances) == nodes
    assert tuple(native.relations()) == edges
    assert closure.read_bytes() == content


def test_failed_preflight_never_runs_worker(constructor, monkeypatch):
    def forbidden(*args):
        raise AssertionError("Worker started on failed preflight")

    monkeypatch.setattr(CandidateConstructor, "_run_worker", forbidden)
    _, result = construct(
        constructor, Proposal((CreateEntity("bad", "a", ("urn:unknown",)),))
    )
    assert stage(result).status is CheckStatus.UNKNOWN
    assert result.artifact is None


def test_revision_invalidates_construction_artifact(constructor):
    session, result = construct(constructor)
    session.replace_proposal(Proposal(proposal().operations[:1]), expected_revision=1)
    with pytest.raises(ContractError, match="stale_candidate"):
        result.artifact.require_current(session.snapshot)


def test_session_check_still_blocks_acceptance(constructor):
    session = OntologySystem(
        constructor.preparer.context, checker=constructor
    ).begin_candidate(proposal())
    report = session.check()
    assert (
        next(r for r in report.results if r.stage is CheckStage.CONSTRUCTION).status
        is CheckStatus.PASS
    )
    assert (
        next(r for r in report.results if r.stage is CheckStage.REASONER).status
        is CheckStatus.UNSUPPORTED
    )
    with pytest.raises(AcceptanceBlockedError):
        session.prepare_acceptance()


def test_each_call_constructs_in_a_new_worker(constructor):
    session, first = construct(constructor)
    second = constructor.construct(session.snapshot)
    assert first.artifact.worker_pid != second.artifact.worker_pid
    assert first.artifact.prepared == second.artifact.prepared
    assert (
        first.artifact.native_symbols_created
        == second.artifact.native_symbols_created
        == 3
    )


def test_read_only_views_and_defensive_rdf(constructor):
    _, result = construct(constructor)
    artifact = result.artifact
    with pytest.raises(FrozenInstanceError):
        artifact.objects[0].iri = "urn:changed"
    graph = artifact.graph()
    graph.remove((None, None, None))
    assert frozenset(artifact.graph()) == artifact.prepared.triples
    with pytest.raises(ContractError):
        replace(artifact, objects=list(artifact.objects))


def copied_adapter(generated, tmp_path, fixture=None, **kwargs):
    root = tmp_path / "out"
    shutil.copytree(generated[0], root)
    if fixture:
        with (root / "construction_model/classes.py").open("a") as f:
            f.write((Path(__file__).parent / "fixtures" / fixture).read_text())
    return adapter(root, generated[1].MODEL, **kwargs)


def test_generated_artifact_change_is_rejected(generated, tmp_path):
    constructor = copied_adapter(generated, tmp_path)
    with (constructor.directory / "construction_model/classes.py").open("a") as f:
        f.write("\n# altered\n")
    _, result = construct(constructor)
    assert stage(result).feedback[0].code == "generated_artifact_changed"
    assert result.artifact is None


def test_constructor_failure_after_native_registration_is_isolated(generated, tmp_path):
    constructor = copied_adapter(generated, tmp_path, "construction_failure.py")
    before = tuple(SymbolGraph().wrapped_instances)
    _, result = construct(constructor)
    assert stage(result).status is CheckStatus.ERROR
    assert stage(result).feedback[0].code == "construction_failed"
    assert "private fixture" not in stage(result).feedback[0].message
    assert result.artifact is None
    assert tuple(SymbolGraph().wrapped_instances) == before


def test_real_timeout_reaps_worker_and_preserves_parent(generated, tmp_path):
    constructor = copied_adapter(
        generated, tmp_path, "construction_timeout.py", timeout_seconds=10
    )
    before = tuple(SymbolGraph().wrapped_instances)
    _, result = construct(constructor)
    assert stage(result).feedback[0].code == "construction_timeout"
    marker = constructor.directory / "construction_model/timeout-marker.txt"
    pid = int(marker.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert tuple(SymbolGraph().wrapped_instances) == before


def test_registry_mismatch_is_rejected(generated):
    model = generated[1].MODEL
    prop = model.properties[NS + "amount"]
    changed = replace(
        model, properties={**model.properties, prop.iri: replace(prop, name="wrong")}
    )
    constructor = adapter(generated[0], changed)
    _, result = construct(constructor)
    assert stage(result).feedback[0].code == "model_registry_changed"


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_invalid_timeout_is_rejected(generated, timeout):
    with pytest.raises(ValueError):
        adapter(generated[0], generated[1].MODEL, timeout_seconds=timeout)


def test_nonfinite_projection_is_reported_unsupported(constructor):
    value = Proposal(
        proposal().operations
        + (
            LiteralAssertion(
                "nan",
                EntityReference(local_id="a"),
                NS + "values",
                LiteralValue("NaN", str(XSD.double)),
            ),
        )
    )
    _, result = construct(constructor, value)
    assert result.report.results[0].status is CheckStatus.PASS
    assert stage(result).status is CheckStatus.UNSUPPORTED
    assert result.artifact is None


def test_shared_target_is_one_registry_identity(constructor):
    value = Proposal(
        proposal().operations
        + (
            CreateEntity("create-b", "b", (NS + "Entity",)),
            ObjectAssertion(
                "link-b",
                EntityReference(local_id="b"),
                NS + "connected",
                EntityReference(iri=NS + "one"),
            ),
        )
    )
    _, result = construct(constructor, value)
    artifact = result.artifact
    target = artifact.object_for_iri(NS + "one")
    assert len(artifact.objects) == 4
    for local in ("a", "b"):
        references = artifact.object_for_local_id(local).field(NS + "connected").values
        assert references == (EntityReference(iri=target.iri),)


def test_reflexivity_uses_existing_generated_helper(generated):
    constructor = adapter(generated[0], generated[1].MODEL, seed_reflexivity=True)
    _, result = construct(constructor)
    assert stage(result).status is CheckStatus.PASS
    artifact = result.artifact
    assert artifact.reflexive_seeds == 3
    assert (
        len(
            [
                f
                for f in artifact.relations
                if f.subject_iri == f.object_iri and f.inferred
            ]
        )
        == 3
    )
    assert artifact.roundtrip_equal is True


def test_foreign_worker_artifact_is_rejected(constructor, monkeypatch):
    from krrood_ontology.construction import _dump_worker_record, _WorkerResult

    _, first = construct(constructor)

    def forged_run(args, **kwargs):
        from subprocess import CompletedProcess

        Path(args[-1]).write_bytes(
            _dump_worker_record(
                _WorkerResult(CheckStatus.PASS, artifact=first.artifact)
            )
        )
        return CompletedProcess(args, 0)

    monkeypatch.setattr("krrood_ontology.construction.subprocess.run", forged_run)
    _, result = construct(constructor)
    assert stage(result).status is CheckStatus.ERROR
    assert stage(result).feedback[0].code == "construction_protocol"
    assert result.artifact is None


def test_legacy_phase6_relations_construct_without_regeneration():
    root = Path(__file__).parents[1] / "examples/relations_generated"
    sys.path.insert(0, str(root))
    try:
        module = importlib.import_module("relations_model")
        preparer = CandidatePreparer.from_manifest(
            module.MODEL,
            root / "manifest.json",
            schema_id="urn:legacy:schema",
            base_revision="r0",
        )
        constructor = CandidateConstructor(preparer, root, "relations_model")
        base = "https://example.org/relations#"
        value = Proposal(
            (
                CreateEntity("first", "a", (base + "Person",)),
                CreateEntity("second", "b", (base + "Person",)),
                ObjectAssertion(
                    "link",
                    EntityReference(local_id="a"),
                    base + "knows",
                    EntityReference(local_id="b"),
                ),
            )
        )
        _, result = construct(constructor, value)
        assert stage(result).status is CheckStatus.PASS
        assert len(result.artifact.prepared.additions) == 3
        assert result.artifact.roundtrip_equal is True
        assert len(result.artifact.relations) == 2
        assert sum(f.inferred for f in result.artifact.relations) == 1
    finally:
        sys.path.remove(str(root))
