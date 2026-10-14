"""Verify local import closure and publication only after isolated checks."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from rdflib import OWL, Literal, URIRef

from krrood_ontology import (
    CompilationOptions,
    LocalImport,
    OntologyCompiler,
    OntologyDocument,
    OntologyInput,
)
from krrood_ontology.generation import ModelGenerator
from krrood_ontology.imports import ImportResolutionError, LocalImportResolver
from krrood_ontology.verification import GeneratedVerificationError, PackageVerifier

ROOT = Path(__file__).parents[1]


def test_missing_import_mapping_does_not_publish(tmp_path):
    with pytest.raises(ImportResolutionError):
        OntologyCompiler().compile(
            OntologyInput(ROOT / "examples/imports/university.ttl"),
            CompilationOptions(tmp_path / "output", "imported_model"),
        )
    assert list(tmp_path.iterdir()) == []


def test_closure_preserves_literal_unicode_and_newlines(tmp_path):
    document = OntologyDocument.load(OntologyInput(ROOT / "examples/university.ttl"))
    alice = URIRef("https://example.org/university#Alice")
    predicate = URIRef("https://example.org/university#name")
    expected = Literal("Alice\u0085User\nRole")
    document.graph.set((alice, predicate, expected))
    source = tmp_path / "source.ttl"
    document.graph.serialize(destination=source, format="turtle")
    result = OntologyCompiler().compile(
        OntologyInput(source), CompilationOptions(tmp_path / "output", "unicode_model")
    )
    closure = OntologyDocument.load(
        OntologyInput(result.output_directory / "source/closure.ttl")
    )
    assert closure.graph.value(alice, predicate) == expected


def test_local_imports_verify_and_load(tmp_path):
    result = OntologyCompiler().compile(
        OntologyInput(ROOT / "examples/imports/university.ttl"),
        CompilationOptions(
            tmp_path / "output",
            "imported_model",
            local_imports=(
                LocalImport(
                    "https://example.org/core",
                    OntologyInput(ROOT / "examples/imports/core.ttl"),
                ),
            ),
        ),
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(result.output_directory)
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(ROOT / "examples/verify_imports.py"),
            str(result.output_directory),
        ],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    manifest = json.loads(result.manifest_file.read_text())
    for record in manifest["source_documents"]:
        assert (result.output_directory / record["snapshot"]).read_bytes() == Path(
            record["source_path"]
        ).read_bytes()
    assert not list(result.output_directory.rglob("*.pyc"))


@pytest.mark.parametrize("case", ["duplicate", "unused"])
def test_import_map_errors(case):
    source = OntologyInput(ROOT / "examples/university.ttl")
    binding = LocalImport("https://example.org/unused", source)
    mappings = (binding, binding) if case == "duplicate" else (binding,)
    with pytest.raises(ImportResolutionError):
        LocalImportResolver().resolve(source, mappings, False)


def test_cyclic_local_imports_terminate(tmp_path):
    root = OntologyDocument.load(
        OntologyInput(ROOT / "examples/imports/university.ttl")
    )
    core = OntologyDocument.load(OntologyInput(ROOT / "examples/imports/core.ttl"))
    core.graph.add(
        (
            URIRef("https://example.org/core"),
            OWL.imports,
            URIRef("https://example.org/university"),
        )
    )
    paths = []
    for name, document in [("university", root), ("core", core)]:
        path = tmp_path / f"{name}.ttl"
        document.graph.serialize(destination=path, format="turtle")
        paths.append(OntologyInput(path))
    result = LocalImportResolver().resolve(
        paths[0],
        (
            LocalImport("https://example.org/core", paths[1]),
            LocalImport("https://example.org/university", paths[0]),
        ),
        False,
    )
    assert len(result.sources) == 2
    assert not list(result.document.graph.triples((None, OWL.imports, None)))


def test_bad_generated_code_fails_before_publication(tmp_path, monkeypatch):
    original = ModelGenerator.write_package

    def broken(self, directory, report, model):
        original(self, directory, report, model)
        (directory / "__init__.py").write_text(
            "raise RuntimeError('broken generated package')\n"
        )

    monkeypatch.setattr(ModelGenerator, "write_package", broken)
    with pytest.raises(GeneratedVerificationError):
        OntologyCompiler().compile(
            OntologyInput(ROOT / "examples/university.ttl"),
            CompilationOptions(tmp_path / "output", "university_model"),
        )
    assert list(tmp_path.iterdir()) == []


def test_verification_timeout_is_actionable(tmp_path, monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("worker", 1)

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(GeneratedVerificationError, match="timed out"):
        PackageVerifier().verify(tmp_path, "model", True)


def test_invalid_individual_assertion_blocks_strict_compile(tmp_path):
    document = OntologyDocument.load(OntologyInput(ROOT / "examples/university.ttl"))
    document.graph.add(
        (
            URIRef("https://example.org/university#Alice"),
            URIRef("https://example.org/university#unknown"),
            URIRef("https://example.org/university#Course42"),
        )
    )
    source = tmp_path / "invalid.ttl"
    document.graph.serialize(destination=source, format="turtle")
    with pytest.raises(GeneratedVerificationError, match="unknown_property"):
        OntologyCompiler().compile(
            OntologyInput(source),
            CompilationOptions(tmp_path / "output", "university_model"),
        )
    assert not (tmp_path / "output").exists()


def test_unresolved_import_omission_is_explicit(tmp_path):
    document = OntologyDocument.load(OntologyInput(ROOT / "examples/imports/core.ttl"))
    document.graph.add(
        (
            URIRef("https://example.org/core"),
            OWL.imports,
            URIRef("https://example.org/unmapped"),
        )
    )
    source = tmp_path / "core.ttl"
    document.graph.serialize(destination=source, format="turtle")
    result = OntologyCompiler().compile(
        OntologyInput(source),
        CompilationOptions(tmp_path / "output", "core_model", structural_only=True),
    )
    manifest = json.loads(result.manifest_file.read_text())
    assert manifest["verification"]["individual_preflight"] == "omitted"
    assert manifest["imports_loaded"] is False
    assert str(OWL.imports) in {
        item["iri"] for item in manifest["model"]["diagnostics"]
    }
