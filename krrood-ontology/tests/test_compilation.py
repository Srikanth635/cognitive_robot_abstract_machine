"""Exercise generated modules in isolated Python processes."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from krrood_ontology import (
    CompilationOptions,
    OntologyCompiler,
    OntologyInput,
    OntologyTranslationError,
)
from krrood_ontology.generation import ModelGenerator


@pytest.mark.parametrize(
    "fixture,package,script",
    [
        (
            "examples/university.ttl",
            "university_model",
            "examples/verify_university.py",
        ),
        ("tests/fixtures/company.ttl", "company_model", "tests/verify_company.py"),
        (
            "tests/fixtures/class_only.ttl",
            "minimal_model",
            "tests/verify_class_only.py",
        ),
    ],
)
def test_generated_models_run(
    fixture: str, package: str, script: str, tmp_path: Path
) -> None:
    root = Path(__file__).parents[1]
    result = OntologyCompiler().compile(
        OntologyInput(root / fixture),
        CompilationOptions(tmp_path / "generated", package),
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        str(result.output_directory) + os.pathsep + environment.get("PYTHONPATH", "")
    )
    completed = subprocess.run(
        [sys.executable, str(root / script)],
        env=environment,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    manifest = json.loads(result.manifest_file.read_text(encoding="utf-8"))
    assert manifest["phase"] == 10
    assert manifest["artifact"] == "python_dataclass_model"
    assert manifest["individuals_loaded"] is False


def test_unsupported_semantics_leave_no_output(tmp_path: Path) -> None:
    source = Path(__file__).parent / "fixtures" / "restrictions.ttl"
    with pytest.raises(OntologyTranslationError):
        OntologyCompiler().compile(
            OntologyInput(source),
            CompilationOptions(tmp_path / "output", "restricted_model"),
        )
    assert list(tmp_path.iterdir()) == []


def test_explicit_structural_only_records_omissions(tmp_path: Path) -> None:
    source = Path(__file__).parent / "fixtures" / "restrictions.ttl"
    result = OntologyCompiler().compile(
        OntologyInput(source),
        CompilationOptions(
            tmp_path / "output", "restricted_model", structural_only=True
        ),
    )
    manifest = json.loads(result.manifest_file.read_text(encoding="utf-8"))
    assert manifest["structural_only"] is True
    assert any(
        item["code"] == "semantic_feature_deferred"
        for item in manifest["model"]["diagnostics"]
    )


def test_generation_is_reproducible(tmp_path: Path) -> None:
    source = OntologyInput(Path(__file__).parents[1] / "examples" / "university.ttl")
    first = OntologyCompiler().compile(
        source, CompilationOptions(tmp_path / "first", "university_model")
    )
    second = OntologyCompiler().compile(
        source, CompilationOptions(tmp_path / "second", "university_model")
    )
    for path in first.output_directory.rglob("*"):
        if path.is_file():
            assert (
                path.read_bytes()
                == (
                    second.output_directory / path.relative_to(first.output_directory)
                ).read_bytes()
            )


def test_existing_destination_is_preserved(tmp_path: Path) -> None:
    source = OntologyInput(Path(__file__).parents[1] / "examples" / "university.ttl")
    output = tmp_path / "output"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("unchanged", encoding="utf-8")
    with pytest.raises(FileExistsError):
        OntologyCompiler().compile(
            source, CompilationOptions(output, "university_model")
        )
    assert marker.read_text(encoding="utf-8") == "unchanged"


def test_cli_reports_translation_errors_without_tracebacks(tmp_path: Path) -> None:
    source = Path(__file__).parent / "fixtures" / "restrictions.ttl"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "krrood_ontology",
            "compile",
            str(source),
            "--output",
            str(tmp_path / "output"),
            "--package",
            "restricted_model",
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "semantic_feature_deferred" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_generation_failure_leaves_no_partial_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(
        self: ModelGenerator, directory: Path, report: object, model: object
    ) -> None:
        directory.mkdir()
        (directory / "classes.py").touch()
        raise RuntimeError("generation failed")

    monkeypatch.setattr(ModelGenerator, "write_package", fail)
    source = OntologyInput(Path(__file__).parents[1] / "examples" / "university.ttl")
    with pytest.raises(RuntimeError):
        OntologyCompiler().compile(
            source, CompilationOptions(tmp_path / "output", "university_model")
        )
    assert list(tmp_path.iterdir()) == []
