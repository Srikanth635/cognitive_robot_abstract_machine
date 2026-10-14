"""Verify generated-package imports, real KRROOD use, and output handling."""

import importlib
import json
import sys
from pathlib import Path

import pytest
from krrood.code_generation.generator import CodeGenerator

from krrood_ontology.backend import KrroodBackend
from krrood_ontology.bootstrap import OntologyBootstrap
from krrood_ontology.config import BootstrapOptions, OntologyInput


@pytest.fixture
def university_input() -> OntologyInput:
    return OntologyInput(Path(__file__).parents[1] / "examples" / "university.ttl")


def test_actual_krrood_generator_is_used() -> None:
    assert type(KrroodBackend().generator) is CodeGenerator


def test_generated_package_imports(
    university_input: OntologyInput, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = OntologyBootstrap().build(
        university_input, BootstrapOptions(tmp_path / "output", "university_vocabulary")
    )
    monkeypatch.syspath_prepend(str(result.output_directory))
    importlib.invalidate_caches()
    try:
        package = importlib.import_module("university_vocabulary")
        assert package.CLASS_IRIS == result.report.classes
        assert package.OBJECT_PROPERTY_IRIS == result.report.object_properties
        assert package.DATATYPE_PROPERTY_IRIS == result.report.datatype_properties
        assert package.INDIVIDUAL_IRIS == result.report.individuals
        assert package.SOURCE_SHA256 == result.report.source_sha256
    finally:
        sys.modules.pop("university_vocabulary.vocabulary", None)
        sys.modules.pop("university_vocabulary", None)
    metadata = json.loads(result.report_file.read_text(encoding="utf-8"))
    assert metadata["phase"] == 1
    assert metadata["artifact"] == "python_vocabulary"
    assert metadata["inference_applied"] is False
    assert metadata["imports_loaded"] is False


def test_existing_output_is_preserved(
    university_input: OntologyInput, tmp_path: Path
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("existing data", encoding="utf-8")
    with pytest.raises(FileExistsError):
        OntologyBootstrap().build(
            university_input, BootstrapOptions(output, "vocabulary")
        )
    assert sentinel.read_text(encoding="utf-8") == "existing data"
    assert sorted(path.name for path in output.iterdir()) == [sentinel.name]


def test_generation_failure_leaves_no_partial_output(
    university_input: OntologyInput, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_after_partial_write(
        self: KrroodBackend, directory: Path, report: object
    ) -> None:
        directory.mkdir()
        (directory / "__init__.py").touch()
        raise RuntimeError("formatter failure")

    monkeypatch.setattr(KrroodBackend, "write_package", fail_after_partial_write)
    output = tmp_path / "output"
    with pytest.raises(RuntimeError):
        OntologyBootstrap().build(
            university_input, BootstrapOptions(output, "vocabulary")
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("name", ["class", "bad-name", "a.b", ""])
def test_invalid_package_name(name: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        BootstrapOptions(tmp_path / "output", name)
