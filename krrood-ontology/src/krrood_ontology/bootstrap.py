"""Orchestrate the first phase and publish a new output directory."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

from krrood_ontology.backend import KrroodBackend
from krrood_ontology.config import BootstrapOptions, OntologyInput
from krrood_ontology.inspection import OntologyInspector, OntologyReport
from krrood_ontology.ontology import OntologyDocument


@dataclass(frozen=True)
class BootstrapResult:
    """Paths and inventory produced by a successful first-phase build."""

    output_directory: Path
    package_directory: Path
    report_file: Path
    report: OntologyReport


@dataclass
class OntologyBootstrap:
    """Build the vocabulary baseline for subsequent class generation."""

    backend: KrroodBackend = field(default_factory=KrroodBackend)
    inspector: OntologyInspector = field(default_factory=OntologyInspector)

    def build(
        self, source: OntologyInput, options: BootstrapOptions
    ) -> BootstrapResult:
        """Create an importable vocabulary and report at a fresh destination."""
        output = options.output_directory
        if output.exists():
            raise FileExistsError(f"Output directory already exists: {output}")
        report = self.inspector.inspect(OntologyDocument.load(source))
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="ontology-", dir=output.parent) as temporary:
            staged = Path(temporary) / "output"
            staged.mkdir()
            self.backend.write_package(staged / options.package_name, report)
            metadata = {
                "phase": 1,
                "artifact": "python_vocabulary",
                "krrood_version": version("krrood"),
                "inference_applied": False,
                "imports_loaded": False,
                "ontology": asdict(report),
            }
            (staged / "report.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            if output.exists():
                raise FileExistsError(f"Output directory already exists: {output}")
            staged.rename(output)
        return BootstrapResult(
            output, output / options.package_name, output / "report.json", report
        )
