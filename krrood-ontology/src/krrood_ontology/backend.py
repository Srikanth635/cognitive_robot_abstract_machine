"""Direct integration with KRROOD's source generator and formatter."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

from krrood.code_generation.formatting import run_black_on_file
from krrood.code_generation.generator import CodeGenerator

from krrood_ontology.inspection import OntologyReport


@dataclass
class KrroodBackend:
    """Generate Python files using the installed KRROOD implementation."""

    generator: CodeGenerator = field(init=False)

    def __post_init__(self) -> None:
        self.generator = CodeGenerator(
            template_directory=str(files("krrood_ontology") / "templates")
        )
        self.generator.environment.filters["python_literal"] = repr

    def write_package(self, directory: Path, report: OntologyReport) -> None:
        """Render, format, and syntax-check a vocabulary package."""
        directory.mkdir()
        for filename, template in (
            ("__init__.py", "package.py.jinja"),
            ("vocabulary.py", "vocabulary.py.jinja"),
        ):
            self.write_file(directory / filename, template, report=report)

    def write_file(self, destination: Path, template: str, **context: object) -> None:
        """Render, format, and syntax-check one Python source file."""
        destination.write_text(
            self.generator.render(template, **context), encoding="utf-8"
        )
        run_black_on_file(str(destination))
        ast.parse(destination.read_text(encoding="utf-8"), filename=str(destination))
