"""Command-line access to inspection, vocabulary, and structural compilation."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from krrood_ontology.bootstrap import OntologyBootstrap
from krrood_ontology.compiler import OntologyCompiler
from krrood_ontology.config import (
    BootstrapOptions,
    CompilationOptions,
    LocalImport,
    OntologyInput,
    RDFFormat,
)
from krrood_ontology.imports import ImportResolutionError
from krrood_ontology.inspection import OntologyInspector
from krrood_ontology.model import OntologyTranslationError
from krrood_ontology.ontology import OntologyDocument
from krrood_ontology.verification import GeneratedVerificationError


def main() -> None:
    """Run an ontology operation and report unsupported mappings."""
    parser = argparse.ArgumentParser(prog="krrood-ontology")
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {version('krrood-ontology')}"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("inspect", "bootstrap", "compile"):
        operation = commands.add_parser(command)
        operation.add_argument("ontology", type=Path)
        operation.add_argument("--format", choices=[item.value for item in RDFFormat])
        if command != "inspect":
            operation.add_argument("--output", type=Path, required=True)
            operation.add_argument("--package", required=True)
        if command == "compile":
            operation.add_argument(
                "--import",
                dest="local_imports",
                action="append",
                default=[],
                metavar="IRI=PATH",
                help="Resolve an ontology import using a supplied local file; repeat as needed",
            )
            operation.add_argument(
                "--binding-policy",
                choices=("explicit", "local-usage"),
                default="explicit",
                help="local-usage permits absent signatures and multiple parents; requires --structural-only",
            )
            operation.add_argument(
                "--scoped-semantics",
                action="store_true",
                help="Enable scoped positive definitions and native relation additions; requires local-usage",
            )
            operation.add_argument(
                "--structural-only",
                action="store_true",
                help="Record unsupported semantic constructs without implementing them",
            )
    arguments = parser.parse_args()
    source = OntologyInput(
        arguments.ontology,
        RDFFormat(arguments.format) if arguments.format else None,
    )
    if arguments.command == "inspect":
        report = OntologyInspector().inspect(OntologyDocument.load(source))
        print(json.dumps(asdict(report), indent=2, ensure_ascii=False))
        return
    if arguments.command == "compile":
        if arguments.binding_policy == "local-usage" and not arguments.structural_only:
            parser.error("--binding-policy local-usage requires --structural-only")
        if arguments.scoped_semantics and arguments.binding_policy != "local-usage":
            parser.error("--scoped-semantics requires --binding-policy local-usage")
        imports = []
        for specification in arguments.local_imports:
            iri, separator, filename = specification.partition("=")
            if not separator or not iri or not filename:
                parser.error("--import requires IRI=PATH")
            imports.append(LocalImport(iri, OntologyInput(Path(filename))))
        try:
            result = OntologyCompiler().compile(
                source,
                CompilationOptions(
                    arguments.output,
                    arguments.package,
                    arguments.structural_only,
                    tuple(imports),
                    arguments.binding_policy,
                    arguments.scoped_semantics,
                ),
            )
        except (
            OntologyTranslationError,
            ImportResolutionError,
            GeneratedVerificationError,
        ) as error:
            parser.exit(2, f"Compilation failed:\n{error}\n")
        for item in result.model.diagnostics:
            print(
                f"warning: {item.code}: {item.message}"
                + (f" [{item.iri}]" if item.iri else ""),
                file=sys.stderr,
            )
        print(result.package_directory)
        print(result.manifest_file)
        return
    result = OntologyBootstrap().build(
        source, BootstrapOptions(arguments.output, arguments.package)
    )
    print(result.package_directory)
    print(result.report_file)
