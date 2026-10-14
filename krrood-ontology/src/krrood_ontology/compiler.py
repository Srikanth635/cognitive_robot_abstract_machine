"""Compile a supported ontology into a Python domain package."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass, field, replace
from importlib import metadata as package_metadata
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

from rdflib.compare import to_canonical_graph

from krrood_ontology.config import CompilationOptions, OntologyInput
from krrood_ontology.features import FeatureReporter
from krrood_ontology.generation import ModelGenerator
from krrood_ontology.imports import LocalImportResolver
from krrood_ontology.inspection import OntologyInspector, OntologyReport
from krrood_ontology.model import ModelDefinition
from krrood_ontology.structural import StructuralTranslator
from krrood_ontology.translation import OntologyTranslator
from krrood_ontology.verification import PackageVerifier


@dataclass(frozen=True)
class CompilationResult:
    """Generated model paths and the decisions recorded in their manifest."""

    output_directory: Path
    package_directory: Path
    manifest_file: Path
    report: OntologyReport
    model: ModelDefinition


@dataclass
class OntologyCompiler:
    """Connect asserted RDF, model/semantic translation, and KRROOD rendering."""

    translator: OntologyTranslator = field(default_factory=OntologyTranslator)
    generator: ModelGenerator = field(default_factory=ModelGenerator)
    inspector: OntologyInspector = field(default_factory=OntologyInspector)
    import_resolver: LocalImportResolver = field(default_factory=LocalImportResolver)
    verifier: PackageVerifier = field(default_factory=PackageVerifier)

    def compile(
        self, source: OntologyInput, options: CompilationOptions
    ) -> CompilationResult:
        """Publish a fresh package after translation and source checks succeed."""
        output = options.output_directory
        if output.exists():
            raise FileExistsError(f"Output directory already exists: {output}")
        resolved = self.import_resolver.resolve(
            source,
            options.local_imports,
            options.structural_only and not options.scoped_semantics,
        )
        document = resolved.document
        report = self.inspector.inspect(document)
        model = (
            StructuralTranslator().translate(document, report)
            if options.binding_policy == "local-usage"
            else self.translator.translate(document, report, options.structural_only)
        )
        if options.scoped_semantics:
            from krrood_ontology.scoped import ScopedSemantics

            model = ScopedSemantics().translate(document, model)
        if resolved.imported_iris:
            report = replace(report, imports=resolved.imported_iris)
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="ontology-", dir=output.parent) as temporary:
            staged = Path(temporary) / "output"
            staged.mkdir()
            self.generator.write_package(staged / options.package_name, report, model)
            source_directory = staged / "source"
            (source_directory / "documents").mkdir(parents=True)
            provenance = []
            for index, original in enumerate(resolved.sources):
                snapshot = (
                    source_directory
                    / "documents"
                    / f"{index:03d}.{original.format.value}"
                )
                content = original.source.path.read_bytes()
                if hashlib.sha256(content).hexdigest() != original.sha256:
                    raise ValueError("Source changed during compilation")
                snapshot.write_bytes(content)
                provenance.append(
                    {
                        "source_path": str(original.source.path),
                        "source_base_iri": original.source.path.as_uri(),
                        "sha256": original.sha256,
                        "format": original.format.value,
                        "snapshot": str(snapshot.relative_to(staged)),
                    }
                )
            canonical = (
                document.graph
                if model.binding_policy == "local-usage"
                else to_canonical_graph(document.graph)
            )
            closure = (
                "\n".join(
                    sorted(
                        line
                        for line in canonical.serialize(format="nt").split("\n")
                        if line
                    )
                )
                + "\n"
            )
            (source_directory / "closure.ttl").write_text(closure, encoding="utf-8")
            coverage = FeatureReporter().report(document, model)
            (staged / "features.json").write_text(
                json.dumps(coverage, indent=2) + "\n", encoding="utf-8"
            )
            verification = self.verifier.verify(
                staged, options.package_name, not options.structural_only
            )
            metadata = {
                "phase": 10,
                "artifact": "python_dataclass_model",
                "krrood_version": version("krrood"),
                "structural_only": options.structural_only,
                "binding_policy": model.binding_policy,
                "scoped_semantics": options.scoped_semantics,
                "reflexivity_policy": "explicit_scoped_initialization",
                "relation_provenance": "native_first_insertion_flag",
                "inference_applied": False,
                "relation_implications": not options.structural_only
                or options.scoped_semantics,
                "update_policy": "additions_after_construction",
                "retraction_supported": False,
                "imports_loaded": bool(resolved.imported_iris),
                "import_policy": "explicit_local_files_no_network",
                "resolved_import_iris": resolved.imported_iris,
                "source_documents": provenance,
                "runtime_ontology": "source/closure.ttl",
                "closure_sha256": hashlib.sha256(closure.encode("utf-8")).hexdigest(),
                "blank_node_policy": (
                    "source_graph_local_ids"
                    if model.binding_policy == "local-usage"
                    else "rdflib_canonical"
                ),
                "feature_report": "features.json",
                "verification": verification,
                "environment": {
                    "python": platform.python_version(),
                    "packages": {
                        name: package_metadata.version(name)
                        for name in (
                            "krrood-ontology",
                            "krrood",
                            "rdflib",
                            "jinja2",
                            "black",
                        )
                    },
                },
                "individuals_loaded": False,
                "individual_loader_available": True,
                "individual_type_policy": "explicit_inheritance_chain",
                "identity_policy": "declared_distinctness_no_merging",
                "rdf_export_available": True,
                "export_default": "source_snapshot",
                "export_current_policy": "mapped_fields_native_first_insertion_flags",
                "runtime_type_validation": False,
                "class_expression_queries": not options.structural_only
                or options.scoped_semantics,
                "expression_query_policy": "positive_known_objects_explicit_registry",
                "automatic_reclassification": False,
                "scoped_validation_available": True,
                "validation_default": "incomplete_scope_owl_identity",
                "validation_profile": (
                    "mapped_fields_positive_definitions"
                    if options.scoped_semantics
                    else "mapped_fields_explicit_structural_types"
                ),
                "ontology": asdict(report),
                "model": asdict(model),
            }
            (staged / "manifest.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            if output.exists():
                raise FileExistsError(f"Output directory already exists: {output}")
            staged.rename(output)
        return CompilationResult(
            output,
            output / options.package_name,
            output / "manifest.json",
            report,
            model,
        )
