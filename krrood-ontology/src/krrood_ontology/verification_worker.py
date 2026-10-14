"""Execute generated-package checks in the compiler's isolated child process."""

import importlib
import json
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path


def main() -> None:
    directory, package_name, runtime, preflight = sys.argv[1:]
    directory = Path(directory).resolve()
    sys.path.insert(0, runtime)
    sys.path.insert(0, str(directory))
    package = importlib.import_module(package_name)
    expected = (directory / package_name / "__init__.py").resolve()
    if Path(package.__file__).resolve() != expected:
        raise ValueError("Imported package is not the generated artifact")
    manifest = directory / "manifest.json"
    phase = json.loads(manifest.read_text())["phase"] if manifest.exists() else 10
    if phase >= 8 and package.model_validator.model is not package.MODEL:
        raise ValueError("Generated validation registration failed")
    for iri, concrete in package.CLASS_BY_IRI.items():
        if not is_dataclass(concrete) or concrete.__ontology_iri__ != iri:
            raise ValueError("Invalid generated class metadata")
    if set(package.CLASS_EXPRESSIONS) - package.CLASS_BY_IRI.keys():
        raise ValueError("Expression metadata references an unregistered class")
    if package.model_queries.definitions is not package.CLASS_EXPRESSIONS:
        raise ValueError("Generated expression query registration failed")
    for mapping in package.MODEL.bindings or (
        package.MODEL.properties.values()
        if package.MODEL.binding_policy == "explicit"
        else ()
    ):
        concrete = (
            package.CLASS_BY_IRI[mapping.domain_iri]
            if mapping.domain_iri
            else package.MODEL.entity_base
        )
        declared = {item.name: item for item in fields(concrete)}[mapping.name]
        if declared.metadata["ontology_iri"] != mapping.iri:
            raise ValueError("Invalid generated field metadata")
        if mapping.kind == "object" and getattr(concrete, mapping.name) is not (
            package.PROPERTY_BINDINGS[(mapping.iri, mapping.domain_iri)]
            if phase >= 7
            else package.PROPERTY_DESCRIPTORS[mapping.iri]
        ):
            raise ValueError("Object descriptor registration failed")
    if preflight == "true":
        from krrood_ontology.config import OntologyInput
        from krrood_ontology.individuals import IndividualPlanner
        from krrood_ontology.ontology import OntologyDocument

        document = OntologyDocument.load(
            OntologyInput(directory / "source" / "closure.ttl")
        )
        IndividualPlanner(package.MODEL, identity_policy="distinct").plan(document)


if __name__ == "__main__":
    main()
