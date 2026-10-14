"""Extend the existing KRROOD backend with domain-model templates."""

from dataclasses import dataclass, field
from pathlib import Path

from krrood_ontology.backend import KrroodBackend
from krrood_ontology.inspection import OntologyReport
from krrood_ontology.model import DescriptorDefinition, ModelDefinition, PropertyKind


@dataclass
class ModelGenerator:
    """Write structural classes and attach KRROOD object-property descriptors."""

    backend: KrroodBackend = field(default_factory=KrroodBackend)

    def write_package(
        self, directory: Path, report: OntologyReport, model: ModelDefinition
    ) -> None:
        """Generate the vocabulary, classes, relations, and public exports."""
        self.backend.write_package(directory, report)
        context = {
            "model": model,
            "value_types": {
                prop.range_type
                for prop in model.properties
                if prop.kind is PropertyKind.DATATYPE
            },
            "object_properties": tuple(
                prop for prop in model.properties if prop.kind is PropertyKind.OBJECT
            ),
        }
        context["bindings"] = (
            model.property_bindings
            if model.binding_policy == "local-usage"
            else model.properties
        )
        context["descriptors"] = model.descriptors or tuple(
            DescriptorDefinition(prop, f"_Property{index}")
            for index, prop in enumerate(context["object_properties"])
        )
        for filename, template in (
            ("classes.py", "classes.py.jinja"),
            ("schema.py", "schema.py.jinja"),
            ("validation.py", "validation.py.jinja"),
            ("relations.py", "relations.py.jinja"),
            ("axioms.py", "axioms.py.jinja"),
            ("runtime.py", "runtime.py.jinja"),
            ("queries.py", "queries.py.jinja"),
            ("relation_runtime.py", "relation_runtime.py.jinja"),
            ("__init__.py", "model_package.py.jinja"),
        ):
            self.backend.write_file(directory / filename, template, **context)
