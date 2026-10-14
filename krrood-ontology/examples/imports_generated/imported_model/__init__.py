"""Generated dataclass models with KRROOD object-property registration."""

from .axioms import RELATION_AXIOMS, RETRACTION_SUPPORTED, UPDATE_POLICY
from .classes import (
    CLASS_BY_IRI,
    PROPERTY_FIELD_NAMES,
    Course,
    Person,
    Student,
)
from .relations import PROPERTY_DESCRIPTORS
from .runtime import MODEL, load_individuals
from .queries import CLASS_EXPRESSIONS, model_queries
from .vocabulary import (
    ANNOTATION_PROPERTY_IRIS,
    CLASS_IRIS,
    DATATYPE_PROPERTY_IRIS,
    INDIVIDUAL_IRIS,
    OBJECT_PROPERTY_IRIS,
    ONTOLOGY_IMPORT_IRIS,
    SOURCE_SHA256,
)

__all__ = [
    "MODEL",
    "load_individuals",
    "CLASS_EXPRESSIONS",
    "model_queries",
    "RELATION_AXIOMS",
    "RETRACTION_SUPPORTED",
    "UPDATE_POLICY",
    "CLASS_BY_IRI",
    "PROPERTY_FIELD_NAMES",
    "PROPERTY_DESCRIPTORS",
    "ANNOTATION_PROPERTY_IRIS",
    "CLASS_IRIS",
    "DATATYPE_PROPERTY_IRIS",
    "INDIVIDUAL_IRIS",
    "OBJECT_PROPERTY_IRIS",
    "ONTOLOGY_IMPORT_IRIS",
    "SOURCE_SHA256",
    "Course",
    "Person",
    "Student",
]
