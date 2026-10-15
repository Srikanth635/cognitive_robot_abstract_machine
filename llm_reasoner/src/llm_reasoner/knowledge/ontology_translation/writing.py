"""
Writing the Python vocabulary modules from the ontology declarations.

A module is rendered with krrood's Jinja code generator and then formatted the way the
repository's pre-commit hook formats Python: black, then docformatter where its result
survives a second black pass. A regenerated module is therefore byte-for-byte what a
committed one looks like after formatting.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from krrood.code_generation.formatting import run_black_on_file
from krrood.code_generation.generator import CodeGenerator
from typing_extensions import Optional, Sequence

from llm_reasoner.exceptions import LLMReasonerError
from llm_reasoner.knowledge.ontologies import OntologyModuleFile
from llm_reasoner.knowledge.ontologies.terms import (
    ClassExpression,
    ComplementOf,
    EnumeratedClass,
    IntersectionOf,
    NamedClass,
    UnionOf,
)
from llm_reasoner.knowledge.ontology_translation.declarations import (
    ClassDeclaration,
    DescriptorDeclaration,
    FieldDeclaration,
    ModuleDeclaration,
    PythonName,
    ValueKind,
)
from llm_reasoner.knowledge.ontology_translation.reading import OntologyReader

# %% constants

TEMPLATE_DIRECTORY = Path(__file__).parent / "templates"
"""
The directory holding the vocabulary module template.
"""


class TemplateName(StrEnum):
    """
    The templates the writer renders.
    """

    VOCABULARY_MODULE = "vocabulary_module.py.jinja"
    """
    One generated vocabulary module.
    """


DOCFORMATTER_ARGUMENTS: Sequence[str] = (
    "--in-place",
    "--black",
    "--make-summary-multi-line",
    "--pre-summary-newline",
    "--close-quotes-on-newline",
)
"""
The docformatter settings of the repository's ``pyproject.toml``.
"""

DOCFORMATTER_EXIT_CODES = (0, 3)
"""
The exit codes with which docformatter reports success: unchanged and changed.
"""

DESCRIPTION_ENTITY_BASE = "DescriptionEntity"
"""
The base class of a generated class that has no microworld superclass.
"""

PROPERTY_DESCRIPTOR_BASE = "ObjectPropertyDescriptor"
"""
The base class of a descriptor whose property has no microworld super-property.
"""

CONCEPT_REFERENCE_TYPE = "ConceptReference"
"""
The type of a field holding terms of an ontology the vocabulary does not compile.
"""


# %% exceptions


@dataclass
class DocstringFormattingFailed(LLMReasonerError):
    """
    docformatter could not format a generated module.
    """

    output: str
    """
    What docformatter reported.
    """

    def __post_init__(self):
        super().__init__(f"docformatter failed on a generated module: {self.output}")


# %% rendering helpers


def docstring(text: str) -> str:
    """
    :return: The text made safe to place between triple double quotes.
    """
    return text.replace("\\", "\\\\").replace('"""', '\\"\\"\\"').strip()


def python_string(text: str) -> str:
    """
    :return: The text as a Python string literal.
    """
    return repr(str(text))


def expression_source(expression: ClassExpression) -> str:
    """
    :return: Python code constructing the class expression.
    """
    if isinstance(expression, NamedClass):
        return f"NamedClass(iri={expression.iri!r})"
    if isinstance(expression, EnumeratedClass):
        return f"EnumeratedClass(members={python_tuple(expression.members)})"
    if isinstance(expression, ComplementOf):
        return f"ComplementOf(operand={expression_source(expression.operand)})"
    if isinstance(expression, (IntersectionOf, UnionOf)):
        operands = "".join(
            f"{expression_source(operand)}, " for operand in expression.operands
        )
        return f"{type(expression).__name__}(operands=({operands}))"
    arguments = [
        f"property_iri={expression.property_iri!r}",
        f"quantifier=Quantifier.{expression.quantifier.name}",
    ]
    if expression.filler is not None:
        arguments.append(f"filler={expression_source(expression.filler)}")
    if expression.value is not None:
        arguments.append(f"value={expression.value!r}")
    if expression.cardinality is not None:
        arguments.append(f"cardinality={expression.cardinality!r}")
    return f"PropertyRestriction({', '.join(arguments)})"


def expression_tuple(expressions: Sequence[ClassExpression]) -> str:
    """
    :return: Python code constructing a tuple of class expressions.
    """
    return "(" + "".join(f"{expression_source(item)}, " for item in expressions) + ")"


def optional_expression(expression: Optional[ClassExpression]) -> str:
    """
    :return: Python code constructing the class expression, or ``None``.
    """
    return "None" if expression is None else expression_source(expression)


def python_tuple(items: Sequence[str]) -> str:
    """
    :return: The strings as a Python tuple literal.
    """
    return repr(tuple(items))


@dataclass
class ModuleRendering:
    """
    The names one module's template needs, written as that module refers to them.
    """

    module: ModuleDeclaration
    """
    The module being rendered.
    """

    def reference(self, name: PythonName) -> str:
        """
        :return: How the module refers to a class, enumeration or descriptor.
        """
        return name.reference_from(self.module.name)

    def bases(self, cls: ClassDeclaration) -> str:
        """
        :return: The base classes of a generated class, as written in its definition.
        """
        if not cls.bases:
            return DESCRIPTION_ENTITY_BASE
        return ", ".join(self.reference(base) for base in cls.bases)

    def super_descriptors(self, descriptor: DescriptorDeclaration) -> str:
        """
        :return: The base classes of a descriptor class, as written in its definition.
        """
        bases = [self.reference(base) for base in descriptor.super_descriptors] or [
            PROPERTY_DESCRIPTOR_BASE
        ]
        return ", ".join(bases)

    def value_type(self, declared: FieldDeclaration) -> str:
        """
        :return: The type of one value of a field.
        """
        if declared.value_kind == ValueKind.LITERAL:
            return declared.literal_type
        if declared.value_kind == ValueKind.CONCEPT_REFERENCE:
            return CONCEPT_REFERENCE_TYPE
        return self.reference(declared.value_type)

    def annotation(self, declared: FieldDeclaration) -> str:
        """
        :return: The type annotation of a field.
        """
        if declared.functional:
            return f"Optional[{self.value_type(declared)}]"
        return f"List[{self.value_type(declared)}]"

    def default(self, declared: FieldDeclaration) -> str:
        """
        :return: The default argument of a field's ``dataclasses.field`` call.
        """
        if declared.functional:
            return "default=None"
        return "default_factory=list"


# %% writer


@dataclass
class VocabularyWriter:
    """
    Generates the Python vocabulary modules from the ontology.
    """

    reader: OntologyReader
    """
    The reader the module declarations come from.
    """

    generator: CodeGenerator = field(
        default_factory=lambda: CodeGenerator(
            template_directory=str(TEMPLATE_DIRECTORY)
        )
    )
    """
    The Jinja code generator rendering the template.
    """

    def __post_init__(self):
        filters = self.generator.environment.filters
        filters["docstring"] = docstring
        filters["python_string"] = python_string
        filters["python_tuple"] = python_tuple
        filters["expression_tuple"] = expression_tuple
        filters["optional_expression"] = optional_expression

    @classmethod
    def for_knowledge_modules(cls) -> VocabularyWriter:
        """
        :return: A writer over the module file of every family.
        """
        return cls(OntologyReader.for_knowledge_modules())

    def rendered(self, module_file: OntologyModuleFile) -> str:
        """
        :return: The module's source as the template renders it, before formatting.
        """
        module = self.reader.module_declaration(module_file)
        rendering = ModuleRendering(module)
        return self.generator.render(
            TemplateName.VOCABULARY_MODULE.value,
            module=module,
            packages={
                read.vocabulary_module_name: read.vocabulary_package
                for read in self.reader.modules
            },
            reference=rendering.reference,
            bases=rendering.bases,
            super_descriptors=rendering.super_descriptors,
            annotation=rendering.annotation,
            default=rendering.default,
        )

    def source(self, module_file: OntologyModuleFile) -> str:
        """
        :return: The module's formatted source.
        """
        with tempfile.TemporaryDirectory() as scratch_directory:
            scratch = Path(scratch_directory) / module_file.vocabulary_path.name
            scratch.write_text(self.rendered(module_file))
            run_black_on_file(str(scratch))
            black_formatted = scratch.read_text()
            self.format_docstrings(scratch)
            docstrings_formatted = scratch.read_text()
            run_black_on_file(str(scratch))
            if scratch.read_text() != docstrings_formatted:
                return black_formatted
            return docstrings_formatted

    @staticmethod
    def format_docstrings(path: Path) -> None:
        """
        Format a file's docstrings in place with docformatter.
        """
        result = subprocess.run(
            [sys.executable, "-m", "docformatter", *DOCFORMATTER_ARGUMENTS, str(path)],
            capture_output=True,
            text=True,
        )
        if result.returncode not in DOCFORMATTER_EXIT_CODES:
            raise DocstringFormattingFailed(result.stdout + result.stderr)

    def write_all(self) -> None:
        """
        Write every vocabulary module next to the ontology file it is generated from.
        """
        for module_file in self.reader.modules:
            module_file.vocabulary_path.write_text(self.source(module_file))
