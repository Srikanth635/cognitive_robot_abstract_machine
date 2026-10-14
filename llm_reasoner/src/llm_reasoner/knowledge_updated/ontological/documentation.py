"""
Reading the documentation written beside attributes.

An attribute is documented by the string placed directly below its declaration, the same
convention as a class docstring. Python discards those strings at runtime, so they are
read back from the source file. This keeps documentation in one place: the text a reader
sees in the code is the text the schema and the prompt carry.
"""

from __future__ import annotations

import ast
import inspect
import sys
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Mapping, Optional

from llm_reasoner.exceptions import LLMReasonerError

# %% errors


@dataclass
class DocumentationSourceUnavailable(LLMReasonerError):
    """
    A class's source file cannot be read, so its documentation cannot be known.
    """

    owner: type
    """
    The class whose module has no readable source file.
    """

    def __str__(self) -> str:
        return (
            f"no readable source file for {self.owner.__module__}, so the "
            f"documentation of {self.owner.__qualname__} cannot be read"
        )


# %% parsed documentation


@dataclass(frozen=True)
class ClassDocumentation:
    """
    The documentation written in one class body.
    """

    text: Optional[str] = None
    """
    The class docstring, or ``None`` when the class declares none.
    """

    attributes: Mapping[str, str] = field(default_factory=dict)
    """
    The documentation of each attribute declared in the class body, by attribute name.
    """


class ClassDocumentationCollector(ast.NodeVisitor):
    """
    Collects the documentation of every class in a module, by qualified name.
    """

    def __init__(self) -> None:
        self.classes: dict[str, ClassDocumentation] = {}
        self._scope: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """
        Record a class, then the classes declared inside it.
        """
        self.classes[".".join([*self._scope, node.name])] = ClassDocumentation(
            text=ast.get_docstring(node),
            attributes=self._attribute_documentation(node.body),
        )
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """
        Enter a function, which qualifies the classes declared inside it.
        """
        self._scope.extend([node.name, "<locals>"])
        self.generic_visit(node)
        del self._scope[-2:]

    visit_AsyncFunctionDef = visit_FunctionDef

    def _attribute_documentation(self, body: list[ast.stmt]) -> dict[str, str]:
        """
        Pair each assigned name with the string statement that immediately follows it.
        """
        documented: dict[str, str] = {}
        for statement, following in zip(body, body[1:]):
            name = self._assigned_name(statement)
            text = self._string_of(following)
            if name is not None and text is not None:
                documented[name] = inspect.cleandoc(text)
        return documented

    @staticmethod
    def _assigned_name(statement: ast.stmt) -> Optional[str]:
        """
        Return the name a statement assigns, when it assigns exactly one plain name.
        """
        if isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            return statement.target.id
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            return statement.targets[0].id
        return None

    @staticmethod
    def _string_of(statement: ast.stmt) -> Optional[str]:
        """
        Return the text of a statement that is only a string literal.
        """
        if (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        ):
            return statement.value.value
        return None


@dataclass(frozen=True)
class ModuleDocumentation:
    """
    The documentation of every class in one module.
    """

    classes: Mapping[str, ClassDocumentation]
    """
    Each class's documentation, by qualified name.
    """

    @classmethod
    def from_source(cls, source: str) -> ModuleDocumentation:
        """
        Parse Python source and collect the documentation of its classes.
        """
        collector = ClassDocumentationCollector()
        collector.visit(ast.parse(source))
        return cls(classes=collector.classes)


# %% reading


@dataclass
class DocumentationReader:
    """
    Answers what the source says about a class and its attributes.
    """

    _modules: dict[str, ModuleDocumentation] = field(
        default_factory=dict, init=False, repr=False
    )
    """
    Parsed modules by name, so each source file is read once.
    """

    def read_class(self, owner: type) -> Optional[str]:
        """
        Return the docstring a class declares, or ``None`` when it declares none.

        :raises DocumentationSourceUnavailable: when the class's source cannot be read.
        """
        return self._own_documentation(owner).text

    def read_attribute(self, owner: type, name: str) -> Optional[str]:
        """
        Return an attribute's documentation, or ``None`` when it has none.

        A subclass inherits the documentation of its bases, and text written on the
        subclass takes precedence.

        :param owner: The class whose attribute is read.
        :param name: The attribute name.
        :raises DocumentationSourceUnavailable: when a declaring class's source cannot
            be read.
        """
        for base in owner.__mro__:
            if not self._declares(base, name):
                continue
            documentation = self._own_documentation(base).attributes.get(name)
            if documentation is not None:
                return documentation
        return None

    def find_undocumented_members(self, enumeration: type[Enum]) -> tuple[str, ...]:
        """
        Return the names of the enumeration members that have no documentation.
        """
        return tuple(
            name
            for name in enumeration.__members__
            if self.read_attribute(enumeration, name) is None
        )

    def find_undocumented_fields(self, dataclass_type: type) -> tuple[str, ...]:
        """
        Return the names of the dataclass fields that have no documentation.
        """
        return tuple(
            declared.name
            for declared in fields(dataclass_type)
            if self.read_attribute(dataclass_type, declared.name) is None
        )

    @staticmethod
    def _declares(base: type, name: str) -> bool:
        """
        Return whether a class declares an attribute of this name in its own body.
        """
        return name in inspect.get_annotations(base) or name in vars(base)

    def _own_documentation(self, owner: type) -> ClassDocumentation:
        """
        Return the documentation written in the body of exactly this class.
        """
        module = self._module_documentation(owner)
        return module.classes.get(owner.__qualname__, ClassDocumentation())

    def _module_documentation(self, owner: type) -> ModuleDocumentation:
        """
        Return the parsed documentation of a class's module, reading it on first use.
        """
        if owner.__module__ not in self._modules:
            self._modules[owner.__module__] = ModuleDocumentation.from_source(
                self._source_path(owner).read_text(encoding="utf-8")
            )
        return self._modules[owner.__module__]

    @staticmethod
    def _source_path(owner: type) -> Path:
        """
        Return the source file of a class's module.

        :raises DocumentationSourceUnavailable: when the module is not a Python source
            file.
        """
        module = sys.modules.get(owner.__module__)
        specification = module.__spec__ if module is not None else None
        if (
            specification is None
            or not specification.has_location
            or not specification.origin.endswith(".py")
        ):
            raise DocumentationSourceUnavailable(owner)
        path = Path(specification.origin)
        if not path.is_file():
            raise DocumentationSourceUnavailable(owner)
        return path
