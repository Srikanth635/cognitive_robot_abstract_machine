"""
Parse Python declarations and lexical references without importing source.
"""

from __future__ import annotations

import ast
import py_compile
import tempfile
from dataclasses import dataclass
from importlib.util import resolve_name
from pathlib import Path

from krrood.code_generation.source_extraction_utils import extract_imports_from

from repository_knowledge.contracts import SourceAnalyzer
from repository_knowledge.model import (
    Diagnostic,
    DiagnosticKind,
    FileExtraction,
    ImportBinding,
    ReferenceRecord,
    RelationshipKind,
    Resolution,
    SourceKind,
    SourceLocation,
    SourceRecord,
    SymbolKind,
    SymbolRecord,
)

# %% isolated syntax validation


@dataclass
class PythonSource(SourceAnalyzer):
    """
    Extract Python source evidence without executing module statements.
    """

    def extract(self, source: SourceRecord) -> FileExtraction:
        if source.source_kind != SourceKind.PYTHON:
            return FileExtraction()
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.py"
            source_path.write_text(source.source_text, encoding="utf-8")
            if py_compile.compile(str(source_path), quiet=2) is None:
                return FileExtraction(
                    diagnostics=[
                        Diagnostic(
                            source.relative_path,
                            DiagnosticKind.SYNTAX,
                            "The configured interpreter cannot compile this source.",
                        )
                    ]
                )
        tree = ast.parse(source.source_text, filename=source.relative_path)
        extraction = FileExtraction()
        module = SymbolRecord(
            source.module_name,
            source.module_name.rsplit(".", 1)[-1],
            source.module_name,
            SymbolKind.MODULE,
            SourceLocation(
                source.relative_path, 1, max(1, len(source.source_text.splitlines()))
            ),
            documentation=ast.get_docstring(tree) or "",
            import_statements=extract_imports_from(ast_tree=tree),
        )
        extraction.symbols.append(module)
        visitor = DeclarationVisitor(source, extraction, [module])
        visitor.visit(tree)
        return extraction


# %% lexical declarations and bindings


@dataclass
class DeclarationVisitor(ast.NodeVisitor):
    """
    Retain lexical identities and unresolved relationships in a syntax tree.
    """

    source: SourceRecord
    """
    The committed file being analysed.
    """

    extraction: FileExtraction
    """
    The declarations and evidence collected so far.
    """

    scopes: list[SymbolRecord]
    """
    The module and currently enclosing declarations.
    """

    is_type_only: bool = False
    """
    Whether the current statement is type-checking-only.
    """

    relationship_kind: RelationshipKind = RelationshipKind.REFERENCES
    """
    The kind of names collected by expression visits.
    """

    def _location(self, node: ast.AST) -> SourceLocation:
        """
        Read the source extent of a declaration or expression.
        """
        if isinstance(node, (ast.stmt, ast.expr, ast.arg, ast.alias)):
            return SourceLocation(
                self.source.relative_path, node.lineno, node.end_lineno or node.lineno
            )
        return self.scopes[-1].location

    def _declare(
        self,
        node: ast.AST,
        symbol_name: str,
        symbol_kind: SymbolKind,
        signature: str = "",
        documentation: str = "",
        bases: list[str] | None = None,
        decorators: list[str] | None = None,
    ) -> SymbolRecord:
        """
        Add a declaration and its precise containment relationship.
        """
        parent = self.scopes[-1]
        symbol = SymbolRecord(
            f"{parent.symbol_identifier}.{symbol_name}",
            symbol_name,
            self.source.module_name,
            symbol_kind,
            self._location(node),
            signature,
            documentation,
            [] if bases is None else bases,
            [] if decorators is None else decorators,
        )
        self.extraction.symbols.append(symbol)
        self.extraction.references.append(
            ReferenceRecord(
                parent.symbol_identifier,
                symbol_name,
                RelationshipKind.DEFINES,
                self._location(node),
                symbol.symbol_identifier,
                Resolution.RESOLVED,
                self.is_type_only,
            )
        )
        self.extraction.bindings.append(
            ImportBinding(
                symbol_name,
                symbol.symbol_identifier,
                parent.symbol_identifier,
                self.is_type_only,
            )
        )
        return symbol

    def _reference(
        self, node: ast.AST, expression: str, relationship_kind: RelationshipKind
    ) -> None:
        """
        Record source evidence whose target is resolved after all files are read.
        """
        self.extraction.references.append(
            ReferenceRecord(
                self.scopes[-1].symbol_identifier,
                expression,
                relationship_kind,
                self._location(node),
                is_type_only=self.is_type_only,
            )
        )

    def _annotation(self, annotation: ast.expr | None) -> None:
        """
        Record names used to describe an annotated type.
        """
        if annotation is None:
            return
        previous_kind = self.relationship_kind
        self.relationship_kind = RelationshipKind.ANNOTATES
        self.visit(annotation)
        self.relationship_kind = previous_kind

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        bases = [ast.unparse(base) for base in node.bases]
        enum_class = any(
            base.rsplit(".", 1)[-1] in ("Enum", "StrEnum", "IntEnum", "IntFlag", "Flag")
            for base in bases
        )
        symbol = self._declare(
            node,
            node.name,
            SymbolKind.ENUM if enum_class else SymbolKind.CLASS,
            f"class {node.name}({', '.join(bases + [ast.unparse(keyword) for keyword in node.keywords])})",
            ast.get_docstring(node) or "",
            bases,
            [ast.unparse(value) for value in node.decorator_list],
        )
        self.scopes.append(symbol)
        for base in node.bases:
            self._reference(base, ast.unparse(base), RelationshipKind.INHERITS)
        for member in node.body:
            self.visit(member)
        self.scopes.pop()

    def _callable(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """
        Collect one callable and the names visible inside its body.
        """
        method = self.scopes[-1].symbol_kind in (SymbolKind.CLASS, SymbolKind.ENUM)
        signature = f"{'async ' if isinstance(node, ast.AsyncFunctionDef) else ''}def {node.name}({ast.unparse(node.args)})"
        if node.returns is not None:
            signature += f" -> {ast.unparse(node.returns)}"
        symbol = self._declare(
            node,
            node.name,
            SymbolKind.METHOD if method else SymbolKind.FUNCTION,
            signature,
            ast.get_docstring(node) or "",
            decorators=[ast.unparse(value) for value in node.decorator_list],
        )
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in node.args.defaults + [
            value for value in node.args.kw_defaults if value is not None
        ]:
            self.visit(default)
        self.scopes.append(symbol)
        parameters = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
        if node.args.vararg is not None:
            parameters.append(node.args.vararg)
        if node.args.kwarg is not None:
            parameters.append(node.args.kwarg)
        for parameter in parameters:
            self._annotation(parameter.annotation)
            self.extraction.bindings.append(
                ImportBinding(parameter.arg, "", symbol.symbol_identifier)
            )
        self._annotation(node.returns)
        for statement in node.body:
            self.visit(statement)
        self.scopes.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._callable(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._callable(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            local_name = alias.asname or alias.name.split(".")[0]
            target_name = alias.name if alias.asname else local_name
            self.extraction.bindings.append(
                ImportBinding(
                    local_name,
                    target_name,
                    self.scopes[-1].symbol_identifier,
                    self.is_type_only,
                )
            )
            self._reference(node, alias.name, RelationshipKind.IMPORTS)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        package = (
            self.source.module_name
            if self.source.is_package
            else self.source.module_name.rpartition(".")[0]
        )
        module_name = node.module or ""
        if node.level:
            if node.level > len(package.split(".")) or not package:
                self.extraction.diagnostics.append(
                    Diagnostic(
                        self.source.relative_path,
                        DiagnosticKind.UNRESOLVED,
                        "A relative import extends beyond the available package context.",
                    )
                )
                return
            module_name = resolve_name("." * node.level + module_name, package)
        for alias in node.names:
            if alias.name == "*":
                self.extraction.diagnostics.append(
                    Diagnostic(
                        self.source.relative_path,
                        DiagnosticKind.UNRESOLVED,
                        "Star-import bindings remain unresolved.",
                    )
                )
                continue
            target_name = ".".join(part for part in (module_name, alias.name) if part)
            self.extraction.bindings.append(
                ImportBinding(
                    alias.asname or alias.name,
                    target_name,
                    self.scopes[-1].symbol_identifier,
                    self.is_type_only,
                )
            )
            self._reference(node, target_name, RelationshipKind.IMPORTS)

    def visit_If(self, node: ast.If) -> None:
        expression = ast.unparse(node.test)
        guarded = expression.rsplit(".", 1)[-1] == "TYPE_CHECKING" or any(
            binding.local_name == expression
            and binding.target_name.rsplit(".", 1)[-1] == "TYPE_CHECKING"
            for binding in self.extraction.bindings
        )
        previous = self.is_type_only
        self.is_type_only = previous or guarded
        self.visit(node.test)
        for statement in node.body:
            self.visit(statement)
        self.is_type_only = previous
        for statement in node.orelse:
            self.visit(statement)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if not isinstance(node.target, ast.Name):
            self._annotation(node.annotation)
            if node.value is not None:
                self.visit(node.value)
            return
        parent = self.scopes[-1]
        if parent.symbol_kind in (SymbolKind.MODULE, SymbolKind.CLASS, SymbolKind.ENUM):
            symbol = self._declare(
                node,
                node.target.id,
                (
                    SymbolKind.VARIABLE
                    if parent.symbol_kind == SymbolKind.MODULE
                    else SymbolKind.FIELD
                ),
                ast.unparse(node),
            )
            self.scopes.append(symbol)
            self._annotation(node.annotation)
            if node.value is not None:
                self.visit(node.value)
            self.scopes.pop()
            return
        self.extraction.bindings.append(
            ImportBinding(node.target.id, "", parent.symbol_identifier)
        )
        self._annotation(node.annotation)
        if node.value is not None:
            self.visit(node.value)

    def visit_Assign(self, node: ast.Assign) -> None:
        parent = self.scopes[-1]
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            if parent.symbol_kind == SymbolKind.ENUM:
                self._declare(
                    node, target.id, SymbolKind.ENUM_MEMBER, ast.unparse(node)
                )
            elif parent.symbol_kind in (SymbolKind.MODULE, SymbolKind.CLASS):
                self._declare(
                    node,
                    target.id,
                    (
                        SymbolKind.VARIABLE
                        if parent.symbol_kind == SymbolKind.MODULE
                        else SymbolKind.FIELD
                    ),
                    ast.unparse(node),
                )
            else:
                self.extraction.bindings.append(
                    ImportBinding(target.id, "", parent.symbol_identifier)
                )
        self.visit(node.value)

    def visit_TypeAlias(self, node: ast.TypeAlias) -> None:
        self._declare(node, node.name.id, SymbolKind.TYPE_ALIAS, ast.unparse(node))
        self._annotation(node.value)

    def visit_Call(self, node: ast.Call) -> None:
        self._reference(node.func, ast.unparse(node.func), RelationshipKind.CALLS)
        for argument in node.args:
            self.visit(argument)
        for keyword in node.keywords:
            self.visit(keyword.value)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.ctx, ast.Load):
            self._reference(node, ast.unparse(node), self.relationship_kind)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self._reference(node, node.id, self.relationship_kind)
