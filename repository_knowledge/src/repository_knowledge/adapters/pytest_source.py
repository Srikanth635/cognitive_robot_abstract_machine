"""
Recognize pytest declarations without importing tests or plugins.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath

from repository_knowledge.assistance import FixtureRecord, TestCatalog, TestRecord
from repository_knowledge.contracts import TestAnalyzer
from repository_knowledge.model import (
    DiagnosticKind,
    IndexData,
    SourceKind,
    SymbolKind,
    SymbolRecord,
)

# %% pytest source vocabulary


class PytestName(StrEnum):
    """
    Names interpreted by the static pytest declaration reader.
    """

    MODULE = "pytest"
    FIXTURE = "pytest.fixture"
    PARAMETRIZE = "pytest.mark.parametrize"
    USE_FIXTURES = "pytest.mark.usefixtures"
    SCOPE = "scope"
    AUTO_USE = "autouse"
    NAME = "name"
    INDIRECT = "indirect"
    ARGUMENT_NAMES = "argnames"
    FUNCTION = "function"
    UNKNOWN = "unknown"
    SELF = "self"
    CLASS = "cls"
    DYNAMIC_FIXTURE = "getfixturevalue"


@dataclass
class MarkerRequests:
    """
    Fixture requests and uncertainties declared by pytest markers.
    """

    fixture_names: list[str]
    """
    Statically requested fixture names.
    """

    uncertainties: list[str]
    """
    Marker expressions needing runtime interpretation.
    """


@dataclass
class FunctionDeclaration:
    """
    Interpret only literal pytest metadata attached to a function.
    """

    node: ast.FunctionDef | ast.AsyncFunctionDef
    """
    The source function being examined.
    """

    aliases: dict[str, str]
    """
    Top-level pytest import aliases in its module.
    """

    def decorator_name(self, node: ast.expr) -> str:
        """
        Resolve a top-level pytest import alias in a decorator expression.
        """
        expression = ast.unparse(node.func if isinstance(node, ast.Call) else node)
        first, separator, rest = expression.partition(".")
        return self.aliases.get(first, first) + separator + rest

    def fixture_arguments(self) -> list[str]:
        """
        Return named fixture requests excluding bound-method receivers.
        """
        return [
            argument.arg
            for argument in [
                *self.node.args.posonlyargs,
                *self.node.args.args,
                *self.node.args.kwonlyargs,
            ]
            if argument.arg not in (PytestName.SELF, PytestName.CLASS)
        ]

    def inspect_markers(self) -> MarkerRequests:
        """
        Return fixture requests and visible dynamic marker limitations.
        """
        requests = self.fixture_arguments()
        uncertainties = []
        for decorator in self.node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            decorator_name = self.decorator_name(decorator)
            if decorator_name == PytestName.USE_FIXTURES:
                for argument in decorator.args:
                    if isinstance(argument, ast.Constant) and isinstance(
                        argument.value, str
                    ):
                        requests.append(argument.value)
                    else:
                        uncertainties.append("Dynamic usefixtures declaration.")
            if decorator_name != PytestName.PARAMETRIZE:
                continue
            keywords = {keyword.arg: keyword.value for keyword in decorator.keywords}
            names = (
                decorator.args[0]
                if decorator.args
                else keywords.get(PytestName.ARGUMENT_NAMES)
            )
            if isinstance(names, ast.Constant) and isinstance(names.value, str):
                parameters = [name.strip() for name in names.value.split(",")]
            elif isinstance(names, (ast.List, ast.Tuple)) and all(
                isinstance(item, ast.Constant) and isinstance(item.value, str)
                for item in names.elts
            ):
                parameters = [item.value for item in names.elts]
            else:
                uncertainties.append("Dynamic parametrization argument names.")
                continue
            indirect = keywords.get(PytestName.INDIRECT)
            if (
                indirect is None
                or isinstance(indirect, ast.Constant)
                and indirect.value is False
            ):
                requests = [name for name in requests if name not in parameters]
            else:
                uncertainties.append(
                    "Indirect parametrization may alter fixture dependencies."
                )
        if any(
            isinstance(node, ast.Attribute) and node.attr == PytestName.DYNAMIC_FIXTURE
            for node in ast.walk(self.node)
        ):
            uncertainties.append("Runtime fixture selection is not resolved.")
        return MarkerRequests(sorted(set(requests)), sorted(set(uncertainties)))

    def read_fixture(self, symbol: SymbolRecord, decorator: ast.expr) -> FixtureRecord:
        """
        Read literal fixture registration metadata and retain dynamic limitations.
        """
        keywords = (
            {keyword.arg: keyword.value for keyword in decorator.keywords}
            if isinstance(decorator, ast.Call)
            else {}
        )
        scope_node = keywords.get(PytestName.SCOPE)
        name_node = keywords.get(PytestName.NAME)
        auto_node = keywords.get(PytestName.AUTO_USE)
        scope = (
            scope_node.value
            if isinstance(scope_node, ast.Constant)
            and isinstance(scope_node.value, str)
            else PytestName.UNKNOWN if scope_node is not None else PytestName.FUNCTION
        )
        fixture_name = (
            name_node.value
            if isinstance(name_node, ast.Constant) and isinstance(name_node.value, str)
            else symbol.symbol_name
        )
        uncertainties = []
        if (
            scope == PytestName.UNKNOWN
            or name_node is not None
            and not isinstance(name_node, ast.Constant)
            or auto_node is not None
            and not isinstance(auto_node, ast.Constant)
        ):
            uncertainties.append("Dynamic fixture name, scope or autouse declaration.")
        return FixtureRecord(
            symbol.symbol_identifier,
            fixture_name,
            str(scope),
            symbol.location,
            self.fixture_arguments(),
            isinstance(auto_node, ast.Constant) and auto_node.value is True,
            (
                symbol.symbol_identifier.rpartition(".")[0]
                if symbol.symbol_kind == SymbolKind.METHOD
                else ""
            ),
            uncertainties,
        )


# %% captured-source discovery


@dataclass
class PytestSource(TestAnalyzer):
    """
    Recognize default pytest file and declaration patterns statically.
    """

    def discover(self, index: IndexData) -> TestCatalog:
        catalog = TestCatalog()
        invalid_paths = {
            item.relative_path
            for item in index.diagnostics
            if item.diagnostic_kind in (DiagnosticKind.SYNTAX, DiagnosticKind.ENCODING)
        }
        symbols_by_path = {}
        for symbol in index.symbols:
            if symbol.symbol_kind in (SymbolKind.FUNCTION, SymbolKind.METHOD):
                symbols_by_path.setdefault(symbol.location.relative_path, []).append(
                    symbol
                )
        for source in index.sources:
            if (
                source.source_kind != SourceKind.PYTHON
                or source.relative_path in invalid_paths
            ):
                continue
            tree = ast.parse(source.source_text)
            aliases = {}
            for node in tree.body:
                if isinstance(node, ast.Import):
                    aliases.update(
                        {
                            item.asname or item.name: item.name
                            for item in node.names
                            if item.name == PytestName.MODULE
                        }
                    )
                elif (
                    isinstance(node, ast.ImportFrom)
                    and node.module == PytestName.MODULE
                ):
                    aliases.update(
                        {
                            item.asname or item.name: f"{node.module}.{item.name}"
                            for item in node.names
                        }
                    )
            declarations = [
                node
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            filename = PurePosixPath(source.relative_path).name
            test_file = filename.startswith("test_") or filename.endswith("_test.py")
            for symbol in symbols_by_path.get(source.relative_path, []):
                nodes = [
                    node
                    for node in declarations
                    if node.name == symbol.symbol_name
                    and symbol.location.start_line
                    <= node.lineno
                    <= symbol.location.end_line
                ]
                if len(nodes) != 1:
                    continue
                declaration = FunctionDeclaration(nodes[0], aliases)
                decorators = [
                    node
                    for node in declaration.node.decorator_list
                    if declaration.decorator_name(node) == PytestName.FIXTURE
                ]
                owner = (
                    symbol.symbol_identifier.rpartition(".")[0]
                    if symbol.symbol_kind == SymbolKind.METHOD
                    else ""
                )
                if decorators:
                    catalog.fixtures.append(
                        declaration.read_fixture(symbol, decorators[0])
                    )
                elif (
                    test_file
                    and symbol.symbol_name.startswith("test_")
                    and (not owner or owner.rsplit(".", 1)[-1].startswith("Test"))
                ):
                    markers = declaration.inspect_markers()
                    catalog.tests.append(
                        TestRecord(
                            symbol.symbol_identifier,
                            symbol.symbol_name,
                            symbol.location,
                            markers.fixture_names,
                            markers.uncertainties,
                        )
                    )
        catalog.tests.sort(
            key=lambda item: (item.location.relative_path, item.symbol_identifier)
        )
        catalog.fixtures.sort(
            key=lambda item: (
                item.fixture_name,
                item.location.relative_path,
                item.symbol_identifier,
            )
        )
        return catalog
