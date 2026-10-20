"""
Translate the shared source graph into CRAMERA architecture entities.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath

from cramera.knowledge.architecture_entities import Package, PythonClass
from cramera.knowledge.architecture_scan import (
    ArchitectureScan,
    ArchitectureScanner,
    PackageDependency,
)
from cramera.knowledge.architecture_source import (
    ArchitectureSource,
    ArchitectureSetting,
)
from cramera.logging_setup import get_logger
from repository_knowledge.adapters.git_source import GitSource
from repository_knowledge.adapters.sqlite_index import SQLiteIndex
from repository_knowledge.adapters.working_tree import WorkingTreeSource
from repository_knowledge.model import (
    IndexData,
    IndexSelection,
    SelectionKind,
    SourceKind,
    SourceRecord,
    SourceScope,
    SymbolKind,
)
from repository_knowledge.operations.freshness import FreshnessCheck
from repository_knowledge.operations.queries import IndexContents

# %% shared graph ownership


@dataclass
class IndexedArchitecture(ArchitectureSource):
    """
    Observe shared source freshness before supplying the architecture view.
    """

    database_path: Path
    """The independently built repository-knowledge cache."""
    revision: str = ArchitectureSetting.CHECKOUT
    """
    The source revision the viewer expects.
    """

    def load(self) -> ArchitectureScan:
        if not self.database_path.is_file():
            return ArchitectureScanner(self.root).scan()
        index = SQLiteIndex(self.database_path).read(
            IndexSelection(SelectionKind.TASK_CONTEXT)
        )
        source = (
            WorkingTreeSource(Path(self.root), index.manifest.included_untracked_paths)
            if index.manifest.source_scope == SourceScope.WORKING_TREE
            else GitSource(Path(self.root))
        )
        freshness = FreshnessCheck(IndexContents(index), source)
        report = freshness.check(self.revision)
        if not report.matches_requested_revision or not report.matches_requested_source:
            get_logger(__name__).warning(
                "Shared architecture is stale; scanning current source without its cache."
            )
            return ArchitectureScanner(self.root).scan()
        result = ArchitectureProjection().translate(index)
        if index.manifest.source_scope == SourceScope.WORKING_TREE:
            freshness.check(self.revision).require_current()
        return result


# %% legacy view translation


class ImportForm(StrEnum):
    """
    Statement forms retained by the shared import extractor.
    """

    IMPORT = "import"
    FROM = "from"


class ArchitectureGroup(StrEnum):
    """
    The viewer's synthetic package for loose repository scripts.
    """

    ROOT = "root"


@dataclass
class ArchitectureProjection:
    """
    Preserve the existing view's naming and grouping of source declarations.
    """

    class_description_characters: int = 140
    """
    The existing view's maximum class-summary length.
    """

    def translate(self, index: IndexData) -> ArchitectureScan:
        """
        Adapt source records without parsing or importing their Python modules.
        """
        sources = self._select_sources(index)
        packages = [
            ArchitectureGroup.ROOT,
            *sorted(
                {self._package(source.relative_path) for source in sources}
                - {ArchitectureGroup.ROOT}
            ),
        ]
        module_symbols = {
            symbol.location.relative_path
            for symbol in index.symbols
            if symbol.symbol_kind == SymbolKind.MODULE
        }
        modules = {
            source.relative_path: source
            for source in sources
            if source.source_kind == SourceKind.PYTHON
            and source.relative_path in module_symbols
        }
        classes = self._classes(index, modules)
        module_counts = Counter(self._package(path) for path in modules)
        class_counts = Counter(declaration.package for declaration in classes)
        descriptions = self._descriptions(sources)
        imports = self._imports(index, modules)
        return ArchitectureScan(
            [
                Package(
                    name=package,
                    description=ArchitectureScanner.PACKAGE_DESCRIPTIONS.get(package)
                    or descriptions.get(package, ""),
                    module_count=module_counts[package],
                    class_count=class_counts[package],
                )
                for package in packages
            ],
            classes,
            [
                PackageDependency(source, target)
                for source in sorted(imports)
                for target in sorted(imports[source])
                if target in packages and source != target
            ],
        )

    def _select_sources(self, index: IndexData) -> list[SourceRecord]:
        """
        Keep the original viewer's handwritten directory and file scope.
        """
        selected = []
        for source in index.sources:
            path = PurePosixPath(source.relative_path)
            if any(
                part.startswith(".")
                or part in ArchitectureScanner.SKIPPED_DIRECTORIES
                or "egg-info" in part
                for part in path.parts[:-1]
            ):
                continue
            if path.name == ArchitectureScanner.GENERATED_MAPPING_FILE:
                continue
            selected.append(source)
        return selected

    @staticmethod
    def _package(relative_path: str) -> str:
        """
        Map loose scripts and member paths to the viewer's package groups.
        """
        path = PurePosixPath(relative_path)
        return path.parts[0] if len(path.parts) > 1 else ArchitectureGroup.ROOT

    def _imports(
        self, index: IndexData, modules: dict[str, SourceRecord]
    ) -> dict[str, set[str]]:
        """
        Translate normalized imports, including unresolved star-import names.
        """
        imports: dict[str, set[str]] = {}
        for symbol in index.symbols:
            if (
                symbol.symbol_kind != SymbolKind.MODULE
                or symbol.location.relative_path not in modules
            ):
                continue
            package = self._package(symbol.location.relative_path)
            for statement in symbol.import_statements:
                words = statement.split()
                if words[0] == ImportForm.FROM:
                    roots = [words[1].split(".")[0]]
                elif words[0] == ImportForm.IMPORT:
                    roots = [
                        item.strip().split()[0].split(".")[0]
                        for item in statement[len(ImportForm.IMPORT) :].split(",")
                    ]
                else:
                    continue
                imports.setdefault(package, set()).update(roots)
        return imports

    def _classes(
        self, index: IndexData, modules: dict[str, SourceRecord]
    ) -> list[PythonClass]:
        """
        Translate nested and enum declarations with direct method counts.
        """
        methods = Counter(
            symbol.symbol_identifier.rpartition(".")[0]
            for symbol in index.symbols
            if symbol.symbol_kind == SymbolKind.METHOD
        )
        result = []
        for symbol in index.symbols:
            if (
                symbol.location.relative_path not in modules
                or symbol.symbol_kind not in (SymbolKind.CLASS, SymbolKind.ENUM)
            ):
                continue
            module = symbol.location.relative_path[:-3].replace("/", ".")
            package = self._package(symbol.location.relative_path)
            bases = tuple(
                (
                    expression.split(".")[-1]
                    if all(part.isidentifier() for part in expression.split("."))
                    else "?"
                )
                for expression in symbol.base_expressions
            )
            result.append(
                PythonClass(
                    name=symbol.symbol_name,
                    package=package,
                    subpackage=ArchitectureScanner._subpackage_of(package, module),
                    module=module,
                    bases=bases,
                    methods=methods[symbol.symbol_identifier],
                    docstring_summary=symbol.documentation.strip().split("\n")[0][
                        : self.class_description_characters
                    ],
                )
            )
        return result

    def _descriptions(self, sources: list[SourceRecord]) -> dict[str, str]:
        """
        Preserve package README summaries from the captured source snapshot.
        """
        descriptions = {}
        for source in sources:
            path = PurePosixPath(source.relative_path)
            if path.name not in ("README.md", "readme.md") or len(path.parts) > 2:
                continue
            lines = (
                line.strip().lstrip("#").strip()
                for line in source.source_text.splitlines()
            )
            first_line = next((line for line in lines if line), "")
            descriptions.setdefault(
                self._package(source.relative_path),
                first_line[: ArchitectureScanner.DESCRIPTION_LENGTH_LIMIT],
            )
        return descriptions
