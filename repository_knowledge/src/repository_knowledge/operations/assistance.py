"""Prepare task evidence from a pinned graph and static test declarations."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import PurePosixPath

from repository_knowledge.assistance import (
    ContextItem,
    ContextOmission,
    ContextRole,
    DiscoveryReport,
    FixtureRecord,
    OmissionReason,
    SourceCitation,
    TaskContext,
    TaskProfile,
    TestCatalog,
    TestRecord,
)
from repository_knowledge.concepts import EvidenceState, KnowledgeBundle
from repository_knowledge.exceptions import InvalidQuery
from repository_knowledge.model import (
    IndexData,
    ReferenceRecord,
    RelationshipKind,
    Resolution,
    SourceKind,
    SourceLocation,
    SymbolRecord,
)
from repository_knowledge.operations.knowledge import KnowledgeValidation

# %% fixture visibility


@dataclass(frozen=True)
class FixtureRequest:
    """A requested fixture name and the overriding declaration requesting it."""

    fixture_name: str
    """The fixture name to resolve."""
    requester_identifier: str = ""
    """The same-name override whose ancestor must supply this request."""


@dataclass
class FixtureClosure:
    """The visible dependency closure for one test declaration."""

    fixtures: list[FixtureRecord] = field(default_factory=list)
    """Uniquely resolved fixture declarations."""
    unresolved: list[str] = field(default_factory=list)
    """Requests that cannot be uniquely established statically."""


@dataclass
class FixtureResolution:
    """Resolve module, class and ancestor-conftest fixture visibility."""

    catalog: TestCatalog
    """The captured fixture declarations."""

    def visible_fixtures(self, test: TestRecord) -> list[FixtureRecord]:
        """Return declarations visible from the selected test's lexical scope."""
        parent = PurePosixPath(test.location.relative_path).parent
        return [
            fixture
            for fixture in self.catalog.fixtures
            if (
                fixture.location.relative_path == test.location.relative_path
                or PurePosixPath(fixture.location.relative_path).name == "conftest.py"
                and PurePosixPath(fixture.location.relative_path).parent
                in (parent, *parent.parents)
            )
            and (
                not fixture.owner_identifier
                or test.symbol_identifier.startswith(fixture.owner_identifier + ".")
            )
        ]

    def resolve(self, test: TestRecord) -> FixtureClosure:
        """Resolve dependency requests in test scope, retaining ambiguities."""
        visible = self.visible_fixtures(test)
        requests = [FixtureRequest(name) for name in test.fixture_names]
        requests.extend(
            FixtureRequest(item.fixture_name) for item in visible if item.autouse
        )
        selected = {}
        unresolved = set()
        visited = set()
        while requests:
            request = requests.pop(0)
            if request in visited:
                continue
            visited.add(request)
            candidates = [
                item for item in visible if item.fixture_name == request.fixture_name
            ]
            candidates.sort(
                key=lambda item: (
                    bool(item.owner_identifier),
                    item.location.relative_path == test.location.relative_path,
                    len(PurePosixPath(item.location.relative_path).parts),
                ),
                reverse=True,
            )
            if request.requester_identifier:
                requester_positions = [
                    position
                    for position, item in enumerate(candidates)
                    if item.symbol_identifier == request.requester_identifier
                ]
                if requester_positions:
                    candidates = candidates[requester_positions[0] + 1 :]
            if not candidates:
                unresolved.add(f"{test.symbol_identifier}: {request.fixture_name}")
                continue
            best = candidates[0]
            same_priority = [
                item
                for item in candidates
                if item.owner_identifier == best.owner_identifier
                and item.location.relative_path == best.location.relative_path
            ]
            if len(same_priority) != 1:
                unresolved.add(f"{test.symbol_identifier}: {request.fixture_name}")
                continue
            selected[best.symbol_identifier] = best
            requests.extend(
                FixtureRequest(
                    dependency,
                    (
                        best.symbol_identifier
                        if dependency == request.fixture_name
                        else ""
                    ),
                )
                for dependency in best.dependencies
            )
        return FixtureClosure(
            sorted(
                selected.values(),
                key=lambda item: (item.fixture_name, item.location.relative_path),
            ),
            sorted(unresolved),
        )


# %% source-backed task queries


@dataclass
class TaskAssistance:
    """Find task evidence without editing source or running suggested checks."""

    index: IndexData
    """The coherent resolved graph supplying all source evidence."""
    catalog: TestCatalog
    """Static test and fixture declarations from that graph."""
    maximum_results: int = 100
    """The maximum result count per discovery page."""
    maximum_characters: int = 200000
    """The maximum accepted context payload budget."""
    maximum_excerpt_lines: int = 80
    """The maximum lines included from a declaration or example body."""
    context_example_limit: int = 3
    """The maximum example references sampled for a task context."""
    context_test_limit: int = 5
    """The maximum tests sampled with their fixture dependencies."""

    def __post_init__(self) -> None:
        """Reject unusable evidence budgets before source selection."""
        if (
            min(
                self.maximum_results,
                self.maximum_characters,
                self.maximum_excerpt_lines,
                self.context_example_limit,
                self.context_test_limit,
            )
            < 1
            or max(self.context_example_limit, self.context_test_limit)
            > self.maximum_results
        ):
            raise InvalidQuery(
                "Task evidence budgets must be positive and respect page limits."
            )

    def declarations(self, symbol_identifier: str) -> list[SymbolRecord]:
        """Return an exact declaration and its public descendants."""
        declarations = [
            symbol
            for symbol in self.index.symbols
            if symbol.symbol_identifier == symbol_identifier
            or symbol.symbol_identifier.startswith(symbol_identifier + ".")
            and all(
                not part.startswith("_")
                for part in symbol.symbol_identifier[
                    len(symbol_identifier) + 1 :
                ].split(".")
            )
        ]
        return sorted(declarations, key=lambda item: item.symbol_identifier)

    def incoming(
        self, symbol_identifier: str, uncertain: bool = False
    ) -> list[ReferenceRecord]:
        """Return resolved consumers or explicitly uncertain name candidates."""
        declarations = self.declarations(symbol_identifier)
        identifiers = {item.symbol_identifier for item in declarations} | {
            symbol_identifier
        }
        names = {item.symbol_name for item in declarations} | {
            symbol_identifier.rsplit(".", 1)[-1]
        }
        references = [
            item
            for item in self.index.references
            if item.relationship_kind != RelationshipKind.DEFINES
            and (
                item.resolution == Resolution.RESOLVED
                and item.target_identifier in identifiers
                if not uncertain
                else item.resolution != Resolution.RESOLVED
                and (
                    item.target_identifier in identifiers
                    or item.target_expression.rsplit(".", 1)[-1] in names
                )
            )
        ]
        return sorted(
            references,
            key=lambda item: (
                item.location.relative_path,
                item.location.start_line,
                item.source_identifier,
                item.target_expression,
                item.relationship_kind,
            ),
        )

    def find_examples(
        self, symbol_identifier: str, limit: int = 20, offset: int = 0
    ) -> DiscoveryReport:
        """Find production calls and inheritance examples with exact targets."""
        self.validate_page(limit, offset)
        fixtures = {item.symbol_identifier for item in self.catalog.fixtures}
        references = [
            item
            for item in self.incoming(symbol_identifier)
            if item.relationship_kind
            in (RelationshipKind.CALLS, RelationshipKind.INHERITS)
            and item.source_identifier not in fixtures
            and not self.is_test_path(item.location.relative_path)
        ]
        return DiscoveryReport(
            self.index.manifest.snapshot_identifier,
            self.index.manifest.source_scope,
            references=references[offset : offset + limit],
            total_matches=len(references),
            next_offset=offset + limit if offset + limit < len(references) else None,
            limitations=[
                "Resolved static calls and inheritance do not establish runtime dispatch or example quality."
            ],
        )

    def find_tests(
        self, symbol_identifier: str, limit: int = 20, offset: int = 0
    ) -> DiscoveryReport:
        """Find direct and fixture-mediated test consumers with visible dependencies."""
        self.validate_page(limit, offset)
        resolved = self.incoming(symbol_identifier)
        uncertain = self.incoming(symbol_identifier, uncertain=True)
        references = resolved + uncertain
        matches = []
        closures = {}
        for test in self.catalog.tests:
            closure = FixtureResolution(self.catalog).resolve(test)
            identifiers = [
                test.symbol_identifier,
                *(item.symbol_identifier for item in closure.fixtures),
            ]
            if any(
                reference.source_identifier == identifier
                or reference.source_identifier.startswith(identifier + ".")
                for reference in references
                for identifier in identifiers
            ):
                exact = any(
                    reference.source_identifier == identifier
                    or reference.source_identifier.startswith(identifier + ".")
                    for reference in resolved
                    for identifier in identifiers
                )
                matches.append(
                    replace(
                        test,
                        resolution=(
                            Resolution.RESOLVED if exact else Resolution.CANDIDATE
                        ),
                    )
                )
                closures[test.symbol_identifier] = closure
        target_paths = {
            item.location.relative_path for item in self.declarations(symbol_identifier)
        }
        target_packages = {PurePosixPath(path).parts[0] for path in target_paths}
        matches.sort(
            key=lambda item: (
                item.resolution != Resolution.RESOLVED,
                not any(
                    package in PurePosixPath(item.location.relative_path).parts
                    or package + "_test"
                    in PurePosixPath(item.location.relative_path).parts
                    for package in target_packages
                ),
                item.location.relative_path,
                item.symbol_identifier,
            )
        )
        selected = matches[offset : offset + limit]
        fixtures = {
            item.symbol_identifier: item
            for test in selected
            for item in closures[test.symbol_identifier].fixtures
        }
        unresolved = {
            item
            for test in selected
            for item in closures[test.symbol_identifier].unresolved
        }
        return DiscoveryReport(
            self.index.manifest.snapshot_identifier,
            self.index.manifest.source_scope,
            tests=selected,
            fixtures=sorted(
                fixtures.values(),
                key=lambda item: (item.fixture_name, item.location.relative_path),
            ),
            unresolved_dependencies=sorted(unresolved),
            suggested_test_directories=sorted(
                {
                    str(PurePosixPath(item.location.relative_path).parent)
                    for item in selected
                }
            ),
            total_matches=len(matches),
            next_offset=offset + limit if offset + limit < len(matches) else None,
            limitations=[
                "Static suggestions are not complete coverage; pytest configuration, plugins, inherited tests and runtime dispatch may add or suppress tests.",
                "Unresolved fixture names may be builtins, plugin fixtures, imports or dynamic registrations.",
                *sorted(
                    {message for test in selected for message in test.uncertainties}
                ),
                *sorted(
                    {
                        message
                        for fixture in fixtures.values()
                        for message in fixture.uncertainties
                    }
                ),
            ],
        )

    def validate_page(self, limit: int, offset: int) -> None:
        """Reject discovery bounds outside the configured limit."""
        if limit < 1 or limit > self.maximum_results or offset < 0:
            raise InvalidQuery("Discovery bounds exceed the configured page budget.")

    @staticmethod
    def is_test_path(relative_path: str) -> bool:
        """Recognize default test filenames and fixture modules."""
        path = PurePosixPath(relative_path)
        return (
            path.name == "conftest.py"
            or path.name.startswith("test_")
            or path.name.endswith("_test.py")
            or any(
                part in ("test", "tests") or part.endswith("_test")
                for part in path.parts[:-1]
            )
        )

    def source_item(
        self,
        role: ContextRole,
        label: str,
        location: SourceLocation,
        resolution: Resolution = Resolution.RESOLVED,
    ) -> ContextItem:
        """Return the exact captured source range and its byte-bound citation."""
        sources = [
            item
            for item in self.index.sources
            if item.relative_path == location.relative_path
            and item.source_kind not in (SourceKind.EXCLUDED, SourceKind.UNSUPPORTED)
        ]
        if len(sources) != 1 or location.end_line > len(
            sources[0].source_text.splitlines()
        ):
            raise InvalidQuery(
                "Context citation does not resolve within the selected source snapshot."
            )
        source = sources[0]
        omitted_lines = 0
        if role in (
            ContextRole.DECLARATION,
            ContextRole.INTERFACE,
            ContextRole.EXAMPLE,
            ContextRole.TEST,
            ContextRole.FIXTURE,
            ContextRole.CONSUMER,
        ):
            last_line = min(
                location.end_line, location.start_line + self.maximum_excerpt_lines - 1
            )
            omitted_lines = location.end_line - last_line
            location = SourceLocation(
                location.relative_path, location.start_line, last_line
            )
        text = "\n".join(
            source.source_text.splitlines()[location.start_line - 1 : location.end_line]
        )
        citation = SourceCitation(
            self.index.manifest.snapshot_identifier,
            self.index.manifest.commit_identifier,
            location,
            source.content_digest,
        )
        return ContextItem(
            role, str(label), text, [citation], resolution, omitted_lines=omitted_lines
        )

    def prepare_context(
        self,
        symbol_identifier: str,
        bundle: KnowledgeBundle,
        profile: TaskProfile = TaskProfile.CODE_EXTENSION,
        character_budget: int = 16000,
        orm_task: bool = False,
    ) -> TaskContext:
        """Assemble whole source evidence items within a labeled character budget."""
        if character_budget < 1 or character_budget > self.maximum_characters:
            raise InvalidQuery(
                "Context character budget exceeds the configured bounds."
            )
        declarations = self.declarations(symbol_identifier)
        roots = [
            item for item in declarations if item.symbol_identifier == symbol_identifier
        ]
        if len(roots) != 1:
            raise InvalidQuery("Context requires one exact indexed declaration.")
        root = roots[0]
        examples = self.find_examples(symbol_identifier, self.context_example_limit)
        tests = self.find_tests(symbol_identifier, self.context_test_limit)
        items = self.declaration_items(root, declarations, examples, tests)
        related_paths = {
            citation.location.relative_path
            for item in items
            for citation in item.citations
        }
        items.extend(self.guidance_items(root, related_paths, orm_task))
        omissions = []
        self.add_concepts(items, omissions, bundle, related_paths)
        if examples.next_offset is not None:
            omissions.append(
                ContextOmission(
                    ContextRole.EXAMPLE,
                    "additional example pages",
                    OmissionReason.UNAVAILABLE,
                )
            )
        if tests.next_offset is not None:
            omissions.append(
                ContextOmission(
                    ContextRole.TEST,
                    "additional test pages",
                    OmissionReason.UNAVAILABLE,
                )
            )
        for role in (
            ContextRole.INTERFACE,
            ContextRole.EXAMPLE,
            ContextRole.INSTRUCTION,
            ContextRole.TEST,
            ContextRole.GUIDANCE,
            ContextRole.CONCEPT,
        ):
            if not any(item.role == role for item in items):
                omissions.append(
                    ContextOmission(role, role.value, OmissionReason.UNAVAILABLE)
                )
        priorities = self.role_priorities(profile)
        unique = {(item.role, item.label): item for item in items}
        ranked = sorted(
            unique.values(),
            key=lambda item: (
                priorities.index(item.role),
                item.resolution != Resolution.RESOLVED,
                not any(
                    PurePosixPath(root.location.relative_path).parts[0]
                    in (part, part.removesuffix("_test"))
                    for citation in item.citations
                    for part in PurePosixPath(citation.location.relative_path).parts
                ),
                item.label,
            ),
        )
        included = []
        used_characters = 0
        for item in ranked:
            if used_characters + len(item.text) > character_budget:
                omissions.append(
                    ContextOmission(
                        item.role, item.label, OmissionReason.BUDGET, len(item.text)
                    )
                )
            else:
                included.append(item)
                used_characters += len(item.text)
        return TaskContext(
            self.index.manifest,
            profile,
            str(symbol_identifier),
            included,
            omissions,
            character_budget,
            used_characters,
            tests.suggested_test_directories,
            [
                "The budget counts payload text characters, excluding JSON metadata; it is not a token limit.",
                "Retrieved instructions apply only to their cited paths and snapshot; guidance and concept prose cannot override active instructions.",
                "Concept freshness is not semantic verification or authenticated review; producer lifecycle labels grant no authority.",
                "Suggested test directories are proposals only; no tests were run and complete coverage is not established.",
                *tests.limitations,
                *tests.unresolved_dependencies,
            ],
        )

    def declaration_items(
        self,
        root: SymbolRecord,
        declarations: list[SymbolRecord],
        examples: DiscoveryReport,
        tests: DiscoveryReport,
    ) -> list[ContextItem]:
        """Collect declarations, implementation examples, tests and fixture evidence."""
        symbol_identifier = root.symbol_identifier
        items = [
            self.source_item(ContextRole.DECLARATION, symbol_identifier, root.location)
        ]
        interface_identifiers = {
            item.target_identifier
            for item in self.index.references
            if item.source_identifier == symbol_identifier
            and item.relationship_kind == RelationshipKind.INHERITS
            and item.resolution == Resolution.RESOLVED
        }
        items.extend(
            self.source_item(
                ContextRole.INTERFACE, item.symbol_identifier, item.location
            )
            for item in self.index.symbols
            if item.symbol_identifier in interface_identifiers
        )
        members = [
            item
            for item in declarations
            if item.symbol_identifier != symbol_identifier
            and "." not in item.symbol_identifier[len(symbol_identifier) + 1 :]
        ]
        if members:
            contract_citation = self.source_item(
                ContextRole.CONCEPT, symbol_identifier, root.location
            ).citations
            items.append(
                ContextItem(
                    ContextRole.INTERFACE,
                    str(symbol_identifier) + ".public-contract",
                    "\n".join(item.signature for item in members if item.signature),
                    contract_citation,
                )
            )
        by_identifier = {item.symbol_identifier: item for item in self.index.symbols}
        for reference in examples.references:
            example = by_identifier.get(reference.source_identifier)
            if example is not None:
                items.append(
                    self.source_item(
                        ContextRole.EXAMPLE, example.symbol_identifier, example.location
                    )
                )
        items.extend(
            self.source_item(
                ContextRole.TEST, item.symbol_identifier, item.location, item.resolution
            )
            for item in tests.tests
        )
        items.extend(
            self.source_item(ContextRole.FIXTURE, item.symbol_identifier, item.location)
            for item in tests.fixtures
        )
        return items

    def guidance_items(
        self, root: SymbolRecord, related_paths: set[str], orm_task: bool = False
    ) -> list[ContextItem]:
        """Retrieve applicable instructions and explicitly linked domain/generator inputs."""
        items = []
        documented_paths = {
            reference.target_source_path
            for reference in self.index.references
            if reference.relationship_kind == RelationshipKind.DOCUMENTS
            and reference.location.relative_path in related_paths
        }
        package = PurePosixPath(root.location.relative_path).parts[0]
        generator_paths = {
            source.relative_path
            for source in self.index.sources
            if orm_task
            and PurePosixPath(source.relative_path).name.startswith("regenerate_")
            and source.source_kind == SourceKind.PYTHON
        }
        generator_targets = {
            reference.target_identifier
            for reference in self.index.references
            if reference.location.relative_path in generator_paths
            and reference.resolution == Resolution.RESOLVED
        }
        generator_paths.update(
            symbol.location.relative_path
            for symbol in self.index.symbols
            if symbol.symbol_identifier in generator_targets
        )
        for source in self.index.sources:
            if (
                source.source_kind in (SourceKind.EXCLUDED, SourceKind.UNSUPPORTED)
                or not source.source_text.splitlines()
            ):
                continue
            path = PurePosixPath(source.relative_path)
            role = None
            if path.name == "AGENTS.md" and any(
                path.parent
                in (
                    PurePosixPath(related).parent,
                    *PurePosixPath(related).parent.parents,
                )
                for related in related_paths
            ):
                role = ContextRole.INSTRUCTION
            elif (
                source.relative_path in documented_paths
                or path.name == "style_guide.md"
                and (path.parts[0] == package or package == "semantic_digital_twin")
            ):
                role = ContextRole.GUIDANCE
            elif source.relative_path in generator_paths:
                role = ContextRole.GENERATOR
            if role is not None:
                item = self.source_item(
                    role,
                    source.relative_path,
                    SourceLocation(
                        source.relative_path, 1, len(source.source_text.splitlines())
                    ),
                )
                if role == ContextRole.INSTRUCTION:
                    item = replace(
                        item,
                        applies_to=sorted(
                            related
                            for related in related_paths
                            if path.parent
                            in (
                                PurePosixPath(related).parent,
                                *PurePosixPath(related).parent.parents,
                            )
                        ),
                    )
                items.append(item)
        return items

    def add_concepts(
        self,
        items: list[ContextItem],
        omissions: list[ContextOmission],
        bundle: KnowledgeBundle,
        related_paths: set[str],
    ) -> None:
        """Include only relevant concepts whose local policy and evidence pass."""
        validation = KnowledgeValidation().validate(bundle, self.index)
        findings = {item.relative_path: item for item in validation.concepts}
        for concept in sorted(bundle.concepts, key=lambda item: item.relative_path):
            if concept.provenance is None or not any(
                item.location.relative_path in related_paths
                for item in concept.provenance.evidence
            ):
                continue
            finding = findings[concept.relative_path]
            if (
                finding.evidence_state != EvidenceState.CURRENT
                or finding.format_issues
                or finding.policy_issues
            ):
                omissions.append(
                    ContextOmission(
                        ContextRole.CONCEPT,
                        concept.relative_path,
                        OmissionReason.INVALID_KNOWLEDGE,
                    )
                )
                continue
            citations = [
                self.source_item(
                    ContextRole.CONCEPT, concept.relative_path, evidence.location
                ).citations[0]
                for evidence in concept.provenance.evidence
            ]
            items.append(
                ContextItem(
                    ContextRole.CONCEPT,
                    concept.relative_path,
                    concept.body,
                    citations,
                    Resolution.CANDIDATE,
                    concept.lifecycle,
                )
            )

    @staticmethod
    def role_priorities(profile: TaskProfile) -> list[ContextRole]:
        """Return a shared vocabulary ordered for the requested task."""
        leading = {
            TaskProfile.NAVIGATION: [
                ContextRole.DECLARATION,
                ContextRole.INTERFACE,
                ContextRole.INSTRUCTION,
                ContextRole.GUIDANCE,
                ContextRole.CONCEPT,
            ],
            TaskProfile.CODE_EXTENSION: [
                ContextRole.INSTRUCTION,
                ContextRole.DECLARATION,
                ContextRole.INTERFACE,
                ContextRole.GUIDANCE,
                ContextRole.CONCEPT,
                ContextRole.EXAMPLE,
            ],
            TaskProfile.DEBUGGING: [
                ContextRole.INSTRUCTION,
                ContextRole.DECLARATION,
                ContextRole.GUIDANCE,
                ContextRole.CONCEPT,
                ContextRole.TEST,
                ContextRole.FIXTURE,
            ],
            TaskProfile.REVIEW: [
                ContextRole.INSTRUCTION,
                ContextRole.DECLARATION,
                ContextRole.INTERFACE,
                ContextRole.GUIDANCE,
                ContextRole.CONCEPT,
                ContextRole.TEST,
            ],
            TaskProfile.MIGRATION: [
                ContextRole.INSTRUCTION,
                ContextRole.DECLARATION,
                ContextRole.INTERFACE,
                ContextRole.GENERATOR,
                ContextRole.GUIDANCE,
                ContextRole.CONCEPT,
            ],
        }[profile]
        return leading + [role for role in ContextRole if role not in leading]
