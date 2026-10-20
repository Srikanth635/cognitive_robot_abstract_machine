"""
Validate curated explanations without granting publication authority.
"""

from __future__ import annotations

import calendar
import hashlib
import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import StrEnum, IntEnum

from repository_knowledge.concepts import (
    BundleValidation,
    Concept,
    ConceptKind,
    ConceptValidation,
    EvidenceState,
    KnowledgeBundle,
    KnowledgeIssue,
    Lifecycle,
    ConceptPage,
    ContentComparison,
)
from repository_knowledge.exceptions import InvalidQuery
from repository_knowledge.operations.exceptions import InvalidConceptTransition
from repository_knowledge.model import IndexData, SourceKind

# %% portable metadata conventions


class TimestampGroup(IntEnum):
    """
    Capture positions in the supported ISO timestamp spelling.
    """

    SECOND = 6
    OFFSET = 8


@dataclass(frozen=True)
class EventFindings:
    """
    Declared actors and policy findings from portable event metadata.
    """

    actors: list[str]
    """
    The untrusted verifier identity spellings.
    """

    issues: list[KnowledgeIssue]
    """
    The policy violations found in the event declarations.
    """


class MetadataField(StrEnum):
    """
    Portable metadata interpreted by the CRAM policy.
    """

    VERIFIED = "verified"
    GENERATED = "generated"
    ACTOR = "by"
    INSTANT = "at"
    SOURCES = "sources"
    IDENTIFIER = "id"
    RESOURCE = "resource"


@dataclass
class MetadataPolicy:
    """
    Check optional declaration shapes without treating them as identities.
    """

    @staticmethod
    def parse_instant(value: str) -> datetime | None:
        """
        Interpret the supported offset-aware ISO timestamp spelling.
        """
        match = re.fullmatch(
            r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,6}))?(Z|[+-]\d{2}:\d{2})",
            value,
        )
        if match is None:
            return None
        year, month, day, hour, minute, second = map(
            int, match.groups()[: TimestampGroup.SECOND]
        )
        if not (
            datetime.min.year <= year <= datetime.max.year
            and datetime.min.month <= month <= datetime.max.month
            and 1 <= day <= calendar.monthrange(year, month)[1]
            and datetime.min.hour <= hour <= datetime.max.hour
            and datetime.min.minute <= minute <= datetime.max.minute
            and datetime.min.second <= second <= datetime.max.second
        ):
            return None
        offset = match.group(TimestampGroup.OFFSET)
        if offset != "Z" and (
            int(offset[1:3]) > datetime.max.hour
            or int(offset[4:]) > datetime.max.minute
        ):
            return None
        return datetime.fromisoformat(value)

    def check_events(self, concept: Concept) -> EventFindings:
        """
        Return declared actors and policy findings for generation and review.
        """
        actors: list[str] = []
        issues: list[KnowledgeIssue] = []
        for family in (MetadataField.GENERATED, MetadataField.VERIFIED):
            value = concept.metadata.get(family, [])
            events = [value] if isinstance(value, dict) else value
            if not isinstance(events, list):
                issues.append(KnowledgeIssue.INVALID_ACTOR)
                continue
            for event in events:
                declaration = ActorDeclaration.from_json(event)
                if family == MetadataField.VERIFIED:
                    actors.append(declaration.actor)
                    issues.append(KnowledgeIssue.UNAUTHENTICATED_REVIEW)
                if (
                    re.fullmatch(
                        r"(?:human:[^\s:]+|process:[^\s:]+|[^\s/:]+/[^\s/]+)",
                        declaration.actor,
                    )
                    is None
                ):
                    issues.append(KnowledgeIssue.INVALID_ACTOR)
                if self.parse_instant(declaration.instant) is None:
                    issues.append(KnowledgeIssue.INVALID_TIMESTAMP)
        return EventFindings(actors, issues)


@dataclass(frozen=True)
class ActorDeclaration:
    """
    An untrusted portable actor event at the metadata boundary.
    """

    actor: str
    """
    The producer-declared identity spelling.
    """

    instant: str
    """
    The producer-declared event timestamp.
    """

    @classmethod
    def from_json(cls, value: object) -> ActorDeclaration:
        """
        Read an external event without accepting it as review evidence.
        """
        if not isinstance(value, dict):
            return cls("", "")
        actor = value.get(MetadataField.ACTOR, "")
        instant = value.get(MetadataField.INSTANT, "")
        return cls(
            actor if isinstance(actor, str) else "",
            instant if isinstance(instant, str) else "",
        )


# %% structural source consistency


@dataclass(frozen=True)
class SourceFindings:
    """
    Source consistency findings and recognized byte-equivalence matches.
    """

    issues: list[KnowledgeIssue]
    """
    The failed source checks.
    """

    text_equivalent_paths: list[str]
    """
    The sources matched through an explicit line-ending equivalence check.
    """


@dataclass
class KnowledgeValidation:
    """
    Compare recorded concept evidence with a coherent source graph.
    """

    content_comparison: ContentComparison = ContentComparison.UTF8_LINE_ENDINGS
    """
    The permitted equality relation for recorded and selected source bytes.
    """

    def validate(
        self, bundle: KnowledgeBundle, index: IndexData, now: datetime | None = None
    ) -> BundleValidation:
        """
        Report independent checks and propagate explicit dependency invalidation.
        """
        instant = now or datetime.now(timezone.utc)
        if instant.utcoffset() is None:
            raise InvalidQuery("Knowledge validation needs an offset-aware clock.")
        paths = [concept.relative_path for concept in bundle.concepts]
        if len(paths) != len(set(paths)):
            raise InvalidQuery("A bundle cannot contain duplicate concept paths.")
        results = [
            self._check_concept(concept, bundle, index, instant)
            for concept in bundle.concepts
        ]
        by_path = {result.relative_path: result for result in results}
        changed = True
        while changed:
            changed = False
            for concept in bundle.concepts:
                if concept.provenance is None:
                    continue
                result = by_path[concept.relative_path]
                dependencies = [
                    *concept.provenance.depends_on,
                    *(path for path in concept.links if path in by_path),
                ]
                invalid = any(
                    target not in by_path
                    or by_path[target].evidence_state != EvidenceState.CURRENT
                    for target in dependencies
                )
                if (
                    not invalid
                    or KnowledgeIssue.STALE_DEPENDENCY in result.evidence_issues
                ):
                    continue
                by_path[concept.relative_path] = replace(
                    result,
                    evidence_state=EvidenceState.STALE,
                    evidence_issues=[
                        *result.evidence_issues,
                        KnowledgeIssue.STALE_DEPENDENCY,
                    ],
                )
                changed = True
        return BundleValidation(
            index.manifest.snapshot_identifier,
            [by_path[path] for path in paths],
            bundle.navigation_issues,
            index.manifest.source_scope,
            index.manifest.commit_identifier,
            instant.isoformat(),
            self.content_comparison,
        )

    def _check_concept(
        self, concept: Concept, bundle: KnowledgeBundle, index: IndexData, now: datetime
    ) -> ConceptValidation:
        """
        Assess one concept without modifying its lifecycle or evidence.
        """
        format_issues = (
            [] if concept.concept_type.strip() else [KnowledgeIssue.MISSING_TYPE]
        )
        policy = self._check_policy(concept, bundle, index)
        events = MetadataPolicy().check_events(concept)
        policy.extend(events.issues)
        source_findings = self._check_evidence(concept, index)
        evidence = list(source_findings.issues)
        if concept.stale_after:
            expiry = MetadataPolicy.parse_instant(concept.stale_after)
            if expiry is None:
                policy.append(KnowledgeIssue.INVALID_TIMESTAMP)
                evidence.append(KnowledgeIssue.INVALID_TIMESTAMP)
            elif now >= expiry:
                evidence.append(KnowledgeIssue.EXPIRED)
        state = EvidenceState.CURRENT
        if concept.provenance is None or not concept.provenance.evidence:
            state = EvidenceState.UNKNOWN
        if evidence:
            state = EvidenceState.STALE
        return ConceptValidation(
            concept.relative_path,
            concept.lifecycle,
            format_issues,
            list(dict.fromkeys(policy)),
            list(dict.fromkeys(evidence)),
            state,
            events.actors,
            text_equivalent_paths=source_findings.text_equivalent_paths,
        )

    @staticmethod
    def _check_policy(
        concept: Concept, bundle: KnowledgeBundle, index: IndexData
    ) -> list[KnowledgeIssue]:
        """
        Check CRAM-specific requirements independently of format conformance.
        """
        issues: list[KnowledgeIssue] = []
        if not concept.title:
            issues.append(KnowledgeIssue.MISSING_TITLE)
        if not concept.description:
            issues.append(KnowledgeIssue.MISSING_DESCRIPTION)
        if concept.concept_type not in ConceptKind:
            issues.append(KnowledgeIssue.UNKNOWN_TYPE)
        if concept.lifecycle not in Lifecycle:
            issues.append(KnowledgeIssue.UNKNOWN_LIFECYCLE)
        targets = {item.relative_path for item in bundle.concepts} | set(
            bundle.navigation
        )
        if any(target not in targets for target in concept.links):
            issues.append(KnowledgeIssue.BROKEN_LINK)
        provenance = concept.provenance
        if provenance is None:
            issues.append(KnowledgeIssue.MISSING_PROVENANCE)
            return issues
        if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", provenance.source_commit) is None:
            issues.append(KnowledgeIssue.INVALID_COMMIT)
        if not provenance.evidence:
            issues.append(KnowledgeIssue.MISSING_EVIDENCE)
        if concept.concept_type == ConceptKind.PACKAGE and not any(
            item.symbol_identifier for item in provenance.evidence
        ):
            issues.append(KnowledgeIssue.MISSING_TYPE_EVIDENCE)
        document_paths = {
            source.relative_path
            for source in index.sources
            if source.source_kind == SourceKind.DOCUMENT
        }
        if concept.concept_type == ConceptKind.CONVENTION and not any(
            item.location.relative_path in document_paths
            for item in provenance.evidence
        ):
            issues.append(KnowledgeIssue.MISSING_TYPE_EVIDENCE)
        source_entries = concept.metadata.get(MetadataField.SOURCES, [])
        declarations = (
            [SourceDeclaration.from_json(value) for value in source_entries]
            if isinstance(source_entries, list)
            else []
        )
        identifiers = [entry.identifier for entry in declarations if entry.resource]
        expected = {item.source_identifier for item in provenance.evidence}
        if (
            not expected.issubset(set(identifiers))
            or not expected.issubset(set(concept.attributions))
            or len(identifiers) != len(set(identifiers))
        ):
            issues.append(KnowledgeIssue.INVALID_ATTRIBUTION)
        if any(path not in targets for path in provenance.depends_on):
            issues.append(KnowledgeIssue.UNKNOWN_DEPENDENCY)
        if concept.lifecycle != Lifecycle.DEPRECATED:
            return issues
        if not provenance.replacement or provenance.replacement not in targets:
            issues.append(KnowledgeIssue.MISSING_REPLACEMENT)
            return issues
        replacements = {
            item.relative_path: item.provenance.replacement
            for item in bundle.concepts
            if item.provenance is not None
        }
        visited = {concept.relative_path}
        target = provenance.replacement
        while target:
            if target in visited:
                issues.append(KnowledgeIssue.REPLACEMENT_CYCLE)
                break
            visited.add(target)
            target = replacements.get(target, "")
        return issues

    def _check_evidence(self, concept: Concept, index: IndexData) -> SourceFindings:
        """
        Check source identity, permitted content equality, spans and declarations.
        """
        if concept.provenance is None:
            return SourceFindings([], [])
        sources = {source.relative_path: source for source in index.sources}
        symbols = {symbol.symbol_identifier: symbol for symbol in index.symbols}
        issues: list[KnowledgeIssue] = []
        text_equivalent_paths: list[str] = []
        for evidence in concept.provenance.evidence:
            source = sources.get(evidence.location.relative_path)
            if source is None or source.source_kind in (
                SourceKind.EXCLUDED,
                SourceKind.UNSUPPORTED,
            ):
                issues.append(KnowledgeIssue.MISSING_SOURCE)
                continue
            if source.content_digest != evidence.content_digest:
                variants = {
                    hashlib.sha256(text.encode("utf-8")).hexdigest()
                    for text in (
                        source.source_text,
                        source.source_text.replace("\n", "\r\n"),
                    )
                }
                if self.content_comparison == ContentComparison.UTF8_LINE_ENDINGS and {
                    source.content_digest,
                    evidence.content_digest,
                }.issubset(variants):
                    text_equivalent_paths.append(source.relative_path)
                else:
                    issues.append(KnowledgeIssue.CHANGED_SOURCE)
            if evidence.location.end_line > len(source.source_text.splitlines()):
                issues.append(KnowledgeIssue.INVALID_SOURCE_RANGE)
            if not evidence.symbol_identifier:
                continue
            symbol = symbols.get(evidence.symbol_identifier)
            if symbol is None:
                issues.append(KnowledgeIssue.MISSING_SYMBOL)
            elif symbol.location != evidence.location:
                issues.append(KnowledgeIssue.CHANGED_SYMBOL_LOCATION)
        return SourceFindings(issues, list(dict.fromkeys(text_equivalent_paths)))


# %% drafting lifecycle


@dataclass
class ConceptLifecycle:
    """
    Prepare draft changes and retain old identities as tombstones.
    """

    def transition(self, concept: Concept, lifecycle: Lifecycle) -> Concept:
        """
        Allow draft or replaced tombstone proposals without stable promotion.
        """
        if lifecycle == Lifecycle.STABLE or (
            lifecycle == Lifecycle.DEPRECATED
            and (concept.provenance is None or not concept.provenance.replacement)
        ):
            raise InvalidConceptTransition(concept.relative_path)
        metadata = dict(concept.metadata)
        metadata.pop(MetadataField.VERIFIED, None)
        metadata.pop(MetadataField.GENERATED, None)
        return replace(concept, lifecycle=lifecycle, metadata=metadata)

    def relocate(
        self, bundle: KnowledgeBundle, old_path: str, new_path: str
    ) -> KnowledgeBundle:
        """
        Keep incoming links meaningful while drafting a replacement document.
        """
        matches = [
            concept for concept in bundle.concepts if concept.relative_path == old_path
        ]
        if len(matches) != 1 or any(
            concept.relative_path == new_path for concept in bundle.concepts
        ):
            raise InvalidConceptTransition(old_path)
        original = matches[0]
        if original.provenance is None:
            raise InvalidConceptTransition(old_path)
        replacement = replace(
            self.transition(original, Lifecycle.DRAFT), relative_path=new_path
        )
        tombstone = self.transition(
            replace(
                original, provenance=replace(original.provenance, replacement=new_path)
            ),
            Lifecycle.DEPRECATED,
        )
        source_identifiers = list(
            dict.fromkeys(
                item.source_identifier for item in original.provenance.evidence
            )
        )
        citations = "".join(f"[^{identifier}]" for identifier in source_identifiers)
        definitions = "\n".join(
            f"[^{item.source_identifier}]: Historical source {item.location.relative_path}"
            for item in original.provenance.evidence
        )
        tombstone = replace(
            tombstone,
            body=f"# Deprecated\n\nContinue at [the replacement](/{new_path}).\n\nHistorical source evidence{citations}.\n\n{definitions}\n",
            links=[new_path],
            attributions=source_identifiers,
        )
        concepts = [
            tombstone if item.relative_path == old_path else item
            for item in bundle.concepts
        ]
        return KnowledgeBundle(
            [*concepts, replacement],
            dict(bundle.navigation),
            dict(bundle.navigation_issues),
        )


@dataclass(frozen=True)
class SourceDeclaration:
    """
    The portable source fields needed to check claim attribution.
    """

    identifier: str
    """
    The stable source footnote key.
    """

    resource: str
    """
    The declared source locator.
    """

    @classmethod
    def from_json(cls, value: object) -> SourceDeclaration:
        """
        Read externally authored source metadata without dereferencing it.
        """
        if not isinstance(value, dict):
            return cls("", "")
        identifier = value.get(MetadataField.IDENTIFIER, "")
        resource = value.get(MetadataField.RESOURCE, "")
        return cls(
            identifier if isinstance(identifier, str) else "",
            resource if isinstance(resource, str) else "",
        )


# %% bounded concept discovery


@dataclass
class KnowledgeQueries:
    """
    Discover explanations while retaining their evidence and authority limits.
    """

    maximum_results: int = 100
    """
    The maximum number of concept documents returned per page.
    """

    def find(
        self,
        bundle: KnowledgeBundle,
        index: IndexData,
        query: str,
        limit: int = 20,
        offset: int = 0,
    ) -> ConceptPage:
        """
        Find concepts by exact path or lexical title, summary and body matches.
        """
        if limit < 1 or limit > self.maximum_results or offset < 0:
            raise InvalidQuery(
                "Concept page bounds exceed the configured result budget."
            )
        text = query.casefold()
        exact = [
            concept
            for concept in bundle.concepts
            if concept.relative_path.casefold() == text
        ]
        matches = exact or [
            concept
            for concept in bundle.concepts
            if text
            in "\n".join(
                (
                    concept.relative_path,
                    concept.title,
                    concept.description,
                    concept.body,
                )
            ).casefold()
        ]
        matches = sorted(matches, key=lambda concept: concept.relative_path)
        selected = matches[offset : offset + limit]
        validation = KnowledgeValidation().validate(bundle, index)
        selected_paths = {concept.relative_path for concept in selected}
        validation = replace(
            validation,
            concepts=[
                item
                for item in validation.concepts
                if item.relative_path in selected_paths
            ],
        )
        next_offset = (
            offset + len(selected) if offset + len(selected) < len(matches) else None
        )
        return ConceptPage(selected, validation, len(matches), next_offset)
