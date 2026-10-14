"""Candidate lifecycle orchestration; native construction and reasoning are adapter work."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

from krrood_ontology.contracts import (
    AcceptancePolicy,
    AcceptanceRequest,
    CandidateContext,
    CandidateSnapshot,
    CheckReport,
    CheckStage,
    CheckStatus,
    Feedback,
    FeedbackCategory,
    Proposal,
    Severity,
    StageResult,
)


class LifecycleError(RuntimeError):
    """A rejected transition that leaves no successful review for stale content."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


class AcceptanceBlockedError(LifecycleError):
    def __init__(self, feedback: tuple[Feedback, ...]):
        self.feedback = feedback
        super().__init__(
            "acceptance_blocked", "Required review conditions have not passed"
        )


class SessionState(str, Enum):
    DRAFT = "draft"
    REVIEWED = "reviewed"
    READY_FOR_ACCEPTANCE = "ready_for_acceptance"
    CLOSED = "closed"


@runtime_checkable
class CandidateChecker(Protocol):
    """Trusted ontology-system adapter, independent of agent and transport APIs."""

    def check(self, snapshot: CandidateSnapshot) -> CheckReport: ...


@dataclass(frozen=True)
class UnavailableChecker:
    """Explicit capability feedback until concrete review adapters are installed."""

    def check(self, snapshot: CandidateSnapshot) -> CheckReport:
        return CheckReport(
            snapshot.token,
            tuple(
                StageResult(
                    stage,
                    CheckStatus.UNSUPPORTED,
                    "not_available",
                    (
                        Feedback(
                            FeedbackCategory.UNSUPPORTED,
                            Severity.ERROR,
                            "phase_not_implemented",
                            f"{stage.value} execution is not implemented in Phase 11",
                        ),
                    ),
                )
                for stage in CheckStage
            ),
        )


@dataclass(init=False)
class CandidateSession:
    """Maintain immutable request revisions and reviews without accepting RDF facts."""

    _snapshot: CandidateSnapshot
    _checker: CandidateChecker = field(repr=False)
    _policy: AcceptancePolicy
    _state: SessionState
    _report: CheckReport | None
    _generation: int
    _lock: Any = field(repr=False)

    def __init__(
        self,
        context: CandidateContext,
        proposal: Proposal,
        *,
        checker: CandidateChecker,
        policy: AcceptancePolicy,
    ):
        if not isinstance(checker, CandidateChecker) or not isinstance(
            policy, AcceptancePolicy
        ):
            raise TypeError("Expected a checker adapter and acceptance policy")
        self._snapshot = CandidateSnapshot.build(uuid4().hex, 1, context, proposal)
        self._checker, self._policy = checker, policy
        self._state, self._report = SessionState.DRAFT, None
        self._generation = 0
        self._lock = RLock()

    @property
    def snapshot(self) -> CandidateSnapshot:
        with self._lock:
            return self._snapshot

    @property
    def state(self) -> SessionState:
        with self._lock:
            return self._state

    @property
    def report(self) -> CheckReport | None:
        with self._lock:
            return self._report

    def _ensure_open(self):
        if self._state is SessionState.CLOSED:
            raise LifecycleError("closed_session", "Session is closed")

    def replace_proposal(
        self, proposal: Proposal, *, expected_revision: int
    ) -> CandidateSnapshot:
        """Replace the complete proposal and invalidate earlier review/acceptance intent."""
        with self._lock:
            self._ensure_open()
            if (
                type(expected_revision) is not int
                or expected_revision != self._snapshot.token.revision
            ):
                raise LifecycleError(
                    "revision_conflict",
                    "Expected revision does not match the current candidate",
                )
            updated = CandidateSnapshot.build(
                self._snapshot.token.session_id,
                expected_revision + 1,
                self._snapshot.context,
                proposal,
            )
            self._snapshot, self._report, self._state = (
                updated,
                None,
                SessionState.DRAFT,
            )
            self._generation += 1
            return updated

    def check(self) -> CheckReport:
        """Run the configured adapter and retain only the current check's result."""
        with self._lock:
            self._ensure_open()
            snapshot = self._snapshot
            self._generation += 1
            generation = self._generation
            self._report, self._state = None, SessionState.DRAFT
        try:
            report = self._checker.check(snapshot)
        except Exception as error:
            report = CheckReport(
                snapshot.token,
                (),
                (
                    Feedback(
                        FeedbackCategory.EXECUTION,
                        Severity.ERROR,
                        "checker_failed",
                        f"Checker raised {type(error).__name__}; no successful review was recorded",
                    ),
                ),
            )
        with self._lock:
            if (
                self._state is SessionState.CLOSED
                or generation != self._generation
                or snapshot.token != self._snapshot.token
            ):
                raise LifecycleError(
                    "stale_review",
                    "Candidate changed, closed or was rechecked while this review was running",
                )
            if not isinstance(report, CheckReport) or report.token != snapshot.token:
                raise LifecycleError(
                    "review_token", "Adapter returned a malformed or foreign review"
                )
            self._report, self._state = report, SessionState.REVIEWED
            return report

    def prepare_acceptance(self) -> AcceptanceRequest:
        """Prepare an intent for the reviewed revision; do not commit facts or objects."""
        with self._lock:
            self._ensure_open()
            if self._report is None:
                raise AcceptanceBlockedError(
                    (
                        Feedback(
                            FeedbackCategory.LIFECYCLE,
                            Severity.ERROR,
                            "missing_review",
                            "The current candidate has no review",
                        ),
                    )
                )
            blockers = self._policy.blockers(self._report)
            if blockers:
                raise AcceptanceBlockedError(blockers)
            self._state = SessionState.READY_FOR_ACCEPTANCE
            return AcceptanceRequest(
                self._snapshot.token,
                self._snapshot.context,
                self._policy.required_checks,
            )

    def close(self) -> None:
        """Prevent further transitions and invalidate in-flight reviews."""
        with self._lock:
            self._generation += 1
            self._state = SessionState.CLOSED


@dataclass(frozen=True)
class OntologySystem:
    """Phase 11 service boundary supplying candidate sessions and adapter configuration."""

    context: CandidateContext
    checker: CandidateChecker | None = field(default=None, repr=False)
    policy: AcceptancePolicy = field(default_factory=AcceptancePolicy)

    def __post_init__(self):
        if not isinstance(self.context, CandidateContext) or not isinstance(
            self.policy, AcceptancePolicy
        ):
            raise TypeError("Expected context and policy records")
        if self.checker is not None and not isinstance(self.checker, CandidateChecker):
            raise TypeError("Expected a checker adapter")

    def begin_candidate(self, proposal: Proposal | None = None) -> CandidateSession:
        """Start an isolated contract revision without creating symbols or RDF assertions."""
        return CandidateSession(
            self.context,
            proposal if proposal is not None else Proposal(()),
            checker=self.checker if self.checker is not None else UnavailableChecker(),
            policy=self.policy,
        )
