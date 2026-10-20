"""Read GitHub identities and enforcement through a credential-isolated boundary."""

from __future__ import annotations

import json
import ssl
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, HTTPSHandler, Request, build_opener
from typing_extensions import TYPE_CHECKING, Any

from repository_knowledge.adapters.exceptions import InvalidIdentityEvidence
from repository_knowledge.contracts import ReviewReader, SourceIdentityReader
from repository_knowledge.publication import (
    ControlEvidence,
    ReviewDecision,
    ReviewEvidence,
    IdentityKind,
    PublicationMode,
    SourceIdentity,
)

if TYPE_CHECKING:
    from http.client import HTTPMessage
    from repository_knowledge.publication import PublicationPolicy

# %% external provider vocabulary


class ApiField(StrEnum):
    """Fields defined by GitHub's external identity API."""

    NUMBER = "number"
    USER = "user"
    LOGIN = "login"
    TYPE = "type"
    HEAD = "head"
    BASE = "base"
    SHA = "sha"
    REPO = "repo"
    FULL_NAME = "full_name"
    REF = "ref"
    MERGED = "merged"
    MERGE_COMMIT = "merge_commit_sha"
    COMMIT = "commit"
    TREE = "tree"
    IDENTIFIER = "id"
    STATE = "state"
    REVIEW_COMMIT = "commit_id"
    PERMISSION = "permission"
    ENFORCE_ADMINS = "enforce_admins"
    ENABLED = "enabled"
    REQUIRED_REVIEWS = "required_pull_request_reviews"
    DISMISS_STALE = "dismiss_stale_reviews"
    REQUIRE_OWNERS = "require_code_owner_reviews"
    REQUIRE_LAST_PUSH = "require_last_push_approval"
    APPROVAL_COUNT = "required_approving_review_count"
    BYPASS = "bypass_pull_request_allowances"
    USERS = "users"
    TEAMS = "teams"
    APPLICATIONS = "apps"
    REQUIRED_CHECKS = "required_status_checks"
    STRICT = "strict"
    CONTEXTS = "contexts"
    FORCE_PUSHES = "allow_force_pushes"
    DELETIONS = "allow_deletions"
    PROTECTION_RULES = "protection_rules"
    PREVENT_SELF_REVIEW = "prevent_self_review"
    REVIEWERS = "reviewers"
    REVIEWER = "reviewer"
    DEPLOYMENT_BRANCHES = "deployment_branch_policy"
    PROTECTED_BRANCHES = "protected_branches"
    CUSTOM_BRANCHES = "custom_branch_policies"
    NAME = "name"
    FORK = "fork"
    PATH = "path"
    HEAD_SHA = "head_sha"
    HEAD_BRANCH = "head_branch"
    EVENT = "event"
    CONCLUSION = "conclusion"
    STATUS = "status"
    REPOSITORY = "repository"
    HEAD_REPOSITORY = "head_repository"
    ENVIRONMENTS = "environments"
    DIGEST = "digest"
    EXPIRED = "expired"
    WORKFLOW_RUN = "workflow_run"
    SIZE = "size_in_bytes"
    PROTECTED = "protected"
    AUTHOR = "author"


class IdentityHeader(StrEnum):
    """HTTP headers used by the identity boundary."""

    AUTHORIZATION = "Authorization"
    ACCEPT = "Accept"
    USER_AGENT = "User-Agent"
    VERSION = "X-GitHub-Api-Version"


class ProviderValue(StrEnum):
    """Provider states required by the publication workflow contract."""

    REQUIRED_REVIEWERS = "required_reviewers"
    DISPATCH = "workflow_dispatch"
    PUSH = "push"
    SUCCESS = "success"
    COMPLETED = "completed"
    APPROVED = "approved"
    AHEAD = "ahead"
    IDENTICAL = "identical"


# %% bounded HTTPS transport


@dataclass
class IdentityTransport(ABC):
    """Read identity resources without giving producer files authority."""

    @abstractmethod
    def read_json(self, resource: str) -> object:
        """Read an external JSON resource from the fixed identity origin."""

    @abstractmethod
    def read_bytes(self, resource: str) -> bytes:
        """Read a bounded artifact response from the fixed identity origin."""


@dataclass
class CredentialRedirect(HTTPRedirectHandler):
    """Keep API authorization out of signed cross-origin artifact downloads."""

    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: HTTPMessage | dict[str, str],
        newurl: str,
    ) -> Request | None:
        """Remove authorization on redirects permitted by :meth:`HTTPRedirectHandler.redirect_request`."""
        if urlsplit(newurl).scheme != "https":
            raise InvalidIdentityEvidence("non-HTTPS artifact redirect")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None:
            redirected.remove_header(IdentityHeader.AUTHORIZATION)
        return redirected


@dataclass
class GitHubTransport(IdentityTransport):
    """Read bounded GitHub API replies with an optional dedicated identity credential."""

    token: str = field(default="", repr=False)
    """The dedicated provider credential, excluded from object representations."""
    timeout_seconds: float = 30.0
    """The maximum wait for one provider request."""
    maximum_bytes: int = 67108864
    """The maximum API or archive response bytes."""

    def read_json(self, resource: str) -> object:
        return json.loads(self.read_bytes(resource))

    def read_bytes(self, resource: str) -> bytes:
        if (
            not resource.startswith("/repos/")
            or "\\" in resource
            or ".." in resource.split("/")
        ):
            raise InvalidIdentityEvidence(resource)
        headers = {
            IdentityHeader.ACCEPT: "application/vnd.github+json",
            IdentityHeader.USER_AGENT: "cram-repository-knowledge",
            IdentityHeader.VERSION: "2026-03-10",
        }
        if self.token:
            headers[IdentityHeader.AUTHORIZATION] = "Bearer " + self.token
        request = Request("https://api.github.com" + resource, headers=headers)
        opener = build_opener(
            CredentialRedirect(), HTTPSHandler(context=ssl.create_default_context())
        )
        with opener.open(request, timeout=self.timeout_seconds) as response:
            content = response.read(self.maximum_bytes + 1)
        if len(content) > self.maximum_bytes:
            raise InvalidIdentityEvidence("oversized provider response")
        return content


# %% external JSON shape validation


@dataclass(frozen=True)
class ProviderObject:
    """A checked external JSON object with explicit scalar and nested reads."""

    contents: dict[str, object]
    """The external provider fields."""

    @classmethod
    def from_json(cls, value: object) -> ProviderObject:
        """Refuse non-object payloads before reading provider fields."""
        if not isinstance(value, dict) or any(
            not isinstance(key, str) for key in value
        ):
            raise InvalidIdentityEvidence("provider object")
        return cls(value)

    def text(self, key: ApiField) -> str:
        """Read a required external text field."""
        value = self.contents.get(key)
        if not isinstance(value, str):
            raise InvalidIdentityEvidence(key)
        return value

    def integer(self, key: ApiField) -> int:
        """Read a required integer without accepting booleans."""
        value = self.contents.get(key)
        if not isinstance(value, int) or isinstance(value, bool):
            raise InvalidIdentityEvidence(key)
        return value

    def boolean(self, key: ApiField) -> bool:
        """Read an explicitly declared provider boolean."""
        value = self.contents.get(key)
        if not isinstance(value, bool):
            raise InvalidIdentityEvidence(key)
        return value

    def nested(self, key: ApiField) -> ProviderObject:
        """Read a required nested provider object."""
        return ProviderObject.from_json(self.contents.get(key))

    def sequence(self, key: ApiField) -> list[object]:
        """Read a required external sequence."""
        value = self.contents.get(key)
        if not isinstance(value, list):
            raise InvalidIdentityEvidence(key)
        return value


@dataclass(frozen=True)
class PullIdentity:
    """The review-relevant fields of a provider pull request."""

    request_number: int
    """The provider request identifier."""
    author_login: str
    """The provider-authenticated proposal author."""
    repository: str
    """The authenticated target repository."""
    branch: str
    """The target branch."""
    head_commit: str
    """The proposal head commit."""
    merged_commit: str
    """The committed merged source."""
    merged: bool
    """Whether merging is confirmed."""

    @classmethod
    def from_json(cls, value: object) -> PullIdentity:
        """Read the external request shape independently of wiki declarations."""
        data = ProviderObject.from_json(value)
        return cls(
            data.integer(ApiField.NUMBER),
            data.nested(ApiField.USER).text(ApiField.LOGIN),
            data.nested(ApiField.BASE).nested(ApiField.REPO).text(ApiField.FULL_NAME),
            data.nested(ApiField.BASE).text(ApiField.REF),
            data.nested(ApiField.HEAD).text(ApiField.SHA),
            data.text(ApiField.MERGE_COMMIT),
            data.boolean(ApiField.MERGED),
        )


@dataclass(frozen=True)
class ReviewIdentity:
    """The immutable review event fields before permission resolution."""

    review_identifier: int
    """The provider review identifier."""
    actor_login: str
    """The authenticated reviewer login."""
    actor_kind: str
    """The provider actor category."""
    state: str
    """The provider's current state for this review."""
    commit_identifier: str
    """The exact commit reviewed."""

    @classmethod
    def from_json(cls, value: object) -> ReviewIdentity:
        """Read a provider review record before assigning repository authority."""
        data = ProviderObject.from_json(value)
        actor = data.nested(ApiField.USER)
        return cls(
            data.integer(ApiField.IDENTIFIER),
            actor.text(ApiField.LOGIN),
            actor.text(ApiField.TYPE),
            data.text(ApiField.STATE),
            data.text(ApiField.REVIEW_COMMIT),
        )


@dataclass(frozen=True)
class EnvironmentIdentity:
    """Deployment reviewer restrictions reported by the identity provider."""

    reviewers: list[str]
    """The individually configured human reviewers."""
    prevents_self_review: bool
    """Whether the initiating user cannot approve their own deployment."""
    protected_branches_only: bool
    """Whether deployment is restricted to protected branches."""

    @classmethod
    def from_json(cls, value: object) -> EnvironmentIdentity:
        """Require explicit individual reviewers and protected deployment branches."""
        data = ProviderObject.from_json(value)
        rules = [
            ProviderObject.from_json(item)
            for item in data.sequence(ApiField.PROTECTION_RULES)
        ]
        required = [
            rule
            for rule in rules
            if rule.text(ApiField.TYPE) == ProviderValue.REQUIRED_REVIEWERS
        ]
        if len(required) != 1:
            raise InvalidIdentityEvidence("environment review rules")
        reviewers = [
            ProviderObject.from_json(item)
            for item in required[0].sequence(ApiField.REVIEWERS)
        ]
        if not reviewers or any(
            item.text(ApiField.TYPE) != IdentityKind.HUMAN for item in reviewers
        ):
            raise InvalidIdentityEvidence("environment reviewer identity")
        branches = data.nested(ApiField.DEPLOYMENT_BRANCHES)
        return cls(
            [item.nested(ApiField.REVIEWER).text(ApiField.LOGIN) for item in reviewers],
            required[0].boolean(ApiField.PREVENT_SELF_REVIEW),
            branches.boolean(ApiField.PROTECTED_BRANCHES)
            and not branches.boolean(ApiField.CUSTOM_BRANCHES),
        )


# %% independent review provider


@dataclass
class GitHubReviews(ReviewReader, SourceIdentityReader):
    """Read fork controls, merged trees and reviewer authority from GitHub."""

    transport: IdentityTransport
    """The authenticated provider transport."""
    page_size: int = 100
    """The requested review-history page size."""
    maximum_pages: int = 100
    """The maximum history pages before refusing incomplete evidence."""

    def read_controls(self, policy: PublicationPolicy) -> ControlEvidence:
        root = "/repos/" + policy.repository
        repository = ProviderObject.from_json(self.transport.read_json(root))
        if repository.text(
            ApiField.FULL_NAME
        ) != policy.repository or not repository.boolean(ApiField.FORK):
            raise InvalidIdentityEvidence(
                "publication repository is not the configured fork"
            )
        if policy.mode == PublicationMode.AUTOMATED:
            branch = ProviderObject.from_json(
                self.transport.read_json(
                    root + "/branches/" + quote(policy.branch, safe="")
                )
            )
            if branch.text(ApiField.NAME) != policy.branch:
                raise InvalidIdentityEvidence("source branch identity")
            return ControlEvidence(
                policy.repository,
                policy.branch,
                branch.boolean(ApiField.PROTECTED),
                False,
                False,
                False,
                False,
                0,
                [],
                False,
            )
        protection = ProviderObject.from_json(
            self.transport.read_json(
                root + "/branches/" + quote(policy.branch, safe="") + "/protection"
            )
        )
        reviews = protection.nested(ApiField.REQUIRED_REVIEWS)
        checks = protection.nested(ApiField.REQUIRED_CHECKS)
        bypass = ProviderObject.from_json(reviews.contents.get(ApiField.BYPASS, {}))
        if any(
            bypass.contents.get(key, [])
            for key in (ApiField.USERS, ApiField.TEAMS, ApiField.APPLICATIONS)
        ):
            raise InvalidIdentityEvidence("review bypass allowance")
        if (
            protection.nested(ApiField.FORCE_PUSHES).boolean(ApiField.ENABLED)
            or protection.nested(ApiField.DELETIONS).boolean(ApiField.ENABLED)
            or not checks.boolean(ApiField.STRICT)
        ):
            raise InvalidIdentityEvidence("weak source protection")
        environment = EnvironmentIdentity.from_json(
            self.transport.read_json(
                root + "/environments/" + quote(policy.environment, safe="")
            )
        )
        configured = {login.casefold() for login in policy.reviewers}
        environment_protected = (
            environment.prevents_self_review
            and environment.protected_branches_only
            and bool(configured)
            and {login.casefold() for login in environment.reviewers}.issubset(
                configured
            )
        )
        contexts = checks.sequence(ApiField.CONTEXTS)
        if any(not isinstance(context, str) for context in contexts):
            raise InvalidIdentityEvidence("required check names")
        return ControlEvidence(
            policy.repository,
            policy.branch,
            True,
            reviews.boolean(ApiField.DISMISS_STALE),
            reviews.boolean(ApiField.REQUIRE_OWNERS),
            protection.nested(ApiField.ENFORCE_ADMINS).boolean(ApiField.ENABLED),
            reviews.boolean(ApiField.REQUIRE_LAST_PUSH),
            reviews.integer(ApiField.APPROVAL_COUNT),
            contexts,
            environment_protected,
        )

    def read_source(
        self, policy: PublicationPolicy, source_commit: str
    ) -> SourceIdentity:
        root = "/repos/" + policy.repository
        data = ProviderObject.from_json(
            self.transport.read_json(root + "/commits/" + quote(source_commit, safe=""))
        )
        if data.text(ApiField.SHA) != source_commit:
            raise InvalidIdentityEvidence("source commit identity")
        comparison = ProviderObject.from_json(
            self.transport.read_json(
                root
                + "/compare/"
                + quote(source_commit, safe="")
                + "..."
                + quote(policy.branch, safe="")
            )
        )
        if comparison.text(ApiField.STATUS) not in (
            ProviderValue.AHEAD,
            ProviderValue.IDENTICAL,
        ):
            raise InvalidIdentityEvidence("source is outside configured branch history")
        author = data.contents.get(ApiField.AUTHOR)
        login = (
            ProviderObject.from_json(author).text(ApiField.LOGIN)
            if author is not None
            else ""
        )
        return SourceIdentity(
            policy.repository,
            policy.branch,
            source_commit,
            data.nested(ApiField.COMMIT).nested(ApiField.TREE).text(ApiField.SHA),
            login,
            True,
        )

    def read_review(
        self, policy: PublicationPolicy, request_number: int
    ) -> ReviewEvidence:
        root = "/repos/" + policy.repository
        pull = PullIdentity.from_json(
            self.transport.read_json(root + f"/pulls/{request_number}")
        )
        if (
            pull.request_number != request_number
            or pull.repository != policy.repository
            or pull.branch != policy.branch
        ):
            raise InvalidIdentityEvidence("pull request identity")
        head = ProviderObject.from_json(
            self.transport.read_json(
                root + "/commits/" + quote(pull.head_commit, safe="")
            )
        )
        merged = ProviderObject.from_json(
            self.transport.read_json(
                root + "/commits/" + quote(pull.merged_commit, safe="")
            )
        )
        decisions = []
        allowed = {login.casefold() for login in policy.reviewers}
        for value in self.read_history(root + f"/pulls/{request_number}/reviews"):
            review = ReviewIdentity.from_json(value)
            permission = ""
            if (
                review.actor_login.casefold() in allowed
                and review.actor_kind == IdentityKind.HUMAN
            ):
                response = ProviderObject.from_json(
                    self.transport.read_json(
                        root
                        + "/collaborators/"
                        + quote(review.actor_login, safe="")
                        + "/permission"
                    )
                )
                permission = response.text(ApiField.PERMISSION)
            decisions.append(
                ReviewDecision(
                    review.review_identifier,
                    review.actor_login,
                    review.actor_kind,
                    review.state,
                    review.commit_identifier,
                    permission,
                )
            )
        return ReviewEvidence(
            pull.repository,
            pull.branch,
            pull.request_number,
            pull.author_login,
            pull.head_commit,
            head.nested(ApiField.COMMIT).nested(ApiField.TREE).text(ApiField.SHA),
            pull.merged_commit,
            merged.nested(ApiField.COMMIT).nested(ApiField.TREE).text(ApiField.SHA),
            decisions,
            pull.merged,
        )

    def read_history(self, resource: str) -> list[object]:
        """Read complete bounded review history rather than trusting its first page."""
        history = []
        for page in range(1, self.maximum_pages + 1):
            values = self.transport.read_json(
                resource + f"?per_page={self.page_size}&page={page}"
            )
            if not isinstance(values, list):
                raise InvalidIdentityEvidence(resource)
            history.extend(values)
            if len(values) < self.page_size:
                return history
        raise InvalidIdentityEvidence(
            "review history exceeds the configured completeness budget"
        )
