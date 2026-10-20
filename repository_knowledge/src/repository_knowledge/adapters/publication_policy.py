"""
Read independently selected publication policy without permissive unknown keys.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from repository_knowledge.adapters.exceptions import InvalidIdentityEvidence
from repository_knowledge.publication import PublicationPolicy, PublicationMode

# %% trusted policy configuration


@dataclass
class PolicyDocument:
    """
    Interpret a policy selected from the trusted checker checkout.
    """

    path: Path
    """The independently trusted TOML policy document."""

    def read(self) -> PublicationPolicy:
        """
        Read the external policy document and reject unknown authority fields.
        """
        data = tomllib.loads(self.path.read_text(encoding="utf-8"))
        return self.from_json(data)

    @staticmethod
    def from_json(data: dict[str, object]) -> PublicationPolicy:
        """
        Validate the external field shape before constructing the trusted policy.
        """
        defaults = vars(PublicationPolicy("", "", []))
        if set(data) - defaults.keys():
            raise InvalidIdentityEvidence("unknown publication policy field")
        for key, value in data.items():
            expected = (
                str
                if isinstance(defaults[key], PublicationMode)
                else type(defaults[key])
            )
            if (
                not isinstance(value, expected)
                or expected is int
                and isinstance(value, bool)
                or expected is list
                and any(not isinstance(item, str) for item in value)
            ):
                raise InvalidIdentityEvidence(key)
        values = dict(data)
        if "mode" in values:
            if values["mode"] not in PublicationMode:
                raise InvalidIdentityEvidence("publication mode")
            values["mode"] = PublicationMode(values["mode"])
        policy = PublicationPolicy(**values)
        components = policy.repository.split("/")
        if (
            len(components) != 2
            or any(
                re.fullmatch(r"[A-Za-z0-9_.-]+", part) is None or part in (".", "..")
                for part in components
            )
            or not policy.branch
            or policy.minimum_reviews < 0
            or policy.mode == PublicationMode.INDEPENDENT_REVIEW
            and policy.minimum_reviews < 1
            or policy.version != PublicationPolicy("", "", []).version
        ):
            raise InvalidIdentityEvidence("publication policy identity or version")
        if any(
            not login or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]*", login) is None
            for login in policy.reviewers
        ) or len({login.casefold() for login in policy.reviewers}) != len(
            policy.reviewers
        ):
            raise InvalidIdentityEvidence("publication reviewer configuration")
        return policy
