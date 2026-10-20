"""
Credential-free identity transport over local API fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from repository_knowledge.adapters.github_identity import IdentityTransport

# %% authenticated API mimic


@dataclass
class FixtureIdentity(IdentityTransport):
    """
    Return explicit provider replies without live external calls.
    """

    documents: dict[str, object]
    """The external JSON fixture keyed by relative API resource."""
    archives: dict[str, bytes] = field(default_factory=dict)
    """
    The immutable archive responses.
    """

    requested_resources: list[str] = field(default_factory=list)
    """
    The resources observed by the mimic.
    """

    def read_json(self, resource: str) -> object:
        self.requested_resources.append(resource)
        return self.documents[resource]

    def read_bytes(self, resource: str) -> bytes:
        self.requested_resources.append(resource)
        return self.archives[resource]
