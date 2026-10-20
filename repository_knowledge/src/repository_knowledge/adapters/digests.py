"""
Canonical content digests delegated to the owned KRROOD wire encoding.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing_extensions import TYPE_CHECKING

from repository_knowledge.adapters.serialization import RecordCodec
from repository_knowledge.contracts import RecordDigester
from repository_knowledge.publication import ExtractionFingerprint

if TYPE_CHECKING:
    from repository_knowledge.model import IndexData

# %% framed record identities


@dataclass
class CanonicalDigest(RecordDigester):
    """
    Bind complete record contents, including nested extension metadata.
    """

    codec: RecordCodec = field(default_factory=RecordCodec)
    """
    The canonical owned-record serialization boundary.
    """

    def digest(self, record: object) -> str:
        return hashlib.sha256(self.codec.encode(record).encode("utf-8")).hexdigest()

    def digest_records(self, records: list[object]) -> str:
        digest = hashlib.sha256()
        for record in records:
            encoded = self.codec.encode(record).encode("utf-8")
            digest.update(str(len(encoded)).encode("ascii"))
            digest.update(b":")
            digest.update(encoded)
        return digest.hexdigest()

    def fingerprint(self, index: IndexData) -> ExtractionFingerprint:
        """
        Bind complete extraction categories without one large serialized graph.
        """
        return ExtractionFingerprint(
            index.manifest,
            self.digest_records(index.sources),
            self.digest_records(index.symbols),
            self.digest_records(index.references),
            self.digest_records(index.diagnostics),
            self.digest_records(index.analyses),
        )
