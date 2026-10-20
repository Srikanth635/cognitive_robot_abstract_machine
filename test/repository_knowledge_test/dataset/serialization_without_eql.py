"""
Round-trip nested evidence while EQL is unavailable.
"""

from __future__ import annotations

import sys
from enum import StrEnum

from repository_knowledge.adapters.serialization import RecordCodec
from repository_knowledge.model import SourceExcerpt, SourceLocation

# %% dependency isolation


class BlockedPackage(StrEnum):
    """
    The optional runtime subsystem excluded from this process.
    """

    EQL = "krrood.entity_query_language"


def main() -> int:
    """
    Return success only when nested evidence needs no EQL import.
    """
    sys.modules[BlockedPackage.EQL] = None
    original = SourceExcerpt(SourceLocation("source.py", 1, 2), "source")
    codec = RecordCodec()
    decoded = codec.decode(codec.encode(original), SourceExcerpt)
    return 0 if decoded == original else 1


if __name__ == "__main__":
    sys.exit(main())
