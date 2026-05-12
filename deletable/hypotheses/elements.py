"""Graph element contracts shared by all hypothesis graph nodes and edges."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import re
from typing import ClassVar
from typing_extensions import Optional


class ClaimStatus(str, Enum):
    """Epistemic status of a reasoner-produced claim."""

    HYPOTHESIS = "hypothesis"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    SUPERSEDED = "superseded"


class GroundingState(str, Enum):
    """How strongly a claim is connected to structured system state."""

    TEXT_ONLY = "text_only"
    SLOT_ALIGNED = "slot_aligned"
    SYMBOL_GROUNDED = "symbol_grounded"


def _utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp for graph metadata."""

    return datetime.now(timezone.utc)


def _shorten_identifier(identifier: str, *, long_segment_threshold: int = 12) -> str:
    """Return a compact display form for long internal identifiers.

    Long colon-delimited ids keep their structure while long segments are
    shortened. Plain ids fall back to a simple prefix abbreviation.
    """

    if len(identifier) <= long_segment_threshold:
        return identifier

    opaque_segment_pattern = re.compile(r"^[0-9a-f]{13,}$", re.IGNORECASE)
    parts = identifier.split(":")
    if len(parts) > 1:
        shortened_parts = [
            part[:8]
            if (
                len(part) > long_segment_threshold
                and opaque_segment_pattern.fullmatch(part) is not None
            )
            else part
            for part in parts
        ]
        return ":".join(shortened_parts)

    return identifier[:8]


@dataclass
class HypothesisMeta:
    """Shared provenance and epistemic metadata for graph elements."""

    source_reasoner: str
    status: ClaimStatus = ClaimStatus.HYPOTHESIS
    grounding: GroundingState = GroundingState.TEXT_ONLY
    confidence: Optional[float] = None
    run_id: Optional[str] = None
    prompt_version: Optional[str] = None
    model_name: Optional[str] = None
    created_at: datetime = field(default_factory=_utc_now)

    @property
    def short_run_id(self) -> Optional[str]:
        """Compact display form for the current run id, if present."""

        if self.run_id is None:
            return None
        return _shorten_identifier(self.run_id)


@dataclass
class HypothesisGraphElement(ABC):
    """Base identity contract for every hypothesis graph element."""

    id: str
    meta: HypothesisMeta

    @property
    def short_id(self) -> str:
        """Compact display form for this element id."""

        return _shorten_identifier(self.id)

    @property
    def display_id(self) -> str:
        """Alias for the compact id used in UI-facing contexts."""

        return self.short_id


@dataclass
class HypothesisNode(HypothesisGraphElement, ABC):
    """Abstract base class for all graph nodes."""

    @property
    def dot_label(self) -> str:
        return type(self).__name__


@dataclass
class HypothesisEdge(HypothesisGraphElement, ABC):
    """Abstract base class for all graph edges."""

    RELATION_NAME: ClassVar[str] = ""

    src_id: str
    dst_id: str

    @property
    def relation_name(self) -> str:
        """Stable relation name derived from the concrete edge type."""

        return type(self).RELATION_NAME
