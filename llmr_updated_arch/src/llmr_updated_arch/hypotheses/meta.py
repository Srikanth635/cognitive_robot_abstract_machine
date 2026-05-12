"""Shared metadata contracts for sg_model hypothesis entities."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import re

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
    """Return a timezone-aware UTC timestamp for hypothesis metadata."""

    return datetime.now(timezone.utc)


def _shorten_identifier(identifier: str, *, long_segment_threshold: int = 12) -> str:
    """Return a compact display form for long internal identifiers."""

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
    """Shared provenance and epistemic metadata for hypothesis entities."""

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
