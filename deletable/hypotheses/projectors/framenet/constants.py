"""Shared constants for the FrameNet reasoner/projector pair.

Placing them here avoids a reasoning → hypotheses import dependency while
giving the projector a stable home for its versioning metadata.
"""

FRAMENET_REASONER_NAME: str = "framenet_reasoner"
FRAMENET_PROMPT_VERSION: str = "framenet_v1"
