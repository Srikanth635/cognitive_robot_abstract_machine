"""Backward-compatible re-exports. Canonical home: :mod:`llmr.grounder`.

``resolve_symbol_class`` now lives in :mod:`llmr.bridge.world_reader` alongside
the other SymbolGraph queries.
"""
from llmr.bridge.world_reader import resolve_symbol_class
from llmr.grounder import EntityGrounder, GroundingResult

__all__ = [
    "EntityGrounder",
    "GroundingResult",
    "resolve_symbol_class",
]
