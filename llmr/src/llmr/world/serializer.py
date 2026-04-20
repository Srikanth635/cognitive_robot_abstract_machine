"""Backward-compatible re-exports. Canonical home: :mod:`llmr.bridge.world_reader`."""
from llmr.bridge.world_reader import (
    WorldSerializationOptions,
    body_bounding_box,
    body_display_name,
    body_xyz,
    serialize_world_from_symbol_graph,
)

__all__ = [
    "WorldSerializationOptions",
    "body_bounding_box",
    "body_display_name",
    "body_xyz",
    "serialize_world_from_symbol_graph",
]
