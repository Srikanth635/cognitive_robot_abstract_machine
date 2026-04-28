"""Gateway package: the only modules that touch krrood directly.

Other llmr modules import plain data structures from here and stay krrood-free.

  introspect   — classify action dataclass fields into :class:`FieldKind`.
  world_reader — read SymbolGraph contents and resolve Symbol classes by name.
  match_reader — snapshot krrood Match expressions into :class:`MatchSnapshot` / :class:`MatchField`.
"""

from llmr.bridge.introspect import (
    ActionSpec,
    DiscoveredField,
    FieldKind,
    DeclaredFieldsIntrospector,
    ActionFieldIntrospector,
    introspect_action,
)
from llmr.bridge.match_reader import (
    MatchSnapshot,
    MatchField,
    snapshot_match,
    bind_slot_value,
    construct_action,
    underspecified_match,
    missing_required_fields,
    render_resolved_slots,
)
from llmr.bridge.world_reader import (
    WorldContextConfig,
    symbol_display_name,
    symbol_xyz,
    symbol_bounding_box,
    render_world_context,
    get_instances,
    resolve_symbol_class,
)

__all__ = [
    "ActionSpec",
    "DiscoveredField",
    "FieldKind",
    "DeclaredFieldsIntrospector",
    "ActionFieldIntrospector",
    "introspect_action",
    "MatchSnapshot",
    "MatchField",
    "snapshot_match",
    "bind_slot_value",
    "construct_action",
    "underspecified_match",
    "missing_required_fields",
    "render_resolved_slots",
    "WorldContextConfig",
    "symbol_display_name",
    "symbol_xyz",
    "symbol_bounding_box",
    "render_world_context",
    "get_instances",
    "resolve_symbol_class",
]
