"""NL → ActionSpec pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing_extensions import Any, Callable, Dict, Optional, Type

from krrood.symbol_graph.symbol_graph import Symbol, SymbolGraph

from krrood.llmr.pipeline.dispatcher import ActionDispatcher, ActionSpec
from krrood.llmr.pipeline.entity_grounder import EntityGrounder, body_display_name
from krrood.llmr.workflows.nodes.slot_filler import ActionSlotSchema, run_slot_filler

logger = logging.getLogger(__name__)


# ── World serialiser ───────────────────────────────────────────────────────────


def _is_structural_link(name: str) -> bool:
    """Return True if *name* looks like a robot kinematic link (not a scene object)."""
    _SUFFIXES = (
        "_link", "_frame", "_joint", "_screw", "_plate",
        "_optical_frame", "_motor", "_pad", "_finger",
    )
    return any(name.endswith(s) for s in _SUFFIXES)


def serialise_world_from_symbol_graph(
    groundable_type: Type[Symbol],
    extra_context: str = "",
) -> str:
    """Build an LLM world-context string from :class:`SymbolGraph` contents.

    :param groundable_type: Symbol subclass representing scene objects (e.g. Body).
    :param extra_context: Optional additional context string appended at the end
    :return: Multi-line string describing the current world state.
    """
    lines = ["## World State Summary\n"]

    try:
        all_instances = list(SymbolGraph().get_instances_of_type(groundable_type))
        all_names = [body_display_name(b) for b in all_instances]
        scene_names = [n for n in all_names if not _is_structural_link(n)]
        if scene_names:
            lines.append(f"Scene objects and surfaces: {', '.join(scene_names)}")
        elif all_names:
            lines.append(f"Bodies present: {', '.join(all_names[:30])}")
            if len(all_names) > 30:
                lines.append(f"  … and {len(all_names) - 30} more.")
        else:
            lines.append("No scene objects found in SymbolGraph.")
    except Exception:
        lines.append("Bodies: unavailable")

    lines.append("\n## Semantic annotations")
    try:
        ann_summary: Dict[str, list] = {}
        # Walk all Symbol instances that have a .bodies attribute (duck typing)
        graph = SymbolGraph()
        for wrapped in graph.wrapped_instances:
            inst = wrapped.instance
            if inst is None:
                continue
            bodies_attr = getattr(inst, "bodies", None)
            if bodies_attr is None:
                continue
            ann_type = type(inst).__name__
            try:
                for body in bodies_attr:
                    b_name = body_display_name(body)
                    if _is_structural_link(b_name):
                        continue
                    ann_summary.setdefault(b_name, []).append(ann_type)
            except Exception:
                pass

        if ann_summary:
            unique_types = sorted({t for types in ann_summary.values() for t in types})
            lines.append(f"Available types: {', '.join(unique_types)}")
            lines.append("Per body:")
            for body_name, types in ann_summary.items():
                lines.append(f"  {body_name}: {', '.join(types)}")
        else:
            lines.append("  None found in this world.")
    except Exception:
        lines.append("  (unavailable)")

    if extra_context:
        lines.append(extra_context)

    return "\n".join(lines)


# ── ActionPipeline ─────────────────────────────────────────────────────────────


@dataclass
class ActionPipeline:
    """Universal NL → :class:`ActionSpec` pipeline.

    :param groundable_type: Symbol subclass to query from SymbolGraph (e.g. Body).
    :param context: Caller-supplied context dict forwarded to every ActionHandler.
    :param world_context_provider: Optional callable returning a world-context
        string.  When provided, replaces the SymbolGraph-based serializer.
    :param action_types: Dict of ``{action_type_str: description}`` injected into
        the slot-filler LLM prompt so it knows what actions are supported.
    """

    groundable_type: Type[Symbol]
    context: Dict[str, Any] = field(default_factory=dict)
    world_context_provider: Optional[Callable[[], str]] = field(default=None)
    action_types: Dict[str, str] = field(default_factory=dict)
    action_schemas: Optional[list] = field(default=None)  # List[ActionSchema] from pycram bridge

    _dispatcher: ActionDispatcher = field(init=False)

    def __post_init__(self) -> None:
        grounder = EntityGrounder(self.groundable_type)
        self._dispatcher = ActionDispatcher(grounder=grounder, context=self.context)

    # ── Main entry point ───────────────────────────────────────────────────────

    def run(self, instruction: str) -> ActionSpec:
        """Execute the full pipeline for *instruction*.

        :param instruction: Natural language instruction.
        :return: Fully specified `:class:``ActionSpec`.
        :raises RuntimeError: On unrecoverable failures in any stage.
        """
        logger.info("ActionPipeline.run: '%s'", instruction)
        schema = self.classify_and_extract(instruction)
        if schema is None:
            raise RuntimeError("Slot-filler failed. Check LLM connectivity and API keys.")
        return self.dispatch(schema)

    # ── Step-by-step accessors ─────────────────────────────────────────────────

    def classify_and_extract(self, instruction: str) -> Optional[ActionSlotSchema]:
        """Phase 1: NL instruction → typed slot schema.

        :return: :class:`ActionSlotSchema` on success; ``None`` on failure.
        """
        world_ctx_str = self._get_world_context()
        logger.debug("World context for slot filling:\n%s", world_ctx_str)
        schema = run_slot_filler(
            instruction=instruction,
            world_context=world_ctx_str,
            action_types=self.action_types,
            action_schemas=self.action_schemas,
        )
        if schema is not None:
            entity_names = [e.name for e in schema.entities]
            logger.info(
                "classify_and_extract – action_type=%s, entities=%s",
                schema.action_type,
                entity_names,
            )
        return schema

    def dispatch(self, schema: ActionSlotSchema) -> ActionSpec:
        """Phase 2: typed slot schema → :class:`ActionSpec`.

        :param schema: Output of :meth:`classify_and_extract`.
        :return: Fully specified :class:`ActionSpec`.
        """
        spec = self._dispatcher.dispatch(schema)
        logger.info("ActionPipeline.dispatch complete – %s resolved.", schema.action_type)
        return spec

    # ── Internal ───────────────────────────────────────────────────────────────

    def _get_world_context(self) -> str:
        """Return world context string from provider or SymbolGraph."""
        if self.world_context_provider is not None:
            try:
                return self.world_context_provider()
            except Exception as exc:
                logger.warning("world_context_provider raised %s — falling back.", exc)
        return serialise_world_from_symbol_graph(self.groundable_type)
