"""Entity grounder — resolves NL entity descriptions to Symbol instances.

Resolution strategy
-------------------
Tier 1  Annotation-based: find instances whose *class name* matches the
        ``semantic_type`` string extracted by the LLM, then optionally
        cross-reference with the groundable type's ``.bodies`` attribute.

Tier 2  Name-based: duck-type access to ``.name`` / ``.name.name`` on every
        groundable instance, substring-matching the extracted name string.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing_extensions import Any, List, Optional, Tuple, Type

from krrood.symbol_graph.symbol_graph import Symbol, SymbolGraph

from krrood.llmr_decoupled.workflows.schemas.common import EntityDescriptionSchema

logger = logging.getLogger(__name__)


# ── Duck-typing helpers ────────────────────────────────────────────────────────
# These access attributes on unknown Symbol instances using plain getattr/duck
# typing. They are the ONLY place in krrood.llmr_decoupled that knows about the
# attribute-chain conventions of the downstream world representation.
# If sdt renames PrefixedName.name, Pose.to_position(), etc., fix it here.


def body_display_name(body: Any) -> str:
    """Return the display string for a Symbol instance (hides PrefixedName chain)."""
    name_obj = getattr(body, "name", None)
    if name_obj is None:
        return ""
    if hasattr(name_obj, "name"):
        return str(name_obj.name)
    return str(name_obj)


def body_xyz(body: Any) -> Optional[Tuple[float, float, float]]:
    """Return ``(x, y, z)`` for any object with a ``.global_pose`` property."""
    try:
        pt = body.global_pose.to_position()
        return float(pt.x), float(pt.y), float(pt.z)
    except Exception:
        return None


def body_bounding_box_dims(
    body: Any,
    reference_frame: Any = None,
) -> Optional[Tuple[float, float, float]]:
    """Return ``(w, d, h)`` bounding box dimensions via duck typing."""
    try:
        ref = reference_frame if reference_frame is not None else body
        bb = body.collision.as_bounding_box_collection_in_frame(ref).bounding_box()
        d = bb.dimensions
        return float(d[0]), float(d[1]), float(d[2])
    except Exception:
        return None


# ── Result type ────────────────────────────────────────────────────────────────


@dataclass
class GroundingResult:
    """Result of an entity grounding attempt."""

    bodies: List[Any] = field(default_factory=list)
    """Candidate Symbol instances that match the description, ranked by confidence."""

    warning: Optional[str] = None
    """Non-fatal diagnostic message (e.g. multiple matches, fallback used)."""


# ── Annotation class resolution via SymbolGraph ───────────────────────────────


def _camel_to_tokens(name: str) -> str:
    """``'DrinkingContainer'`` → ``'drinking container'``."""
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name).lower()


def resolve_symbol_class(semantic_type: str) -> Optional[Type[Symbol]]:
    """Resolve a semantic type string to a Symbol subclass via the SymbolGraph class diagram.

    Walks ``SymbolGraph().class_diagram`` — all Symbol subclasses are registered
    there at instantiation time, so no sdt import is needed.

    :param semantic_type: String from the LLM slot schema.
    :return: Matching Symbol subclass, or ``None`` if nothing found.
    """
    query = semantic_type.strip().lower()
    query_tokens = query.replace("_", " ").replace("-", " ")

    try:
        class_diagram = SymbolGraph().class_diagram
    except Exception:
        logger.debug("SymbolGraph not yet initialised — cannot resolve '%s'.", semantic_type)
        return None

    for wrapped_cls in class_diagram.wrapped_classes:
        cls = wrapped_cls.clazz
        # 1. exact class name
        if cls.__name__.lower() == query:
            return cls
        # 2. camel-case expanded
        if _camel_to_tokens(cls.__name__) == query_tokens:
            return cls
        # 3. _synonyms classvar (Set[str]) if present
        synonyms = getattr(cls, "_synonyms", set())
        if any(s.lower() == query_tokens for s in synonyms):
            return cls

    return None


# ── EntityGrounder ─────────────────────────────────────────────────────────────


class EntityGrounder:
    """Grounds an :class:`EntityDescriptionSchema` to Symbol instances in the world.

    Uses :class:`~krrood.symbol_graph.symbol_graph.SymbolGraph` as the sole
    data source — no world object or sdt import required.

    :param groundable_type: The Symbol subclass representing groundable world
        entities (e.g. ``Body`` from sdt).  Passed by the caller at setup time;
        krrood.llmr_decoupled never imports this class directly.
    """

    def __init__(self, groundable_type: Type[Symbol]) -> None:
        self._groundable_type = groundable_type

    # ── Main entry point ───────────────────────────────────────────────────────

    def ground(self, description: EntityDescriptionSchema) -> GroundingResult:
        """Resolve *description* to Symbol instances.

        Tries annotation grounding (Tier 1) first, then name-based (Tier 2).
        Applies spatial/attribute refinement when multiple candidates remain.

        :param description: LLM-extracted entity description.
        :return: :class:`GroundingResult` with matching instances and diagnostic info.
        """
        if description.semantic_type:
            result = self._annotation_ground(description)
            if result.bodies:
                logger.debug(
                    "Tier 1 grounding: semantic_type=%r → %d instance(s): %s",
                    description.semantic_type,
                    len(result.bodies),
                    [body_display_name(b) for b in result.bodies],
                )
                return result
            logger.debug(
                "Annotation grounding for type '%s' returned no results, "
                "falling back to name search.",
                description.semantic_type,
            )

        result = self._name_ground(description)
        if result.bodies:
            return result

        warning = (
            f"No instances found for '{description.name}' "
            f"(semantic_type={description.semantic_type!r}). "
            "Check that the object exists in the world."
        )
        logger.warning(warning)
        return GroundingResult(bodies=[], warning=warning)

    # ── Tier 1: annotation-based ───────────────────────────────────────────────

    def _annotation_ground(self, description: EntityDescriptionSchema) -> GroundingResult:
        """Ground via semantic annotation type resolved from SymbolGraph class diagram."""
        cls = resolve_symbol_class(description.semantic_type)
        if cls is None:
            logger.debug(
                "Cannot resolve '%s' to a Symbol subclass.", description.semantic_type
            )
            return GroundingResult()

        try:
            annotations = list(SymbolGraph().get_instances_of_type(cls))
        except Exception as exc:
            logger.warning("SymbolGraph.get_instances_of_type raised: %s", exc)
            return GroundingResult()

        if not annotations:
            return GroundingResult()

        # Collect groundable instances via the annotation's .bodies attribute (duck typing)
        candidates: List[Any] = []
        for ann in annotations:
            try:
                for body in ann.bodies:
                    if body not in candidates:
                        candidates.append(body)
            except AttributeError:
                # Annotation has no .bodies — treat the annotation itself as groundable
                if ann not in candidates:
                    candidates.append(ann)

        if description.name and candidates:
            name_lower = description.name.lower()
            name_filtered = [
                b for b in candidates if name_lower in body_display_name(b).lower()
            ]
            if name_filtered:
                candidates = name_filtered

        candidates = self._refine(candidates, description)
        return GroundingResult(
            bodies=candidates,
            warning=self._multi_match_warning(candidates, description.name),
        )

    # ── Tier 2: name-based ─────────────────────────────────────────────────────

    def _name_ground(self, description: EntityDescriptionSchema) -> GroundingResult:
        """Ground by name — substring scan over all groundable Symbol instances."""
        if not description.name:
            return GroundingResult()

        name_lower = description.name.lower()
        try:
            all_instances = list(SymbolGraph().get_instances_of_type(self._groundable_type))
        except Exception as exc:
            logger.warning("SymbolGraph.get_instances_of_type raised: %s", exc)
            return GroundingResult()

        candidates = [
            b for b in all_instances if name_lower in body_display_name(b).lower()
        ]

        if not candidates:
            return GroundingResult()

        candidates = self._refine(candidates, description)
        return GroundingResult(
            bodies=candidates,
            warning=self._multi_match_warning(candidates, description.name),
        )

    # ── Refinement ─────────────────────────────────────────────────────────────

    def _refine(
        self, candidates: List[Any], description: EntityDescriptionSchema
    ) -> List[Any]:
        """Apply spatial context and attribute filters to narrow candidates."""
        if description.spatial_context and len(candidates) > 1:
            refined = self._filter_by_spatial_context(candidates, description.spatial_context)
            if refined:
                candidates = refined

        if description.attributes and len(candidates) > 1:
            refined = self._filter_by_attributes(candidates, description.attributes)
            if refined:
                candidates = refined

        return candidates

    def _filter_by_spatial_context(
        self, candidates: List[Any], spatial_context: str
    ) -> List[Any]:
        """Narrow candidates using a spatial context hint via SymbolGraph class lookup."""
        context_lower = spatial_context.lower()

        # Try to find a surface annotation type by name from SymbolGraph class diagram
        surface_cls = resolve_symbol_class("HasSupportingSurface")
        if surface_cls is not None:
            try:
                surface_annotations = list(SymbolGraph().get_instances_of_type(surface_cls))
                matched_surfaces = [
                    ann for ann in surface_annotations
                    if _camel_to_tokens(type(ann).__name__) in context_lower
                    or context_lower in _camel_to_tokens(type(ann).__name__)
                ]
                if matched_surfaces:
                    proximity_filtered = [
                        c for c in candidates if self._near_any_surface(c, matched_surfaces)
                    ]
                    if proximity_filtered:
                        return proximity_filtered
            except Exception as exc:
                logger.debug("Surface-based spatial filter failed: %s", exc)

        # Fallback: anchor body name substring match via SymbolGraph
        try:
            all_instances = list(SymbolGraph().get_instances_of_type(self._groundable_type))
        except Exception:
            return candidates

        anchor_bodies = [
            b for b in all_instances
            if body_display_name(b).lower() in context_lower
        ]
        if not anchor_bodies:
            return candidates

        def _in_subtree(body: Any, anchor: Any) -> bool:
            current = body
            while current is not None:
                if current is anchor:
                    return True
                parent_conn = getattr(current, "parent_connection", None)
                current = getattr(parent_conn, "parent", None) if parent_conn else None
            return False

        tree_filtered = [
            c for c in candidates if any(_in_subtree(c, anchor) for anchor in anchor_bodies)
        ]
        return tree_filtered if tree_filtered else candidates

    @staticmethod
    def _near_any_surface(body: Any, surfaces: list) -> bool:
        """Return True if *body* is positioned above any of the *surfaces*."""
        try:
            xyz = body_xyz(body)
            if xyz is None:
                return True
            bz = xyz[2]
            for ann in surfaces:
                try:
                    ann_body = ann.bodies[0]
                    ann_xyz = body_xyz(ann_body)
                    dims = body_bounding_box_dims(ann_body)
                    if ann_xyz is not None and dims is not None:
                        surface_top_z = ann_xyz[2] + dims[2] / 2
                        if bz >= surface_top_z - 0.05:
                            return True
                except Exception:
                    continue
        except Exception:
            pass
        return False

    def _filter_by_attributes(
        self, candidates: List[Any], attributes: dict
    ) -> List[Any]:
        """Filter by key/value attributes from the entity description."""
        filtered = []
        for body in candidates:
            body_str = body_display_name(body).lower()
            ann_type_names = " ".join(
                type(a).__name__.lower()
                for a in getattr(body, "_semantic_annotations", [])
            )
            combined = body_str + " " + ann_type_names
            for value in attributes.values():
                if value.lower() in combined:
                    filtered.append(body)
                    break
        return filtered if filtered else candidates

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _multi_match_warning(candidates: List[Any], name: Optional[str]) -> Optional[str]:
        if len(candidates) > 1:
            names = [body_display_name(b) for b in candidates]
            return (
                f"Grounding for '{name}' returned {len(candidates)} candidates: "
                f"{names}. All passed to the action handler."
            )
        return None


# ── Module-level convenience ───────────────────────────────────────────────────


def ground_entity(
    description: EntityDescriptionSchema,
    groundable_type: Type[Symbol],
) -> GroundingResult:
    """Convenience wrapper around :class:`EntityGrounder`.

    :param description: LLM-extracted entity description.
    :param groundable_type: The Symbol subclass to search in SymbolGraph.
    :return: :class:`GroundingResult`.
    """
    return EntityGrounder(groundable_type).ground(description)
