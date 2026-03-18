"""Entity grounding: EntityDescriptionSchema → List[Body].

Bridges the LLM's symbolic entity description and the concrete ``Body`` objects
that live in the Semantic Digital Twin world.

## Three-tier grounding strategy

### Tier 1 – Semantic annotation type (highest priority)
When the LLM provides a ``semantic_type`` (e.g. "Milk", "Cup"), it is resolved
to a concrete ``SemanticAnnotation`` subclass by scanning the SDT annotation
registry.  ``world.get_semantic_annotations_by_type(cls)`` is then called and
``annotation.bodies`` extracts the root ``Body`` objects.  This is the most
semantically correct path: it uses the world's own ontology rather than
string matching.

### Tier 2 – Body name substring match
If no semantic_type is provided, or Tier 1 returns no results, a substring
search over ``world.bodies`` by the LLM-extracted name is performed.

### Tier 3 – Spatial / attribute refinement (cross-tier)
When multiple candidates survive Tier 1 or Tier 2, ``spatial_context`` and
``attributes`` from the entity description are used to narrow the list:
  - Spatial context is matched against ``HasSupportingSurface`` annotations
    (Table, Counter, etc.) and candidate bodies are filtered by proximity to
    that surface's bounding box.
  - Attributes are matched against body names and annotation type names.
"""

from __future__ import annotations

import inspect
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Type

from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    HasSupportingSurface,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import (
    Body,
    SemanticAnnotation,
)

from .workflows.pydantics.pick_up_schemas import EntityDescriptionSchema

logger = logging.getLogger(__name__)


# ── Public result type ─────────────────────────────────────────────────────────


@dataclass
class GroundingResult:
    """Result of an entity grounding attempt."""

    bodies: List[Body] = field(default_factory=list)
    """Candidate Body objects that match the description, ranked by confidence."""

    tier: str = "none"
    """Which grounding tier succeeded: 'annotation', 'name', 'fuzzy', or 'none'."""

    # kept for backwards compatibility
    @property
    def used_eql(self) -> bool:
        return False

    warning: Optional[str] = None
    """Non-fatal diagnostic message (e.g. multiple matches, fallback used)."""


# ── Annotation class registry ──────────────────────────────────────────────────


def _all_annotation_subclasses() -> List[Type[SemanticAnnotation]]:
    """Return all concrete (non-abstract) SemanticAnnotation subclasses."""
    result: List[Type[SemanticAnnotation]] = []

    def _recurse(cls: type) -> None:
        for sub in cls.__subclasses__():
            if not inspect.isabstract(sub):
                result.append(sub)
            _recurse(sub)

    _recurse(SemanticAnnotation)
    return result


def _camel_to_tokens(name: str) -> str:
    """'DrinkingContainer' → 'drinking container'."""
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name).lower()


def resolve_annotation_class(semantic_type: str) -> Optional[Type[SemanticAnnotation]]:
    """Resolve a semantic type string to a SemanticAnnotation subclass.

    Matching order (all case-insensitive):
      1. Exact class name match  (``Milk`` → ``Milk``)
      2. CamelCase-expanded match  (``DrinkingContainer`` → ``"drinking container"``)
      3. ``_synonyms`` classvar  (user-defined aliases on annotation classes)

    :param semantic_type: String from the LLM slot schema.
    :return: Matching class, or ``None`` if nothing found.
    """
    query = semantic_type.strip().lower()
    query_tokens = query.replace("_", " ").replace("-", " ")

    for cls in _all_annotation_subclasses():
        # 1. exact class name
        if cls.__name__.lower() == query:
            return cls
        # 2. camel-case expanded
        if _camel_to_tokens(cls.__name__) == query_tokens:
            return cls
        # 3. _synonyms classvar (Set[str] on the class)
        synonyms = getattr(cls, "_synonyms", set())
        if any(s.lower() == query_tokens for s in synonyms):
            return cls

    return None


# ── EntityGrounder ─────────────────────────────────────────────────────────────


class EntityGrounder:
    """Grounds an ``EntityDescriptionSchema`` to ``Body`` objects in the world.

    :param world: The Semantic Digital Twin world instance.
    """

    def __init__(self, world: World) -> None:
        self._world = world

    # ── Main entry point ───────────────────────────────────────────────────────

    def ground(self, description: EntityDescriptionSchema) -> GroundingResult:
        """Resolve an entity description to world bodies.

        Tries annotation grounding (Tier 1) first, then name-based (Tier 2).
        Applies spatial / attribute refinement when multiple candidates remain.

        :param description: LLM-extracted entity description.
        :return: ``GroundingResult`` with matching bodies and diagnostic info.
        """
        # ── Tier 1: semantic annotation type ──────────────────────────────────
        if description.semantic_type:
            result = self._annotation_ground(description)
            if result.bodies:
                return result
            logger.debug(
                "Annotation grounding for type '%s' returned no results, "
                "falling back to name search.",
                description.semantic_type,
            )

        # ── Tier 2: body name substring ───────────────────────────────────────
        result = self._name_ground(description)
        if result.bodies:
            return result

        # ── Nothing found ─────────────────────────────────────────────────────
        warning = (
            f"No bodies found for '{description.name}' "
            f"(semantic_type={description.semantic_type!r}). "
            "Check that the object exists in the world."
        )
        logger.warning(warning)
        return GroundingResult(bodies=[], tier="none", warning=warning)

    # ── Tier 1: annotation-based grounding ────────────────────────────────────

    def _annotation_ground(
        self, description: EntityDescriptionSchema
    ) -> GroundingResult:
        """Ground via SDT semantic annotation type.

        Resolves ``description.semantic_type`` to a ``SemanticAnnotation``
        subclass, queries the world for all annotations of that type, and
        extracts their root bodies.  Optionally refines by name, spatial
        context, and attributes.
        """
        cls = resolve_annotation_class(description.semantic_type)
        if cls is None:
            logger.debug(
                "Cannot resolve '%s' to a SemanticAnnotation subclass.",
                description.semantic_type,
            )
            return GroundingResult()

        try:
            annotations = self._world.get_semantic_annotations_by_type(cls)
        except Exception as exc:
            logger.warning("get_semantic_annotations_by_type raised: %s", exc)
            return GroundingResult()

        if not annotations:
            return GroundingResult()

        # Collect bodies from all matching annotations
        candidates: List[Body] = []
        for ann in annotations:
            for body in ann.bodies:
                if body not in candidates:
                    candidates.append(body)

        # Optional name sub-filter (keep if name appears in body name)
        if description.name and candidates:
            name_lower = description.name.lower()
            name_filtered = [
                b for b in candidates
                if name_lower in self._body_name(b).lower()
            ]
            if name_filtered:
                candidates = name_filtered

        # Spatial / attribute refinement
        candidates = self._refine(candidates, description)

        return GroundingResult(
            bodies=candidates,
            tier="annotation",
            warning=self._multi_match_warning(candidates, description.name),
        )

    # ── Tier 2: name-based grounding ──────────────────────────────────────────

    def _name_ground(self, description: EntityDescriptionSchema) -> GroundingResult:
        """Ground by substring-matching the entity name over all world bodies."""
        if not description.name:
            return GroundingResult()

        name_lower = description.name.lower()
        candidates = [
            b for b in self._world.bodies
            if name_lower in self._body_name(b).lower()
        ]

        if not candidates:
            return GroundingResult()

        candidates = self._refine(candidates, description)
        return GroundingResult(
            bodies=candidates,
            tier="name",
            warning=self._multi_match_warning(candidates, description.name),
        )

    # ── Cross-tier refinement ─────────────────────────────────────────────────

    def _refine(
        self, candidates: List[Body], description: EntityDescriptionSchema
    ) -> List[Body]:
        """Apply spatial context and attribute filters to narrow candidates."""
        if description.spatial_context and len(candidates) > 1:
            refined = self._filter_by_spatial_context(
                candidates, description.spatial_context
            )
            if refined:
                candidates = refined

        if description.attributes and len(candidates) > 1:
            refined = self._filter_by_attributes(candidates, description.attributes)
            if refined:
                candidates = refined

        return candidates

    # ── Spatial context filter ─────────────────────────────────────────────────

    def _filter_by_spatial_context(
        self, candidates: List[Body], spatial_context: str
    ) -> List[Body]:
        """Narrow candidates using a spatial context hint.

        Strategy:
          1. Resolve the spatial context to a ``HasSupportingSurface``
             annotation (e.g. "table" → Table, "counter" → Counter).
          2. Filter candidates whose z-position is at or above the supporting
             surface's bounding box top face (within a 0.3m tolerance).
          3. Fall back to kinematic-tree parent matching if no surface found.
        """
        context_lower = spatial_context.lower()

        # --- SDT surface-based filtering ---
        try:
            surface_annotations = self._world.get_semantic_annotations_by_type(
                HasSupportingSurface
            )
            matched_surfaces = [
                ann for ann in surface_annotations
                if _camel_to_tokens(type(ann).__name__) in context_lower
                or context_lower in _camel_to_tokens(type(ann).__name__)
            ]
            if matched_surfaces:
                surface_bodies: List[Body] = []
                for ann in matched_surfaces:
                    surface_bodies.extend(ann.bodies)

                # Keep candidates that are positioned near any matched surface
                proximity_filtered = [
                    c for c in candidates
                    if self._near_any_surface(c, matched_surfaces)
                ]
                if proximity_filtered:
                    return proximity_filtered
        except Exception as exc:
            logger.debug("Surface-based spatial filter failed: %s", exc)

        # --- Fallback: kinematic-tree parent matching ---
        anchor_bodies = [
            b for b in self._world.bodies
            if self._body_name(b).lower() in context_lower
        ]
        if not anchor_bodies:
            return candidates

        def _in_subtree(body: Body, anchor: Body) -> bool:
            current = body
            while current is not None:
                if current is anchor:
                    return True
                parent_conn = getattr(current, "parent_connection", None)
                current = getattr(parent_conn, "parent", None) if parent_conn else None
            return False

        tree_filtered = [
            c for c in candidates
            if any(_in_subtree(c, anchor) for anchor in anchor_bodies)
        ]
        return tree_filtered if tree_filtered else candidates

    def _near_any_surface(
        self, body: Body, surfaces: list
    ) -> bool:
        """Return True if *body* is positioned above any of the *surfaces*."""
        try:
            from .hybrid_resolver import _pose_to_xyz  # local import to avoid circular
            body_xyz = _pose_to_xyz(body.global_pose)
            if body_xyz is None:
                return True  # cannot determine — keep candidate
            bz = body_xyz[2]

            for ann in surfaces:
                try:
                    bb = ann.as_bounding_box_collection_in_frame(
                        self._world.root
                    ).bounding_box()
                    dims = bb.dimensions
                    surface_top_z = float(
                        _pose_to_xyz(ann.bodies[0].global_pose)[2]
                    ) + dims[2] / 2
                    if bz >= surface_top_z - 0.05:  # 5cm tolerance
                        return True
                except Exception:
                    continue
        except Exception:
            pass
        return False

    # ── Attribute filter ───────────────────────────────────────────────────────

    def _filter_by_attributes(
        self, candidates: List[Body], attributes: dict
    ) -> List[Body]:
        """Filter by key/value attributes from the entity description.

        A body is retained if ANY attribute value appears in the body name or
        annotation type names — intentionally lenient to avoid over-filtering.
        """
        filtered = []
        for body in candidates:
            body_str = self._body_name(body).lower()
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
    def _body_name(body: Body) -> str:
        """Return a normalised string name for a Body."""
        name_obj = getattr(body, "name", None)
        if name_obj is None:
            return ""
        if hasattr(name_obj, "name"):
            return str(name_obj.name)
        return str(name_obj)

    @staticmethod
    def _multi_match_warning(candidates: List[Body], name: str) -> Optional[str]:
        if len(candidates) > 1:
            names = [
                str(getattr(getattr(b, "name", None), "name", b))
                for b in candidates
            ]
            return (
                f"Grounding for '{name}' returned {len(candidates)} candidates: "
                f"{names}. All passed to PartialDesignator."
            )
        return None


# ── Module-level convenience function ─────────────────────────────────────────


def ground_entity(
    description: EntityDescriptionSchema,
    world: World,
) -> GroundingResult:
    """Convenience wrapper around ``EntityGrounder.ground``.

    :param description: LLM-extracted entity description.
    :param world: The SDT world instance.
    :return: ``GroundingResult``.
    """
    return EntityGrounder(world).ground(description)
