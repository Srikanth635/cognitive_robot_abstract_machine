"""PyCRAM Body Resolver
======================

Provides callable ``BodyResolver`` implementations that bind the symbolic
``CRAMEntityInfo`` objects produced by the CRAM parser to live PyCRAM
``Body`` world objects at runtime.

A *body resolver* is any callable with the signature::

    (entity: CRAMEntityInfo) -> Optional[Body]

The ``CRAMToPyCRAMSerializer.to_partial_designator()`` method accepts one as
its ``body_resolver`` argument.

Implementations provided here
-------------------------------
``pycram_body_resolver``
    Live resolver.  Uses PyCRAM's ``ObjectDesignatorDescription``
    (``BelieveObject``) to look up bodies in the currently active
    ``DesignatorDescription`` world by name, then falls back to a
    semantic-type search.

``make_world_body_resolver(world)``
    Factory that creates a resolver bound to a *specific* PyCRAM world
    object, bypassing the global world singleton.  Useful in multi-world
    setups or testing.

``make_name_map_resolver(name_map)``
    Factory that creates a resolver backed by a plain Python ``dict``
    mapping name strings to ``Body`` objects.  Useful for unit tests,
    batch offline processing, or simulation stubs.

``ChainedBodyResolver``
    Composes multiple resolvers in priority order — tries each in turn
    until one returns a non-``None`` result.

Usage example
-------------
::

    from llmr.serializers.body_resolver import (
        pycram_body_resolver,
        make_name_map_resolver,
        ChainedBodyResolver,
    )
    from llmr.serializers import CRAMToPyCRAMSerializer

    ser = CRAMToPyCRAMSerializer()

    # --- With a live PyCRAM world (robot execution) ---
    partial = ser.serialize(
        cram_string,
        body_resolver=pycram_body_resolver,
    )
    partial.perform()

    # --- With an injected world object ---
    from pycram.datastructures.enums import Arms
    resolver = make_world_body_resolver(my_world)
    partial = ser.serialize(cram_string, body_resolver=resolver, arm=Arms.LEFT)

    # --- Offline / test mode ---
    from unittest.mock import MagicMock
    cup_body  = MagicMock(name="cup")
    table_body = MagicMock(name="table")
    resolver = make_name_map_resolver({"cup": cup_body, "table": table_body})
    partial = ser.serialize(cram_string, body_resolver=resolver)

    # --- Chain: prefer world lookup, fall back to name map ---
    fallback_map = make_name_map_resolver({"cup": cup_body})
    resolver = ChainedBodyResolver(pycram_body_resolver, fallback_map)
    partial = ser.serialize(cram_string, body_resolver=resolver)
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Optional, Any

from .cram_to_pycram import CRAMEntityInfo

logger = logging.getLogger(__name__)

# Type alias — mirrors the one in cram_to_pycram for convenience
BodyResolver = Callable[[CRAMEntityInfo], Optional[Any]]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _candidate_names(entity: CRAMEntityInfo) -> list[str]:
    """Return a prioritised list of name candidates for *entity*.

    Order: tag first (most specific), then semantic_type.
    """
    names = []
    if entity.tag:
        names.append(entity.tag)
    if entity.semantic_type and entity.semantic_type != entity.tag:
        names.append(entity.semantic_type)
    return names


# ─────────────────────────────────────────────────────────────────────────────
# 1. Live PyCRAM resolver — uses ObjectDesignatorDescription / BelieveObject
# ─────────────────────────────────────────────────────────────────────────────

def pycram_body_resolver(entity: CRAMEntityInfo) -> Optional[Any]:
    """Resolve *entity* to a PyCRAM ``Body`` using the active world.

    Resolution strategy
    -------------------
    1. Try ``ObjectDesignatorDescription(names=[name]).ground()`` for each
       candidate name (tag, then semantic_type).
    2. If no name matches, scan ``world.bodies`` for a body whose
       ``semantic_annotations`` contain a type matching the semantic_type
       string (best-effort, case-insensitive substring match).
    3. Return ``None`` if nothing is found (the mapper treats ``None``
       gracefully for optional roles).

    Parameters
    ----------
    entity:
        Symbolic entity description from the CRAM parser.

    Returns
    -------
    Body or None
    """
    if not entity:
        return None

    candidates = _candidate_names(entity)
    if not candidates:
        return None

    try:
        from pycram.designator import ObjectDesignatorDescription
    except ImportError:
        logger.warning(
            "pycram not installed — cannot resolve entity '%s' to a Body", entity
        )
        return None

    # ── Strategy 1: resolve by name ───────────────────────────────────────
    for name in candidates:
        try:
            desc = ObjectDesignatorDescription(names=[name])
            body = desc.ground()  # returns first matching Body
            if body is not None:
                logger.debug("Resolved '%s' → %r (by name)", name, body)
                return body
        except Exception:
            pass  # no match for this name

    # ── Strategy 2: substring match on tag (e.g. "milk" matches "milk.stl") ─
    try:
        from pycram.designator import ObjectDesignatorDescription
        desc = ObjectDesignatorDescription()
        for name in candidates:
            name_lower = name.lower()
            for body in desc:
                body_name = (getattr(body, "name", "") or "").lower()
                if name_lower in body_name or body_name in name_lower:
                    logger.debug(
                        "Resolved '%s' → %r (substring match)", name, body
                    )
                    return body
    except Exception:
        pass

    # ── Strategy 3: scan world bodies by semantic type (fuzzy) ────────────
    sem_type = entity.semantic_type
    if sem_type:
        try:
            from pycram.designator import ObjectDesignatorDescription
            desc = ObjectDesignatorDescription()  # no name filter → all bodies
            sem_lower = sem_type.lower()
            for body in desc:
                body_name = getattr(body, "name", "") or ""
                if sem_lower in body_name.lower():
                    logger.debug(
                        "Resolved '%s' → %r (by semantic-type fuzzy match)", sem_type, body
                    )
                    return body
        except Exception:
            pass

    logger.debug("Could not resolve entity '%s' to any Body in the world.", entity)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 2. World-injected resolver factory
# ─────────────────────────────────────────────────────────────────────────────

def make_world_body_resolver(world: Any) -> BodyResolver:
    """Return a resolver bound to *world* instead of the global world singleton.

    Parameters
    ----------
    world:
        A PyCRAM world object (e.g. ``BulletWorld``, ``Multiverse``).
        The resolver iterates ``world.bodies`` directly.

    Returns
    -------
    BodyResolver
        A callable ``(CRAMEntityInfo) -> Optional[Body]``.
    """
    def _resolver(entity: CRAMEntityInfo) -> Optional[Any]:
        if not entity:
            return None
        candidates = _candidate_names(entity)
        if not candidates:
            return None

        # ── Match by name ─────────────────────────────────────────────────
        for name in candidates:
            for body in getattr(world, "bodies", []):
                raw = getattr(body, "name", "")
                # PrefixedName: use local part only ("countertop" not "apartment/countertop")
                body_name = str(raw.name) if hasattr(raw, "name") else str(raw)
                if body_name == name or body_name.lower() == name.lower():
                    logger.debug(
                        "[world-resolver] Resolved '%s' → %r", name, body
                    )
                    return body

        # ── Substring match on tag (e.g. "milk" matches "milk.stl") ───────
        for name in candidates:
            name_lower = name.lower()
            for body in getattr(world, "bodies", []):
                raw = getattr(body, "name", "")
                body_name = (str(raw.name) if hasattr(raw, "name") else str(raw)).lower()
                if name_lower in body_name or body_name in name_lower:
                    logger.debug(
                        "[world-resolver] Resolved '%s' (substring) → %r", name, body
                    )
                    return body

        # ── Normalized name match (strip underscores/hyphens) ─────────────
        # Handles cases like "counter_top" matching "countertop"
        for name in candidates:
            normalized = name.replace("_", "").replace("-", "").lower()
            for body in getattr(world, "bodies", []):
                raw = getattr(body, "name", "")
                body_name = (str(raw.name) if hasattr(raw, "name") else str(raw)).lower()
                body_normalized = body_name.replace("_", "").replace("-", "")
                if normalized == body_normalized or normalized in body_normalized or body_normalized in normalized:
                    logger.debug(
                        "[world-resolver] Resolved '%s' (normalized) → %r", name, body
                    )
                    return body

        # ── Fuzzy semantic-type match ──────────────────────────────────────
        if entity.semantic_type:
            sem_lower = entity.semantic_type.lower()
            for body in getattr(world, "bodies", []):
                raw = getattr(body, "name", "")
                body_name = (str(raw.name) if hasattr(raw, "name") else str(raw)).lower()
                if sem_lower in body_name:
                    logger.debug(
                        "[world-resolver] Resolved '%s' (fuzzy) → %r",
                        entity.semantic_type, body,
                    )
                    return body

        logger.debug(
            "[world-resolver] Could not resolve '%s' in provided world.", entity
        )
        return None

    _resolver.__doc__ = (
        f"Body resolver bound to world {world!r}. "
        "Resolves by exact name, then fuzzy semantic type."
    )
    return _resolver


# ─────────────────────────────────────────────────────────────────────────────
# 3. Static name-map resolver factory (testing / offline)
# ─────────────────────────────────────────────────────────────────────────────

def make_name_map_resolver(name_map: Dict[str, Any]) -> BodyResolver:
    """Return a resolver backed by a static ``{name: Body}`` dict.

    Lookup order: tag → semantic_type → case-insensitive tag → ``None``.

    Parameters
    ----------
    name_map:
        Mapping from object name strings to Body objects.
        Keys are matched case-sensitively first, then case-insensitively.

    Returns
    -------
    BodyResolver

    Example
    -------
    ::

        resolver = make_name_map_resolver({
            "cup":   my_cup_body,
            "table": my_table_body,
            "knife": my_knife_body,
        })
    """
    _lower_map = {k.lower(): v for k, v in name_map.items()}

    def _resolver(entity: CRAMEntityInfo) -> Optional[Any]:
        if not entity:
            return None
        for name in _candidate_names(entity):
            # Exact match
            if name in name_map:
                return name_map[name]
            # Case-insensitive fallback
            if name.lower() in _lower_map:
                return _lower_map[name.lower()]
        return None

    _resolver.__doc__ = f"Static name-map resolver over {list(name_map.keys())}."
    return _resolver


# ─────────────────────────────────────────────────────────────────────────────
# 4. Chained resolver — tries multiple resolvers in priority order
# ─────────────────────────────────────────────────────────────────────────────

class ChainedBodyResolver:
    """Tries each resolver in order, returning the first non-``None`` result.

    Parameters
    ----------
    *resolvers:
        One or more ``BodyResolver`` callables, tried left-to-right.

    Example
    -------
    ::

        resolver = ChainedBodyResolver(
            pycram_body_resolver,           # live world first
            make_name_map_resolver(stubs),  # fall back to stubs
        )
    """

    def __init__(self, *resolvers: BodyResolver) -> None:
        self._resolvers = resolvers

    def __call__(self, entity: CRAMEntityInfo) -> Optional[Any]:
        for resolver in self._resolvers:
            result = resolver(entity)
            if result is not None:
                return result
        return None

    def __repr__(self) -> str:
        return f"ChainedBodyResolver({', '.join(repr(r) for r in self._resolvers)})"
