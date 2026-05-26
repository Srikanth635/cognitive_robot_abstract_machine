"""Centralized manager for the active SDT world and robot view."""

from typing import Any, Callable, List, Tuple

_ACTIVE_WORLD: Any = None
_ACTIVE_ROBOT_VIEW: Any = None

# Callables registered by other modules via register_world_cache().
# All are invoked by set_active_world() so stale caches never outlive a world switch.
_WORLD_CACHE_CLEARERS: List[Callable[[], None]] = []


def register_world_cache(clear_fn: Callable[[], None]) -> None:
    """Register a cache-clear callable to be called on every set_active_world()."""
    _WORLD_CACHE_CLEARERS.append(clear_fn)


def set_active_world(world: Any, robot_view: Any) -> None:
    """Store references to the active SDT world and robot view, then clear world caches."""
    global _ACTIVE_WORLD, _ACTIVE_ROBOT_VIEW
    _ACTIVE_WORLD = world
    _ACTIVE_ROBOT_VIEW = robot_view
    for clear_fn in _WORLD_CACHE_CLEARERS:
        try:
            clear_fn()
        except Exception:
            pass


def get_active_world() -> Tuple[Any, Any]:
    """Return (world, robot_view) for the active SDT world.

    Raises:
        RuntimeError: If no world has been initialised yet.
    """
    if _ACTIVE_WORLD is None:
        raise RuntimeError(
            "No active world is set. Call set_active_world() after loading the environment."
        )
    return _ACTIVE_WORLD, _ACTIVE_ROBOT_VIEW
