from dataclasses import dataclass

from ...world import World


@dataclass
class WorldFetcher:
    """
    A ros service that allows other processes to fetch the entire world modification list from this world.
    The modification list is sent via a JSON string message.
    """

    world: World
