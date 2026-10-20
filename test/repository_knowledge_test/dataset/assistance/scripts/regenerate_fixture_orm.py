"""
A generator consumes source declarations rather than generated mappings.
"""

from first.base import Same

# %% generator inputs


def build_mapping() -> type:
    """
    Expose the source declaration used by a generator.
    """
    return Same
