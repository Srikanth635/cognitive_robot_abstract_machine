"""
Consume a public re-export from a separate source file.
"""

from first import PublicBase

# %% exported base


class Consumer(PublicBase[int]):
    """
    Retain source evidence for an exported generic base.
    """
