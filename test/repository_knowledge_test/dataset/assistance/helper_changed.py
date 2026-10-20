"""
A changed public callable for contract comparison.
"""

# %% changed declaration


def helper(argument: int, *, multiplier: int = 1) -> int:
    """
    Apply a declared multiplier.
    """
    return argument * multiplier
