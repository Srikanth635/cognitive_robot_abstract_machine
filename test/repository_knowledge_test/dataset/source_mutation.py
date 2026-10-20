"""
The source changes checked against clean rebuilds.
"""

from enum import StrEnum

# %% source mutation cases


class SourceMutation(StrEnum):
    """
    The editing situations requiring equivalent refreshed evidence.
    """

    MODIFIED = "edit"
    DELETED = "delete"
    RENAMED = "rename"
    STAGED_ADDITION = "staged_addition"
    UNTRACKED_ADDITION = "untracked_addition"
    INVALID_SYNTAX = "syntax_failure"
    CHANGED_SOURCE_ROOTS = "source_roots"
    DIFFERENT_BRANCH = "branch"
