"""
Capture the Jupyter execution protocol independently of displayed output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from nbformat import NotebookNode
    from typing_extensions import Any, Self

# %% execution reply schema


class ExecutionField(StrEnum):
    """
    The protocol fields needed to identify an interrupted execution.
    """

    CONTENT = "content"
    STATUS = "status"
    ERROR_NAME = "ename"


class ExecutionStatus(StrEnum):
    """
    The execution outcome checked by the interrupt regression.
    """

    ERROR = "error"


@dataclass
class ExecutionReply:
    """
    The protocol outcome without notebook display formatting.
    """

    status: str
    """
    The kernel's execution status.
    """

    exception_name: str
    """
    The reported exception, if any.
    """

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> Self:
        """
        Read the authoritative outcome from Jupyter reply content.
        """
        return cls(
            payload[ExecutionField.STATUS], payload.get(ExecutionField.ERROR_NAME, "")
        )


@dataclass
class ExecutionRecorder:
    """
    Retain the outcomes reported by notebook execution hooks.
    """

    replies: list[ExecutionReply] = field(default_factory=list)
    """
    The observed execution outcomes in cell order.
    """

    def record(
        self, cell: NotebookNode, cell_index: int, execute_reply: dict[str, Any]
    ) -> None:
        """
        Capture one reply after its cell has completed.
        """
        self.replies.append(
            ExecutionReply.from_json(execute_reply[ExecutionField.CONTENT])
        )
