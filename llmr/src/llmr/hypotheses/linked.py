"""GraphLinked mixin — attribute-based graph navigation from hypothesis node instances.

Usage:

    with graph_context(graph):
        frame_node.roles        # FrameHypothesisNode.roles
        plan_node.phases        # MotionPlanHypothesisNode.phases

    # Or via the graph itself:
    with graph.query_context():
        frame_node.roles
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Generator, List, Optional

if TYPE_CHECKING:
    from llmr.hypotheses.graph import HypothesisGraph

_graph_context: ContextVar[Optional["HypothesisGraph"]] = ContextVar(
    "_graph_context", default=None
)


def _current_graph() -> "HypothesisGraph":
    graph = _graph_context.get()
    if graph is None:
        raise RuntimeError(
            "No HypothesisGraph in context. Wrap the query with "
            "`with graph_context(graph): ...` or `with graph.query_context(): ...` "
            "before accessing graph-linked node attributes."
        )
    return graph


@contextmanager
def graph_context(
    graph: "HypothesisGraph",
) -> Generator["HypothesisGraph", None, None]:
    """Context manager that makes *graph* available to GraphLinked node attributes.

    Supports nesting: the previous context is restored on exit.
    """
    token = _graph_context.set(graph)
    try:
        yield graph
    finally:
        _graph_context.reset(token)


class GraphLinked:
    """Mixin granting attribute-based graph navigation from node instances.

    Mix this into any HypothesisNode subclass to gain `linked()` and
    `linked_sources()`. Both require an active graph context (set via
    `graph_context(graph)` or `graph.query_context()`).
    """

    def linked(
        self,
        edge_type: type,
        node_type: Optional[type] = None,
    ) -> List:
        """Return nodes reachable from self via *edge_type* outgoing edges."""
        return _current_graph().get_targets(self.id, edge_type, node_type)  # type: ignore[attr-defined]

    def linked_sources(
        self,
        edge_type: type,
        node_type: Optional[type] = None,
    ) -> List:
        """Return nodes pointing to self via *edge_type* incoming edges."""
        return _current_graph().get_sources(self.id, edge_type, node_type)  # type: ignore[attr-defined]
