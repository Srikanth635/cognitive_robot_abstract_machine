"""GraphLinked mixin — context-free attribute-based graph navigation.

Nodes injected into a HypothesisGraph receive a direct weak-ref (_graph_ref)
at add_node time.  Navigation properties (linked, linked_sources) use that
reference directly — no context manager required.

    graph.add_node(frame)
    frame.roles        # works anywhere, no wrapper needed
    plan.phases        # same
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from llmr.hypotheses.graph import HypothesisGraph


class GraphLinked:
    """Mixin granting context-free attribute-based graph navigation.

    HypothesisGraph.add_node injects _graph_ref (a weakref) into every
    GraphLinked node it stores.  Call linked() / linked_sources() from any
    property to traverse typed edges without a context manager.
    """

    def _get_graph(self) -> "HypothesisGraph":
        ref = getattr(self, "_graph_ref", None)
        if ref is not None:
            graph = ref()
            if graph is not None:
                return graph
        raise RuntimeError(
            f"{type(self).__name__} has no graph reference. "
            "Add the node to a HypothesisGraph before navigating."
        )

    def linked(
        self,
        edge_type: type,
        node_type: Optional[type] = None,
    ) -> List:
        """Return nodes reachable from self via *edge_type* outgoing edges."""
        return self._get_graph().get_targets(self.id, edge_type, node_type)  # type: ignore[attr-defined]

    def linked_sources(
        self,
        edge_type: type,
        node_type: Optional[type] = None,
    ) -> List:
        """Return nodes pointing to self via *edge_type* incoming edges."""
        return self._get_graph().get_sources(self.id, edge_type, node_type)  # type: ignore[attr-defined]
