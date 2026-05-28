"""Reasoning trace collector and pyvis renderer for the agentic pipeline."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Node model
# ---------------------------------------------------------------------------

_SUB_AGENT_TOOLS = {"query_scene_perception", "query_kinematics", "query_action_schema"}

_NODE_STYLES = {
    "orchestrator": {"color": "#4472C4", "shape": "box",     "font_color": "#ffffff", "size": 28},
    "sub_agent":    {"color": "#70AD47", "shape": "ellipse",  "font_color": "#ffffff", "size": 22},
    "tool":         {"color": "#ED7D31", "shape": "ellipse",  "font_color": "#ffffff", "size": 18},
}


@dataclass
class TraceNode:
    """One node in the reasoning trace tree.

    The root is always the Orchestrator; children are sub-agents or primitive tools.
    Input/output summaries are truncated so the tree stays displayable.
    """

    run_id: str
    name: str
    node_type: str          # "orchestrator" | "sub_agent" | "tool"
    parent_run_id: Optional[str] = None
    input_summary: str = ""
    output_summary: str = ""
    children: List["TraceNode"] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Callback handler
# ---------------------------------------------------------------------------

class TraceCollector(BaseCallbackHandler):
    """Collects LangChain run events and builds a two-level trace tree."""

    raise_error: bool = False   # suppress internal LangChain assertion

    def __init__(self) -> None:
        super().__init__()
        root_id = str(uuid.uuid4())
        self.root = TraceNode(
            run_id=root_id,
            name="Orchestrator",
            node_type="orchestrator",
        )
        self._nodes: Dict[str, TraceNode] = {root_id: self.root}
        self._root_id = root_id
        # Maps every run_id → its parent_run_id, for ALL event types.
        # Primitive tool calls have parent_run_id pointing to intermediate
        # LangGraph chain/agent runs (not tool nodes), so we walk up this
        # map until we find a run_id that is a known tool node.
        self._run_parent: Dict[str, str] = {}

    # -- helpers --

    def _node_type(self, name: str) -> str:
        return "sub_agent" if name in _SUB_AGENT_TOOLS else "tool"

    def _track(self, run_id: uuid.UUID, parent_run_id: Optional[uuid.UUID]) -> None:
        if parent_run_id is not None:
            self._run_parent[str(run_id)] = str(parent_run_id)

    def _parent_node(self, parent_run_id: Optional[uuid.UUID]) -> TraceNode:
        """Walk up the run-parent chain until we reach a known tool/agent node."""
        if parent_run_id is None:
            return self.root
        current = str(parent_run_id)
        seen: set = set()
        while current and current not in seen:
            if current in self._nodes:
                return self._nodes[current]
            seen.add(current)
            current = self._run_parent.get(current, "")
        return self.root

    @staticmethod
    def _truncate(text: str, n: int = 120) -> str:
        s = str(text).replace("\n", " ").strip()
        return s[:n] + "…" if len(s) > n else s

    # -- LangChain events: track intermediate runs so parent-walk works --

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Any,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Track this run's parent so tool events can walk up to a known node."""
        self._track(run_id, parent_run_id)

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: Any,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Track LLM runs in the parent map for tool-parent resolution."""
        self._track(run_id, parent_run_id)

    # -- LangChain events: tool calls --

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Create a TraceNode for this tool call and attach it to its parent."""
        self._track(run_id, parent_run_id)
        name = serialized.get("name") or kwargs.get("name", "unknown_tool")
        parent = self._parent_node(parent_run_id)
        node = TraceNode(
            run_id=str(run_id),
            name=name,
            node_type=self._node_type(name),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            input_summary=self._truncate(input_str),
        )
        parent.children.append(node)
        self._nodes[str(run_id)] = node

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        """Record the tool's output summary on its existing TraceNode."""
        node = self._nodes.get(str(run_id))
        if node:
            node.output_summary = self._truncate(str(output), 200)


# ---------------------------------------------------------------------------
# Pyvis renderer helpers
# ---------------------------------------------------------------------------


def _build_pyvis_network() -> Any:
    """Construct and configure the pyvis Network with fixed layout and style settings."""
    from pyvis.network import Network
    net = Network(
        height="520px",
        width="100%",
        directed=True,
        bgcolor="#1e1e2e",
        font_color="#cdd6f4",
        notebook=True,
        cdn_resources="in_line",
    )
    net.set_options("""
    {
      "edges": {
        "arrows": { "to": { "enabled": true, "scaleFactor": 0.6 } },
        "color": { "color": "#585b70", "highlight": "#cba6f7" },
        "smooth": { "type": "cubicBezier", "forceDirection": "vertical", "roundness": 0.4 }
      },
      "layout": {
        "hierarchical": {
          "enabled": true,
          "direction": "UD",
          "sortMethod": "directed",
          "nodeSpacing": 160,
          "levelSeparation": 110
        }
      },
      "physics": { "enabled": false },
      "interaction": { "hover": true, "tooltipDelay": 100 }
    }
    """)
    return net


def _add_trace_node(net: Any, node: TraceNode) -> None:
    """Recursively add a TraceNode and its children to the pyvis Network."""
    style = _NODE_STYLES[node.node_type]
    tooltip = (
        f"<b>{node.name}</b><br>"
        f"<b>In:</b> {node.input_summary or '—'}<br>"
        f"<b>Out:</b> {node.output_summary or '—'}"
    )
    net.add_node(
        node.run_id,
        label=node.name,
        title=tooltip,
        color=style["color"],
        shape=style["shape"],
        font={"color": style["font_color"], "size": 13},
        size=style["size"],
    )
    for child in node.children:
        _add_trace_node(net, child)
        net.add_edge(node.run_id, child.run_id)


def _generate_trace_html(net: Any, title: str) -> str:
    """Generate the pyvis HTML string and inject a title banner into the body."""
    html_str = net.generate_html(name="trace.html")
    banner = (
        f'<div style="font-family:monospace;color:#cba6f7;font-size:13px;'
        f'padding:6px 12px;background:#181825;">{title}</div>'
    )
    return html_str.replace("<body>", f"<body>{banner}", 1)


def _write_html_to_file(html_str: str, output_path: str, open_browser: bool) -> None:
    """Write rendered HTML to a file and optionally open it in the default browser."""
    import os
    import webbrowser
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_str)
    logger.info("Trace saved → %s", os.path.abspath(output_path))
    if open_browser:
        webbrowser.open(f"file://{os.path.abspath(output_path)}")


def _display_html_inline(html_str: str) -> None:
    """Display the HTML string inline in a Jupyter notebook cell."""
    try:
        from IPython.display import display, HTML
        display(HTML(html_str))
    except ImportError:
        logger.warning("IPython not available — pass output_path to save as HTML file.")


# ---------------------------------------------------------------------------
# Pyvis renderer
# ---------------------------------------------------------------------------


def render_trace(
    collector: TraceCollector,
    title: str = "Agent Reasoning Trace",
    output_path: Optional[str] = None,
    open_browser: bool = True,
) -> None:
    """Render the trace tree as an interactive pyvis graph.

    If output_path is given, the HTML is saved to that file and opened in the
    default browser (set open_browser=False to skip). Otherwise the graph is
    displayed inline in a Jupyter notebook cell.
    """
    try:
        from pyvis.network import Network  # noqa: F401 — import guard only
    except ImportError as e:
        logger.warning("render_trace requires pyvis: %s", e)
        return

    net = _build_pyvis_network()
    _add_trace_node(net, collector.root)
    html_str = _generate_trace_html(net, title)

    if output_path is not None:
        _write_html_to_file(html_str, output_path, open_browser)
    else:
        _display_html_inline(html_str)
