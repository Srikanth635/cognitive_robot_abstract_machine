"""LangGraph multi-agent model reasoning graph (FrameNet + Flanagan)."""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState

from ..agents.flanagan_agent import flanagan_agent_node
from ..agents.framenet_agent import framenet_node
from ..states.all_states import ModelReasoningState

_graph_memory = MemorySaver()


def _results_aggregator_node(state: MessagesState) -> dict:
    """Collect outputs from parallel reasoning agents."""
    return {"messages": "Aggregated model outputs"}


model_reasoning_graph = (
    StateGraph(ModelReasoningState)
    .add_node("framenet_reasoner", framenet_node)
    .add_node("flanagan_reasoner", flanagan_agent_node)
    .add_node("aggregator", _results_aggregator_node)
    .add_edge(START, "framenet_reasoner")
    .add_edge(START, "flanagan_reasoner")
    .add_edge("flanagan_reasoner", "aggregator")
    .add_edge("framenet_reasoner", "aggregator")
    .add_edge("aggregator", END)
    .compile(checkpointer=_graph_memory)
)
