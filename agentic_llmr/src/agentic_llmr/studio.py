"""LangGraph Studio entry point — exposes the orchestrator graph for Studio visualization."""

import os
from langchain_openai import ChatOpenAI
from agentic_llmr.core.orchestrator import ReActAgent

def _build_graph():
    llm = ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
    )
    return ReActAgent(llm=llm).agent_executor

graph = _build_graph()
