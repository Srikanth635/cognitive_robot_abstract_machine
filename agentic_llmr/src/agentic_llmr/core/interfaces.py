"""Core abstract interfaces and shared subgraph utilities for the agentic LLM package."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END
from pydantic import BaseModel

from agentic_llmr.core.state import _merge_dicts


# ---------------------------------------------------------------------------
# Artifact base class
# ---------------------------------------------------------------------------

class Artifact(BaseModel):
    """Base class for all tool artifacts returned alongside LLM-visible content.

    Every concrete artifact must implement to_fact_entry() → (key, value) so the
    ExecuteToolsNode can route structured data directly into state channels without
    any name-based dispatch table.
    """

    def to_fact_entry(self) -> Tuple[str, Any]:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement to_fact_entry()."
        )


# ---------------------------------------------------------------------------
# Abstract agent base
# ---------------------------------------------------------------------------

class BaseCognitiveAgent(ABC):
    """Abstract base class for all cognitive agents in the system."""

    @abstractmethod
    def resolve_action(self, instruction: str, template_context: str = "") -> str:
        """Run the agent loop to resolve an instruction into executable parameters."""
        ...


# ---------------------------------------------------------------------------
# Agentic tool base
# ---------------------------------------------------------------------------

class AgenticTool(BaseTool, ABC):
    """Base class for all agentic tools.

    Two implementation patterns are supported:

    Pattern A — structured artifact (most tools):
        Override `_query(**kwargs) -> Artifact` to query the world and return a typed
        artifact.  Optionally override `_format(artifact) -> str` to customise the
        LLM-visible text.  The base `_run` wires these together and returns
        `(str, artifact)` so ExecuteToolsNode can route the artifact into state.

    Pattern B — no structured artifact (scratchpad tools, stubs):
        Override `_run(**kwargs)` directly and return `(str, None)`.
    """

    response_format: str = "content_and_artifact"

    def _run(self, **kwargs: Any) -> Tuple[str, Optional[BaseModel]]:
        try:
            artifact = self._query(**kwargs)
            return self._format(artifact), artifact
        except Exception as e:
            return self._handle_error(e), None

    def _query(self, **kwargs: Any) -> BaseModel:
        raise NotImplementedError(
            f"{type(self).__name__} must implement _query() or override _run() directly."
        )

    def _format(self, artifact: BaseModel) -> str:
        return artifact.model_dump_json(indent=2)

    def _handle_error(self, error: Exception) -> str:
        return f"Tool execution failed: {str(error)}. Please review your parameters."


# ---------------------------------------------------------------------------
# Shared subgraph utilities
# ---------------------------------------------------------------------------

class ExecuteToolsNode:
    """Generic tool-execution node for LangGraph subgraphs.

    Runs every tool call in the last AIMessage, extracts the (content, artifact) pair,
    and routes artifact facts into a named state channel via artifact.to_fact_entry().

    Parameters
    ----------
    tools :
        AgenticTool instances available to this node.
    state_key :
        The state channel key to write artifact facts into
        (e.g. "scene_facts", "kinematic_facts", "action_schema").
    merge :
        If True (default), nested dict values are deep-merged via _merge_dicts so
        successive tool calls accumulate facts rather than overwrite them.
        Set False for flat fact channels like "action_schema" where last-write wins.
    """

    def __init__(self, tools: list, state_key: str, merge: bool = True) -> None:
        self._tools: Dict[str, AgenticTool] = {t.name: t for t in tools}
        self._state_key = state_key
        self._merge = merge

    def __call__(self, state: dict) -> dict:
        last_msg = state["messages"][-1]
        tool_messages = []
        facts_update: Dict[str, Any] = {}

        for tc in getattr(last_msg, "tool_calls", []):
            tool = self._tools.get(tc["name"])
            if tool is None:
                tool_messages.append(
                    ToolMessage(
                        content=f"Tool '{tc['name']}' not found.",
                        tool_call_id=tc["id"],
                        name=tc["name"],
                    )
                )
                continue

            try:
                result = tool._run(**tc["args"])
                if isinstance(result, tuple) and len(result) == 2:
                    content, artifact = result
                else:
                    content, artifact = str(result), None
            except Exception as e:
                content, artifact = f"Tool error: {e}", None

            if artifact is not None:
                key, value = artifact.to_fact_entry()
                if self._merge:
                    existing = facts_update.get(key, {})
                    facts_update[key] = (
                        _merge_dicts(existing, value)
                        if isinstance(existing, dict) and isinstance(value, dict)
                        else value
                    )
                else:
                    facts_update[key] = value

            tool_messages.append(
                ToolMessage(content=str(content), tool_call_id=tc["id"], name=tc["name"])
            )

        return {"messages": tool_messages, self._state_key: facts_update}


def tools_condition(state: dict) -> str:
    """Conditional edge: route to execute_tools if the last message has tool calls."""
    if getattr(state["messages"][-1], "tool_calls", None):
        return "execute_tools"
    return END


def make_prepare_query(facts_spec: List[Tuple[str, str]]) -> Callable:
    """Return a prepare_query node function for a subgraph.

    Parameters
    ----------
    facts_spec :
        List of (state_key, header_label) pairs.  Each entry injects the named
        state channel into the HumanMessage context under the given label so the
        subagent LLM can avoid redundant queries.

        Example::

            make_prepare_query([
                ("scene_facts",     "Already known (do NOT re-query)"),
                ("kinematic_facts", "Known kinematic facts (do NOT re-query these)"),
            ])
    """
    def prepare_query(state: dict) -> dict:
        # Prefer the scoped sub-task the supervisor handed this specialist; fall
        # back to the full instruction when routed without one.
        task = state.get("current_task") or state.get("instruction", "")
        query = f"Instruction: {task}"
        if ctx := state.get("template_context", ""):
            query += f"\nContext: {ctx}"
        for state_key, label in facts_spec:
            facts = state.get(state_key, {})
            if facts:
                facts_str = "\n".join(f"  {k}: {v}" for k, v in facts.items())
                query += f"\n\n{label}:\n{facts_str}"
        return {"messages": [HumanMessage(content=query)]}
    return prepare_query


def make_call_model(llm_with_tools: Any, system_prompt: str) -> Callable:
    """Return a call_model node function bound to a tools-enabled LLM and system prompt."""
    def call_model(state: dict) -> dict:
        msgs = [SystemMessage(content=system_prompt)] + list(state["messages"])
        return {"messages": [llm_with_tools.invoke(msgs)]}
    return call_model
