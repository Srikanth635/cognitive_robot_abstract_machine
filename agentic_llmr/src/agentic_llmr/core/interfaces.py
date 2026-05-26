"""Core abstract interfaces for the Agentic LLM package."""

from abc import ABC, abstractmethod
from typing import Any, Type
from pydantic import BaseModel, Field, PrivateAttr
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool


class BaseCognitiveAgent(ABC):
    """Abstract base class for all cognitive agents in the system."""

    @abstractmethod
    def resolve_action(self, instruction: str, template_context: str = "") -> str:
        """Run the agent loop to resolve an instruction into executable parameters."""
        pass


class AgenticTool(BaseTool, ABC):
    """Base class for all tools in the agentic ecosystem."""

    def _handle_error(self, error: Exception) -> str:
        """Standardized error handling for all agentic tools."""
        return f"Tool execution failed: {str(error)}. Please review your parameters."


class _SubAgentInput(BaseModel):
    query: str = Field(description="Natural language query to delegate to the specialist sub-agent.")


class SubAgentTool(AgenticTool):
    """Generic wrapper that forwards a query string to a LangGraph sub-agent.

    Subclasses only need to declare ``name``, ``description``, and optionally
    ``args_schema`` if the input model differs from the default single-query field.
    The ``_agent`` and all invocation logic live here once.
    """

    args_schema: Type[BaseModel] = _SubAgentInput
    _agent: Any = PrivateAttr()

    def __init__(self, agent: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._agent = agent

    def _run(self, query: str, run_manager=None) -> str:
        try:
            config = {"callbacks": run_manager.get_child()} if run_manager else {}
            result = self._agent.invoke(
                {"messages": [HumanMessage(content=query)]}, config=config
            )
            return result["messages"][-1].content
        except Exception as e:
            return self._handle_error(e)
