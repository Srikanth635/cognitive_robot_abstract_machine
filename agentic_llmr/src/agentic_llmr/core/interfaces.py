"""Core abstract interfaces for the Agentic LLM package."""

from abc import ABC, abstractmethod
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
