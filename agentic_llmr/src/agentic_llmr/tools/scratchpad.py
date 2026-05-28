"""Scratchpad tools — file-based read/write scratchpad for agent reasoning notes."""

import logging
import os
from typing import Type
from pydantic import BaseModel, Field
from agentic_llmr.core.interfaces import AgenticTool

logger = logging.getLogger(__name__)


class WriteScratchpadInput(BaseModel):
    content: str = Field(description="The markdown or text content to write to the scratchpad.")
    mode: str = Field(
        description="Use 'overwrite' to clear and write, or 'append' to add to the bottom.",
        default="append",
    )


class WriteScratchpadTool(AgenticTool):
    name: str = "write_scratchpad"
    description: str = (
        "Write plans, reasoning, or notes to an explicit markdown scratchpad file. "
        "Useful for keeping track of complex, multi-step constraints."
    )
    args_schema: Type[BaseModel] = WriteScratchpadInput
    scratchpad_path: str = "agent_scratchpad.md"

    def _run(self, content: str, mode: str = "append") -> str:
        try:
            logger.debug(f"[Memory Tool] Writing to {self.scratchpad_path} (mode: {mode})...")
            file_mode = "a" if mode == "append" else "w"
            with open(self.scratchpad_path, file_mode, encoding="utf-8") as f:
                f.write(content + "\n")
            return f"Successfully wrote to {self.scratchpad_path}."
        except Exception as e:
            return self._handle_error(e)


class ReadScratchpadInput(BaseModel):
    pass


class ReadScratchpadTool(AgenticTool):
    name: str = "read_scratchpad"
    description: str = "Read the current contents of your explicit markdown scratchpad file."
    args_schema: Type[BaseModel] = ReadScratchpadInput
    scratchpad_path: str = "agent_scratchpad.md"

    def _run(self) -> str:
        try:
            logger.debug(f"[Memory Tool] Reading {self.scratchpad_path}...")
            if not os.path.exists(self.scratchpad_path):
                return "The scratchpad is currently empty (file does not exist)."
            with open(self.scratchpad_path, "r", encoding="utf-8") as f:
                content = f.read()
            return content if content else "The scratchpad is empty."
        except Exception as e:
            return self._handle_error(e)
