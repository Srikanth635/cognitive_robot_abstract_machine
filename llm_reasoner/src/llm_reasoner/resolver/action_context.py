"""
The action a resolution serves, stated in that resolution's prompt: a resolution
choosing what describes an action, such as the situation the action is performed in,
reads the action's own request, the slots its query leaves open with their candidates in
the world, and the values its query fixes.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.entity_query_language.query.match import Match

from llm_reasoner.resolver.context import PromptContextProvider
from llm_reasoner.resolver.prompt_format import (
    PromptBlock,
    PromptSection,
    render_blocks,
)
from llm_reasoner.resolver.prompts import PromptBuilder
from llm_reasoner.resolver.reflection import SlotKind, reflect_slots


@dataclass
class ActionContextProvider(PromptContextProvider):
    """
    States the action a resolution serves: its query in English, the slots the query
    leaves open and the values it fixes, written as the action's own resolution writes
    them, so the individuals the action involves can be read in the world.
    """

    action: Match
    """
    The query for the action.
    """

    builder: PromptBuilder
    """
    Writes the action's request, open slots and given values.
    """

    def context_for(self, expression: Match) -> str:
        slots = reflect_slots(self.action)
        open_slots = [
            self.builder.slot_entry(slot, self.action)
            for slot in slots
            if slot.kind is not SlotKind.BOUND
        ]
        given = [
            self.builder.given_entry(slot, self.action)
            for slot in slots
            if slot.kind is SlotKind.BOUND
        ]
        blocks = [
            PromptBlock(
                section=PromptSection.REQUEST,
                body=self.builder.verbalize_request(self.action),
            )
        ]
        if open_slots:
            blocks.append(
                PromptBlock(
                    section=PromptSection.SLOTS, body=self.builder.as_json(open_slots)
                )
            )
        if given:
            blocks.append(
                PromptBlock(
                    section=PromptSection.GIVEN, body=self.builder.as_json(given)
                )
            )
        return render_blocks(blocks)

    @property
    def section(self) -> PromptSection:
        return PromptSection.ACTION
