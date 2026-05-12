"""Action construction and strict-required validation."""

from __future__ import annotations

from dataclasses import dataclass

from llmr_updated_arch.core.errors import LLMUnresolvedRequiredFields
from llmr_updated_arch.integrations.krrood.introspect import ActionFieldIntrospector
from llmr_updated_arch.integrations.krrood.match_reader import (
    MatchSnapshot,
    construct_action,
    missing_required_fields,
)


@dataclass
class ActionMaterializer:
    """Construct the final action instance from a resolved match snapshot."""

    introspector: ActionFieldIntrospector

    def materialize(self, match_snapshot: MatchSnapshot, *, strict_required: bool) -> object:
        if strict_required:
            unresolved = missing_required_fields(match_snapshot, self.introspector)
            if unresolved:
                raise LLMUnresolvedRequiredFields(
                    action_name=match_snapshot.action_name,
                    unresolved_fields=unresolved,
                )
        return construct_action(match_snapshot)
