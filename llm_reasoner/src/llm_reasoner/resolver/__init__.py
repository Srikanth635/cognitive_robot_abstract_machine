"""
Language-model resolution of underspecified actions against world state.
"""

from llm_reasoner.resolver.backend import Exchange, LanguageModelBackend
from llm_reasoner.resolver.context import (
    CandidateRenderer,
    CompositeContextProvider,
    PromptContextProvider,
    ReprCandidateRenderer,
    SchemaOnlyContextProvider,
    StructuralCandidateRenderer,
    TaskContextProvider,
    context_sections,
)
from llm_reasoner.resolver.exceptions import (
    LanguageModelBackendError,
    MissingSlotInResponse,
    ModelResponseUnparseable,
    NoCandidateReferents,
    NoValidCandidates,
    ReferentNotModelled,
    ReferentSlotHasNoDomain,
    RepairBudgetExhausted,
    RetryableModelFailure,
    SlotValueRejected,
    UnknownSlotInResponse,
    UnresolvableSlot,
)

from llm_reasoner.resolver.prompts import (
    SYSTEM_PROMPT_TEMPLATE,
    PromptBuilder,
    SourceDetail,
)
from llm_reasoner.resolver.reflection import (
    GENERATABLE_TYPES,
    Slot,
    SlotKind,
    SlotReflector,
    is_generatable,
    open_slots,
    reflect_slots,
    sortal_base,
    sortal_domain,
    union_alternatives,
)

# %% public API

__all__ = [
    "CandidateRenderer",
    "CompositeContextProvider",
    "Exchange",
    "GENERATABLE_TYPES",
    "LanguageModelBackend",
    "LanguageModelBackendError",
    "MissingSlotInResponse",
    "ModelResponseUnparseable",
    "NoCandidateReferents",
    "NoValidCandidates",
    "PromptBuilder",
    "PromptContextProvider",
    "ReferentNotModelled",
    "ReferentSlotHasNoDomain",
    "RepairBudgetExhausted",
    "ReprCandidateRenderer",
    "RetryableModelFailure",
    "SYSTEM_PROMPT_TEMPLATE",
    "SchemaOnlyContextProvider",
    "Slot",
    "SlotKind",
    "SlotReflector",
    "SlotValueRejected",
    "SourceDetail",
    "StructuralCandidateRenderer",
    "TaskContextProvider",
    "UnknownSlotInResponse",
    "UnresolvableSlot",
    "context_sections",
    "is_generatable",
    "open_slots",
    "reflect_slots",
    "sortal_base",
    "sortal_domain",
    "union_alternatives",
]
