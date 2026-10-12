from __future__ import annotations

from dataclasses import asdict, dataclass, field

from typing_extensions import Any, Dict, List, Optional

RESPONSE_REASON = "reason"
"""
The key under which an unparseable response's reason is shown to the model.
"""

# %% base


class LanguageModelBackendError(Exception):
    """Base class for every failure raised while resolving a match with a model."""


# %% retryable failures


class RetryableModelFailure(LanguageModelBackendError):
    """A failure caused by the model's output, which asking again may fix."""

    def details(self) -> Dict[str, Any]:
        """
        :return: What the failure found, as data a repair shows the model.
        """
        return asdict(self)


@dataclass
class ModelResponseUnparseable(RetryableModelFailure):
    """Raised when a response is not the requested JSON array."""

    raw_response: str
    """The response text that could not be parsed."""

    reason: str
    """What made the response unusable."""

    def __str__(self) -> str:
        return f"{self.reason}\n--- raw response ---\n{self.raw_response[:2000]}"

    def details(self) -> Dict[str, Any]:
        return {RESPONSE_REASON: self.reason}


@dataclass
class UnknownSlotInResponse(RetryableModelFailure):
    """Raised when a response mentions a slot that is not part of the match."""

    slot_path: str
    """The unrecognised access path."""

    known_paths: List[str]
    """The access paths the match actually declares."""

    def __str__(self) -> str:
        return (
            f"Response contained unknown slot {self.slot_path!r}. "
            f"Known slots: {', '.join(self.known_paths)}"
        )


@dataclass
class MissingSlotInResponse(RetryableModelFailure):
    """Raised when a response leaves a slot unanswered."""

    slot_path: str
    """The access path that was not filled."""

    def __str__(self) -> str:
        return f"Response did not fill slot {self.slot_path!r}."


@dataclass
class UnknownGround(RetryableModelFailure):
    """Raised when an answer rests on something its source does not have."""

    source: str
    """The source the answer named it in."""

    name: str
    """The name the answer gave."""

    def __str__(self) -> str:
        return f"Response rests on {self.name!r}, which {self.source!r} does not have."


@dataclass
class SlotValueRejected(RetryableModelFailure):
    """Raised when a value cannot be decoded into its slot's declared type."""

    slot_path: str
    """The access path the value was offered for."""

    value: Any
    """The offered value."""

    reason: str
    """Why the value does not fit."""

    def __str__(self) -> str:
        return f"Value {self.value!r} rejected for {self.slot_path!r}: {self.reason}"


# %% terminal failures


@dataclass
class UnresolvableSlot(LanguageModelBackendError):
    """Raised when a slot asks for something a model cannot supply."""

    slot_path: str
    """The access path that cannot be filled."""

    declared_type: type
    """The annotated type of the attribute."""

    hint: str = ""
    """How the caller can reformulate the query."""

    def __str__(self) -> str:
        return (
            f"Cannot resolve {self.slot_path!r} of type "
            f"{self.declared_type.__name__}. {self.hint}"
        )


@dataclass
class ReferentSlotHasNoDomain(UnresolvableSlot):
    """
    Raised when an individual-typed slot is left unspecified with no extension.

    Values can be generated; individuals must be selected from an extension, so such a
    slot has to be written as ``variable(T)`` or ``variable(T, domain)``.
    """


@dataclass
class NoCandidateReferents(LanguageModelBackendError):
    """
    Raised when the extension of an individual-typed slot is empty.

    States that the world contains no such thing, which the caller should surface
    rather than work around.
    """

    slot_path: str
    """The access path whose extension is empty."""

    declared_type: type
    """The annotated type of the attribute."""

    def __str__(self) -> str:
        return (
            f"No candidates available for {self.slot_path!r} of type "
            f"{self.declared_type.__name__}."
        )


@dataclass
class ReferentNotModelled(LanguageModelBackendError):
    """
    Raised when a referent plausibly exists but has not been perceived or annotated.

    Distinct from :class:`NoCandidateReferents` because the appropriate response is to
    perceive rather than to replan.
    """

    slot_path: str
    """The access path that could not be grounded."""

    description: str = ""
    """The description that failed to ground."""


@dataclass
class RepairBudgetExhausted(LanguageModelBackendError):
    """Raised when every attempt, including repairs, produced an unusable response."""

    attempts: int
    """How many completions were requested in total."""

    last_error: Optional[Exception] = None
    """The failure of the final attempt."""

    def __str__(self) -> str:
        return (
            f"Gave up after {self.attempts} attempt(s). Last error: {self.last_error}"
        )


@dataclass
class NoValidCandidates(RetryableModelFailure):
    """Raised when every constructed candidate fails the match's where-conditions."""

    constructed: int
    """How many candidates were built before filtering."""

    broken_conditions: List[str] = field(default_factory=list)
    """The conditions, in English, that none of the candidates meets."""

    def __str__(self) -> str:
        broken = "".join(f"\n- {condition}" for condition in self.broken_conditions)
        return (
            f"All {self.constructed} constructed candidate(s) failed the query's "
            f"where-conditions. None of them meets:{broken}"
        )


# %% verdicts


@dataclass
class NoFittingReferent(LanguageModelBackendError):
    """
    Raised when the model answers that none of a referent slot's candidates is what the
    request asks for, rather than substituting a near miss.
    """

    slot_path: str
    """The access path of the slot."""

    reason: str
    """Why the model found no candidate fitting."""

    def __str__(self) -> str:
        return f"No candidate fits {self.slot_path!r}: {self.reason}"


@dataclass
class AmbiguousReferent(LanguageModelBackendError):
    """
    Raised when the model answers that several candidates of a referent slot fit equally
    and nothing in the request prefers one, so the caller can narrow the request.
    """

    slot_path: str
    """The access path of the slot."""

    candidates: List[Any]
    """The candidates fitting equally, as the individuals themselves."""

    reason: str
    """Why the model found none of them preferred."""

    def __str__(self) -> str:
        return (
            f"{len(self.candidates)} candidates fit {self.slot_path!r} equally: "
            f"{self.reason}"
        )


@dataclass
class UnsatisfiableRequest(LanguageModelBackendError):
    """
    Raised when the model answers that the query's conditions cannot all hold together,
    rather than answering with something that breaks one of them.
    """

    reason: str
    """Why the model found the conditions unable to hold together."""

    def __str__(self) -> str:
        return f"The query's conditions cannot all hold: {self.reason}"


# %% configuration


@dataclass
class ClientCannotUseTools(LanguageModelBackendError):
    """Raised when tools are offered through a client that cannot let a model call them."""

    client_identity: str
    """The client given."""

    def __str__(self) -> str:
        return f"The client {self.client_identity} cannot let a model call tools."
