"""The generation loop.

    underspecified action
      -> module selection        which microworlds does this action need
      -> schema derivation       the output shape, from the compiled classes
      -> ask                     one completion
      -> decode                  a fragment graph, checked against the schema
      -> construct               instances, wired by id in a second pass
      -> validate                the microworld's own acceptance constraints
      -> repair, or admit with provenance

SHAPED AFTER THE RESOLVER, NOT SUBCLASSED FROM IT
--------------------------------------------------
The pipeline is deliberately the same shape as ``resolver.LanguageModelBackend``:
ask, decode, construct, validate, repair on rejection. What differs is every
stage's content. There is no query to reflect over, no world to select
individuals from, and the result is a graph rather than one object, so the
resolver's ``_evaluate`` could not be reused without carrying an EQL ``Match``
through code that has none.

What IS reused is genuine: ``Exchange`` for provenance, and the stateless repair
pattern -- fold the failure into the next prompt and ask again, so history lives
in the caller and every attempt is reproducible from its two prompts.

WHY VALIDATION FAILURE IS NOT AN ERROR
--------------------------------------
A hypothesis that fails its constraints repeatedly is retained as refuted rather
than discarded. It is evidence about the ontology as much as about the model: a
constraint that a competent model keeps violating is often a constraint that
says something the vocabulary does not let it say.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

from llm_reasoner.clients.base import Exchange, LanguageModelClient
from llm_reasoner.exceptions import LLMReasonerError
from llm_reasoner.knowledge.generation.assumptions import extract_assumptions
from llm_reasoner.knowledge.generation.prompts import (
    SYSTEM_PROMPT,
    GenerationPromptBuilder,
)
from llm_reasoner.knowledge.generation.schema import (
    FieldKind,
    FieldSchema,
    GenerationSchema,
    derive_schema,
    to_json_schema,
)
from llm_reasoner.knowledge.generation.wire import (
    CanonicalFragmentIdentityAllocator,
    GenerationWireDecoder,
)
from llm_reasoner.knowledge.hypothesis import (
    HypothesisRecord,
    Provenance,
    ValidationReport,
)
from llm_reasoner.knowledge.microworlds.registry import (
    ModuleSelection,
    select_modules,
)
from llm_reasoner.knowledge.ontological.constraints import ConstraintSuite
from llm_reasoner.knowledge.ontological.decorators import (
    REGISTRY,
    ConceptReference,
    OntologyRegistry,
)

JSON_OBJECT = re.compile(r"\{.*}", re.DOTALL)
"""Locates the JSON object in a response that may carry surrounding text.

A fallback for clients without constrained decoding. Greedy across newlines, so
it is only sound when the response is a single object -- which the system prompt
demands, and which a client supporting a typed output schema guarantees outright.
"""


class GenerationFailure(LLMReasonerError):
    """Base for failures of the generation loop."""


class ResponseUnusable(GenerationFailure):
    """The response could not be turned into a fragment graph.

    Retryable: the repair prompt states the reason, and a model that emitted
    prose around its JSON or misnamed a type can usually correct it.
    """


class RepairBudgetExhausted(GenerationFailure):
    """Repair ran out of attempts without producing a usable graph.

    Carries every exchange that was tried, not just a message. Without them the
    raw responses that actually failed are gone at the exact moment someone
    needs to look at one -- the caller is left staring at a JSON decode error
    with no way to see the JSON.
    """

    def __init__(self, message: str, exchanges: Sequence[Exchange] = ()) -> None:
        super().__init__(message)
        self.exchanges: tuple[Exchange, ...] = tuple(exchanges)


@dataclass
class GenerationResult:
    """What one generation attempt produced, admissible or not."""

    record: HypothesisRecord
    selection: ModuleSelection
    exchanges: tuple[Exchange, ...]

    @property
    def admitted(self) -> bool:
        return self.record.servable


@dataclass
class HypothesisGenerator:
    """Produces stereotypical hypotheses for a kind of action."""

    client: LanguageModelClient
    suite: ConstraintSuite
    """The acceptance constraints of the microworld being generated for. Owned
    by the microworld, not by this loop, which is what lets a second microworld
    plug in without touching anything here."""

    root_class: str
    """Class of the top-level fragment, e.g. ``PouringPhaseStructure``."""

    prompt_version: str = "generation/v2"
    max_repairs: int = 2
    registry: OntologyRegistry = REGISTRY
    prompt_builder: GenerationPromptBuilder = field(
        default_factory=GenerationPromptBuilder
    )
    notes: tuple[str, ...] = ()

    log: list[Exchange] = field(default_factory=list, init=False)
    """Every exchange from the most recent call to :meth:`generate`, in order.

    Mirrors ``resolver.LanguageModelBackend.log``: kept on the object rather
    than only inside the call, so a raw response can be inspected after a
    failure without having to catch the exception first. Reset at the start of
    each ``generate`` call, so it always reflects the most recent attempt."""

    # -- entry point -------------------------------------------------------

    def generate(
        self,
        task: ConceptReference,
        action_label: Optional[str] = None,
        selection: Optional[ModuleSelection] = None,
    ) -> GenerationResult:
        """Generate one hypothesis for a kind of action.

        :param task: The action, as a reference with its ancestors materialised
            so the capability registry can place it.
        :param action_label: Plain-language name used in the prompt; the task's
            own label when omitted.
        :param selection: Pre-resolved modules, when the caller has already
            selected them.
        """
        selection = selection or select_modules(task)
        schema = derive_schema(
            [m.name for m in selection.modules],
            root=self.root_class,
            registry=self.registry,
        )
        prompt = self.prompt_builder.build(
            schema=schema,
            action_label=action_label or str(task),
            degraded=selection.degraded,
            notes=self.notes,
        )

        json_schema = to_json_schema(schema)
        self.log = []
        exchanges: list[Exchange] = self.log
        last_decode_error: Optional[str] = None
        last_success: Optional[tuple[Any, Sequence[Any]]] = None
        """The most recent (root, violations) pair from an attempt that decoded,
        whether or not it went on to pass validation.

        Bundled together deliberately, not tracked as two separate variables.
        A later attempt can fail to DECODE after an earlier one already failed
        VALIDATION, and if the two kinds of failure are tracked independently
        the loop can end holding one attempt's root next to a different
        attempt's violations -- or worse, next to an empty violations list from
        the decode failure, which used to make a perfectly good earlier result
        look like it had nothing to report. Keeping them as one pair makes that
        mismatch structurally impossible."""

        for attempt in range(self.max_repairs + 1):
            # constrained where the client supports it, free text where not --
            # the schema removes invented type names and misspelled fields
            # outright, which is the class of failure repair handles worst
            # because a model rarely learns a name it never saw
            raw = self.client.complete_structured(SYSTEM_PROMPT, prompt, json_schema)
            exchanges.append(
                Exchange(
                    system=SYSTEM_PROMPT,
                    user=prompt,
                    raw_response=raw,
                    client_identity=self.client.identity,
                    attempt=attempt,
                )
            )
            try:
                root, fragments = self._build(raw, schema)
            except ResponseUnusable as error:
                last_decode_error = str(error)
                prompt = self.prompt_builder.build_repair(
                    prompt, raw, (), reason=last_decode_error
                )
                continue

            violations = self.suite.evaluate(fragments)
            errors = [v for v in violations if getattr(v, "severity", "") == "error"]
            if not errors:
                return self._admit(root, task, selection, exchanges, violations)

            last_success = (root, violations)
            prompt = self.prompt_builder.build_repair(prompt, raw, errors)

        # every attempt used, and none produced an admissible hypothesis
        return self._refuse(task, selection, exchanges, last_success, last_decode_error)

    # -- decoding ----------------------------------------------------------

    def _build(self, raw: str, schema: GenerationSchema) -> tuple[Any, list[Any]]:
        """Turn a response into a wired graph of instances.

        Two passes, the same shape the Turtle codec uses: everything is
        instantiated by type first, then references are resolved. One pass cannot
        work, because fragments legitimately refer to each other in both
        directions and neither can be built first.
        """
        payload = self._parse(raw)
        declaration = GenerationWireDecoder(
            schema=schema,
            error_type=ResponseUnusable,
        ).decode(payload)

        # pass 1: instantiate
        instances: dict[str, Any] = {}
        identities = CanonicalFragmentIdentityAllocator()
        for fragment in declaration.fragments:
            class_metadata = self.registry.by_name(fragment.type_name)
            if class_metadata is None:
                raise ResponseUnusable(f"Type {fragment.type_name!r} is not declared.")
            instance = class_metadata.cls()
            instance.local_id = identities.allocate(fragment.type_name)
            if fragment.description:
                instance.label = fragment.description
            instances[fragment.identifier] = instance

        # pass 2: wire
        for fragment in declaration.fragments:
            instance = instances[fragment.identifier]
            class_schema = schema.get(fragment.type_name)
            if class_schema is None:
                raise ResponseUnusable(
                    f"Type {fragment.type_name!r} disappeared from the schema."
                )
            fields_by_name = {
                field_schema.name: field_schema for field_schema in class_schema.fields
            }
            for fragment_field in fragment.fields:
                field_schema = fields_by_name[fragment_field.name]
                decoded = self._decode(
                    field_schema,
                    fragment_field.value,
                    instances,
                    fragment.identifier,
                )
                if decoded is not None:
                    self._assign(
                        instance,
                        fragment_field.name,
                        decoded,
                        field_schema.collection,
                    )

        return instances[declaration.root.identifier], list(instances.values())

    @staticmethod
    def _id_hint(item: str, instances: Mapping[str, Any]) -> str:
        """
        Name what the model could have meant, from ids it actually declared.

        Pass 1 instantiates every fragment before pass 2 wires any reference, so
        the full set of ids the model used is already known by the time a
        reference fails to resolve -- there is no reason to leave the model
        guessing what it should have written when the answer is sitting in
        ``instances``. A close match is usually the real cause: the model
        assigned one id to a fragment and referenced a slightly different
        spelling of it elsewhere, which is a mistake naming the actual ids
        fixes directly rather than one a generic "not found" ever could.
        """
        import difflib

        known = sorted(instances)
        close = difflib.get_close_matches(item, known, n=3, cutoff=0.5)
        if close:
            return f"Did you mean one of: {close}? Ids you declared: {known}"
        return f"Ids you declared: {known}"

    @staticmethod
    def _parse(raw: str) -> Mapping[str, Any]:
        located = JSON_OBJECT.search(raw)
        if located is None:
            raise ResponseUnusable("No JSON object found in the response.")
        try:
            payload = json.loads(located.group(0))
        except json.JSONDecodeError as error:
            raise ResponseUnusable(f"Invalid JSON: {error}") from error
        if not isinstance(payload, dict):
            raise ResponseUnusable("Expected a JSON object at the top level.")
        return payload

    def _decode(
        self,
        field_schema: FieldSchema,
        value: Any,
        instances: Mapping[str, Any],
        owner_id: str,
    ) -> Any:
        values = value if isinstance(value, list) else [value]
        out = []
        for item in values:
            if item is None:
                continue
            if field_schema.kind is FieldKind.FRAGMENT:
                if not isinstance(item, str):
                    raise ResponseUnusable(
                        f"{owner_id}.{field_schema.name} must reference a "
                        "fragment by its id string."
                    )
                target = instances.get(item)
                if target is None:
                    raise ResponseUnusable(
                        f"{owner_id}.{field_schema.name} references unknown "
                        f"fragment id {item!r}. " + self._id_hint(item, instances)
                    )
                expected_meta = self.registry.by_name(field_schema.type_name)
                if expected_meta is not None and not isinstance(
                    target, expected_meta.cls
                ):
                    raise ResponseUnusable(
                        f"{owner_id}.{field_schema.name} references "
                        f"{type(target).__name__}, expected "
                        f"{field_schema.type_name} or one of its subtypes."
                    )
                out.append(target)
            elif field_schema.kind is FieldKind.CONCEPT:
                if not isinstance(item, str):
                    raise ResponseUnusable(
                        f"{owner_id}.{field_schema.name} must be a term name " "or IRI."
                    )
                out.append(self._concept_reference(field_schema, item, owner_id))
            elif field_schema.options:
                member = self._enum_member(field_schema, item, owner_id)
                out.append(member)
            else:
                out.append(item)
        if not out:
            return None
        return out if field_schema.collection else out[0]

    def _concept_reference(
        self, field_schema: FieldSchema, item: str, owner_id: str
    ) -> ConceptReference:
        """Normalize a selected ontology concept to its registered IRI."""
        if not field_schema.options:
            return ConceptReference(iri=item, label=item.rsplit("#", 1)[-1])
        concept_metadata = self.registry.by_name(item)
        if concept_metadata is None:
            concept_metadata = self.registry.meta_for_iri(item)
        if (
            concept_metadata is None
            or concept_metadata.local_name not in field_schema.options
        ):
            raise ResponseUnusable(
                f"{owner_id}.{field_schema.name}: {item!r} is not one of "
                + " | ".join(field_schema.options)
            )
        return ConceptReference(
            iri=concept_metadata.iri,
            label=concept_metadata.local_name,
        )

    def _enum_member(self, field_schema: FieldSchema, item: Any, owner_id: str) -> Any:
        meta = self.registry.by_name(field_schema.type_name)
        if meta is None or not issubclass(meta.cls, Enum):
            raise ResponseUnusable(
                f"{owner_id}.{field_schema.name}: {field_schema.type_name} is "
                "not an enumeration."
            )
        for member in meta.cls:
            if item in (member.name, member.value):
                return member
        raise ResponseUnusable(
            f"{owner_id}.{field_schema.name}: {item!r} is not one of "
            + " | ".join(field_schema.options)
        )

    @staticmethod
    def _assign(instance: Any, name: str, value: Any, collection: bool) -> None:
        if collection:
            current = getattr(instance, name, None)
            if isinstance(current, list):
                current.extend(v for v in value if v not in current)
                return
        setattr(instance, name, value)

    # -- outcomes ----------------------------------------------------------

    def _provenance(
        self, selection: ModuleSelection, exchanges: Sequence[Exchange]
    ) -> Provenance:
        return Provenance(
            generator=self.client.identity,
            prompt_version=self.prompt_version,
            module_versions=dict(selection.module_versions),
            degraded=selection.degraded,
            exchanges=tuple(exchanges),
        )

    def _admit(
        self,
        root: Any,
        task: ConceptReference,
        selection: ModuleSelection,
        exchanges: Sequence[Exchange],
        violations: Sequence[Any],
    ) -> GenerationResult:
        provenance = self._provenance(selection, exchanges)
        record = HypothesisRecord.from_root(root, task, provenance)
        record.assumptions = extract_assumptions(
            record.scope.fragments, registry=self.registry
        )
        record.validated(
            ValidationReport.from_violations(
                violations,
                checks_run=len(self.suite.constraints),
                repair_attempts=provenance.repair_count,
            )
        )
        return GenerationResult(
            record=record, selection=selection, exchanges=tuple(exchanges)
        )

    def _refuse(
        self,
        task: ConceptReference,
        selection: ModuleSelection,
        exchanges: Sequence[Exchange],
        last_success: Optional[tuple[Any, Sequence[Any]]],
        decode_error: Optional[str],
    ) -> GenerationResult:
        """Return a refuted record rather than raising, when there is one to
        return.

        A hypothesis that failed its constraints is worth keeping: it says which
        constraint a competent model could not satisfy, which is as often a fact
        about the vocabulary as about the model.

        The condition for raising instead is now simply "nothing ever decoded
        across every attempt" -- ``last_success is None`` -- rather than a
        property of whichever attempt happened to run last. Tying the decision
        to the final attempt specifically used to discard a perfectly good
        earlier result whenever the LAST attempt regressed to a decode failure
        after an earlier one had already failed only validation; the bundled
        ``last_success`` pair (see :meth:`generate`) makes that impossible by
        construction, and the raise now only fires when there is genuinely
        nothing to retain.

        The retained graph is stored in the record's scope via
        ``HypothesisRecord.from_root`` -- the same construction path
        :meth:`_admit` uses -- rather than left in an empty scope. Without this,
        "kept as evidence" was only true of the stringified violation messages;
        the fragments those messages are ABOUT were built, checked, and then
        discarded, which defeats the reason for keeping a refused record at
        all. ``KnowledgeScope.admit_all`` accepts this safely even though the
        graph fails domain constraints, because scope admission checks
        structural well-formedness -- declared class, not occurrence-level, has
        identity -- not the acceptance suite's semantic rules.
        """
        if last_success is None:
            raise RepairBudgetExhausted(
                f"{len(exchanges)} attempt(s), last failure: {decode_error}",
                exchanges=exchanges,
            )
        root, violations = last_success
        provenance = self._provenance(selection, exchanges)
        record = HypothesisRecord.from_root(root, task, provenance)
        record.validated(
            ValidationReport.from_violations(
                violations,
                checks_run=len(self.suite.constraints),
                repair_attempts=provenance.repair_count,
            )
        )
        return GenerationResult(
            record=record, selection=selection, exchanges=tuple(exchanges)
        )

    # -- introspection -----------------------------------------------------

    def describe(
        self,
        task: ConceptReference,
        action_label: Optional[str] = None,
        selection: Optional[ModuleSelection] = None,
    ) -> str:
        """Assemble the prompt without calling a model."""
        selection = selection or select_modules(task)
        schema = derive_schema(
            [m.name for m in selection.modules],
            root=self.root_class,
            registry=self.registry,
        )
        return self.prompt_builder.build(
            schema=schema,
            action_label=action_label or str(task),
            degraded=selection.degraded,
            notes=self.notes,
        )


__all__ = [
    "GenerationFailure",
    "GenerationResult",
    "HypothesisGenerator",
    "RepairBudgetExhausted",
    "ResponseUnusable",
]
