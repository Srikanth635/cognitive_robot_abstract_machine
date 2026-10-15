"""
The resolver's prompt vocabulary: the sections its prompts are made of, the rules its
model answers by, and the keys of the data its prompts hold, written in the structure
:mod:`llm_reasoner.prompt_structure` defines.
"""

from __future__ import annotations

from enum import StrEnum

from llm_reasoner.prompt_structure import (  # noqa: F401 (the resolver's names for them)
    SECTION_SEPARATOR,
    PromptBlock,
    PromptRule,
    render_blocks,
)

# %% sections


class PromptSection(StrEnum):
    """
    The sections a prompt is made of, each named by its tag.
    """

    ROLE = "role"
    """
    What the model is asked to do.
    """

    INPUTS = "inputs"
    """
    What each section of a request means.
    """

    RULES = "rules"
    """
    How the model answers.
    """

    REQUEST = "request"
    """
    The query, stated in English.
    """

    TASK = "task"
    """
    What the caller is trying to achieve.
    """

    SITUATION = "situation"
    """
    What is currently true.
    """

    ACTION = "action"
    """
    The action a request serves, stated as its own request and given values.
    """

    SOURCE = "source"
    """
    The definition of a type holding slots.
    """

    DESCRIPTION = "description"
    """
    The docstring of a type holding slots.
    """

    SLOTS = "slots"
    """
    The open slots, as data.
    """

    GIVEN = "given"
    """
    The values the query fixes, as data.
    """

    OUTPUT = "output"
    """
    The shape of the answer.
    """

    REJECTED_ANSWER = "rejected_answer"
    """
    An earlier answer that was refused.
    """

    REJECTION = "rejection"
    """
    Why the earlier answer was refused, as data.
    """


class BlockAttribute(StrEnum):
    """
    The attributes a section's tag may carry.
    """

    TYPE = "type"
    """
    The type a section belongs to.
    """

    KIND = "kind"
    """
    The kind of rejection a section states.
    """


# %% rules


class RuleName(StrEnum):
    """
    The rules a model answers by, each named for what it governs.
    """

    ANSWER_FORMAT = "answer_format"
    """
    How the answer is written as a whole.
    """

    ELEMENT_KEYS = "element_keys"
    """
    Which keys each element of the answer holds.
    """

    VALUE_SLOT = "value_slot"
    """
    How a slot taking a plain value is answered.
    """

    ESTABLISHED_VALUES = "established_values"
    """
    How value slots asking for facts rather than choices are answered.
    """

    DEFAULT = "default"
    """
    What a slot's default stands for.
    """

    UNION = "union"
    """
    How a slot accepting several types is answered.
    """

    KIND_SLOT = "kind_slot"
    """
    How a slot asking for a kind is answered.
    """

    REFERENT_SLOT = "referent_slot"
    """
    How a slot asking for an individual is answered.
    """

    FINDING_CANDIDATES = "finding_candidates"
    """
    How the candidates of a slot that lists none are found.
    """

    MEANING_HOLDER = "meaning_holder"
    """
    Where the meaning of a candidate is found.
    """

    KNOWN_CANDIDATE = "known_candidate"
    """
    What must be known of a candidate before it is chosen.
    """

    NO_FITTING_CANDIDATE = "no_fitting_candidate"
    """
    How to say that no candidate of a slot fits.
    """

    EQUALLY_FITTING = "equally_fitting"
    """
    How to say that several candidates of a slot fit equally.
    """

    UNSATISFIABLE = "unsatisfiable"
    """
    How to say that the request's conditions cannot all hold.
    """

    CONDITIONS = "conditions"
    """
    How the request's conditions bind the answer.
    """

    GIVEN = "given"
    """
    How the values the query fixes are used.
    """

    CANDIDATES = "candidates"
    """
    How the elements of the answer differ, and how many there are.
    """

    WHY = "why"
    """
    What the explanation of an element says.
    """

    REPAIR = "repair"
    """
    How a refused answer is answered again.
    """

    TEXT_ANSWER = "text_answer"
    """
    Where the answer goes when the model may call tools.
    """

    KNOWLEDGE = "knowledge"
    """
    How the knowledge about the requested action is read and used.
    """

    FRAME_KNOWLEDGE = "frame_knowledge"
    """
    How the action's frame and the perspectives on it are read and used, when they are
    offered.
    """

    GROUNDS = "grounds"
    """
    How an element names what its choices rest on.
    """


# %% data keys


class SlotKey(StrEnum):
    """
    The keys of a slot's entry in the slots section, and of a given value's entry.
    """

    PATH = "path"
    """
    The slot's access path, which is also its key in the answer.
    """

    TYPE = "type"
    """
    The name of the slot's declared type.
    """

    MEANING = "meaning"
    """
    The slot's docstring.
    """

    WITHIN = "within"
    """
    The nested description the slot is filled through, when it sits inside one.
    """

    ACCEPTS = "accepts"
    """
    What the slot may be answered with.
    """

    DEFAULT = "default"
    """
    The value the slot's class uses when none is given.
    """

    CANDIDATES = "candidates"
    """
    The individuals or kinds the slot chooses from.
    """

    VALUE = "value"
    """
    The value the query gives for a slot.
    """


class AcceptKey(StrEnum):
    """
    The keys of one entry of what a slot accepts.
    """

    TYPE = "type"
    """
    A kind of value accepted.
    """

    ENUM = "enum"
    """
    The names of the enum members accepted.
    """

    HOLDS = "holds"
    """
    What an accepted value names, such as a candidate's handle.
    """


class AcceptedType(StrEnum):
    """
    The kinds of value a slot may accept.
    """

    NUMBER = "number"
    """
    A JSON number.
    """

    STRING = "string"
    """
    A JSON string.
    """

    BOOLEAN = "boolean"
    """
    A JSON boolean.
    """

    NULL = "null"
    """
    Nothing, leaving the slot empty.
    """


class CandidateReference(StrEnum):
    """
    How an answer names one of a slot's candidates.
    """

    TOKEN = "token"
    """
    By the integer token it is listed under.
    """

    HANDLE = "handle"
    """
    By its handle, as the world tools give it out.
    """


class CandidateKey(StrEnum):
    """
    The keys describing the candidates of a slot.
    """

    COUNT = "count"
    """
    How many candidates the slot has.
    """

    KIND = "kind"
    """
    The identifier of the kind the candidates are of.
    """

    LISTED = "listed"
    """
    The candidates themselves, when few enough to list.
    """

    TOKEN = "token"
    """
    The integer token a listed candidate is answered with.
    """

    HANDLE = "handle"
    """
    The handle a listed candidate is answered with.
    """

    DESCRIPTION = "description"
    """
    What a listed candidate is, line by line.
    """

    HELD_AS = "held_as"
    """
    The fields of other individuals that hold a listed candidate.
    """


class SchemaKey(StrEnum):
    """
    The JSON schema keywords the output section uses.
    """

    TYPE = "type"
    """
    The type of the value described.
    """

    ITEMS = "items"
    """
    What each element of an array is.
    """

    MIN_ITEMS = "minItems"
    """
    The fewest elements an array holds.
    """

    MAX_ITEMS = "maxItems"
    """
    The most elements an array holds.
    """

    PROPERTIES = "properties"
    """
    The keys of an object and what each holds.
    """

    REQUIRED = "required"
    """
    The keys an object must hold.
    """

    ADDITIONAL_PROPERTIES = "additionalProperties"
    """
    Whether an object may hold keys not named.
    """

    ANY_OF = "anyOf"
    """
    The schemas a value may match, any one of them.
    """


class SchemaType(StrEnum):
    """
    The JSON schema types the output section uses.
    """

    ARRAY = "array"
    """
    A JSON array.
    """

    OBJECT = "object"
    """
    A JSON object.
    """

    STRING = "string"
    """
    A JSON string.
    """
