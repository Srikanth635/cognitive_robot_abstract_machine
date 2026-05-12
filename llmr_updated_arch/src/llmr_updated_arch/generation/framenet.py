"""FrameNet semantic reasoner for the llmr pipeline.

Annotates :attr:`~llmr_updated_arch.schemas.ActionAnnotationBundle.frames` with a
:class:`~llmr_updated_arch.schemas.FrameNetAnnotation` derived from the original NL
instruction. It reads `semantics.instruction`, so no extra instruction
parameter is needed at construction time.

Usage::

    from llmr_updated_arch.generation.framenet import FrameNetReasoner

    backend = LLMBackend(
        llm=llm,
        instruction="pick up the milk from the table",
        symbol_type=WorldBody,
        reasoners=[FrameNetReasoner(llm=llm)],
    )
    result = next(iter(backend.evaluate(match)))
    annotation = backend.semantics.frames
    print(annotation.frame)           # "Getting"
    print(annotation.core.theme)      # "milk"
    print(annotation.core.source)     # "table"
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.prompts import ChatPromptTemplate

from llmr_updated_arch.integrations.krrood.match_reader import render_resolved_slots
from llmr_updated_arch.generation import Reasoner
from llmr_updated_arch.schemas import FrameNetAnnotation

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from llmr_updated_arch.integrations.krrood.match_reader import MatchSnapshot as MatchData
    from llmr_updated_arch.schemas import ActionAnnotationBundle as ActionSemantics

logger = logging.getLogger(__name__)

# ── Prompt ────────────────────────────────────────────────────────────────────

_FRAMENET_PROMPT_TEMPLATE = """\
You are a computational linguist specializing in Frame Semantics and FrameNet annotation.\
 Your task is to analyze natural language instructions and produce structured semantic\
 representations following FrameNet principles.

FRAME SEMANTICS FUNDAMENTALS:
A frame represents a conceptual structure describing a type of event, relation, or state\
 and the participants involved. Each frame has:
- CORE elements: Conceptually necessary participants that define the frame
- PERIPHERAL elements: Additional, optional circumstances that modify the event

YOUR TASK:
Analyze the semantic structure of the given instruction, identify the primary frame evoked,\
 and map all relevant frame elements.

OUTPUT FORMAT (YAML-style, no braces):

framenet: <semantic_label>            # Snake_case label for the action type
frame: <FrameNet_frame_name>          # Official FrameNet frame (CamelCase)
lexical-unit: <lemma.pos>             # The verb/predicate that evokes the frame

core:
  agent: <WHO>                        # Volitional entity performing action (defaults to robot)
  theme: <WHAT_MOVES>                 # Entity undergoing motion or location change
  patient: <WHAT_AFFECTED>            # Entity being physically modified or directly affected
  instrument: <WITH_WHAT>             # Tool or means used to perform action
  source: <FROM_WHERE>                # Origin location or initial state
  goal: <TO_WHERE>                    # Destination location or target state
  result: <OUTCOME>                   # Resulting state or configuration after action

peripheral:
  location: <WHERE>                   # General spatial setting (broader than source/goal)
  manner: <HOW>                       # Adverbial modification of action execution
  direction: <WHICH_WAY>              # Directional vector of motion (upward, leftward, etc.)
  time: <WHEN>                        # Temporal context or sequence position
  purpose: <WHY>                      # Intended function or reason for action
  quantity: <HOW_MANY>                # Count or measure of entities involved
  portion: <WHICH_PART>               # Specific part of entity involved (top, handle, etc.)
  speed: <HOW_FAST>                   # Rate of action execution (slowly, quickly)
  path: <TRAJECTORY>                  # Route or trajectory taken during motion

ANNOTATION GUIDELINES:

1. FRAME SELECTION:
   - Identify the main action verb and its semantic type
   - Match to appropriate FrameNet frame (e.g., Getting, Placing, Cutting, Filling)
   - Use lexical unit format: <lemma>.<pos> (e.g., "pick_up.v", "place.v")

2. CORE vs PERIPHERAL:
   - CORE: Elements without which the frame wouldn't make sense
   - PERIPHERAL: Circumstantial details that enrich but aren't essential
   - Leave fields null if not applicable — don't force-fill

3. THEME vs PATIENT (CRITICAL DISTINCTION):
   - THEME: Object undergoing MOTION or LOCATION change (pick up, move, place, transfer)
     e.g. "Pick up the bottle" -> theme: bottle
   - PATIENT: Object undergoing PHYSICAL MODIFICATION (cut, break, fill, heat)
     e.g. "Cut the apple" -> patient: apple
   - For motion verbs use THEME (leave patient null)
   - For change-of-state verbs use PATIENT (leave theme null)
   - Some frames use BOTH (e.g., "Pour water into glass" -> theme: water, patient: glass)

4. SOURCE vs GOAL vs LOCATION:
   - Source: Specific starting point of motion/transfer
   - Goal: Specific endpoint or destination
   - Location: Static setting where action occurs (when no clear source/goal)

5. SEMANTIC PRECISION:
   - Extract implicit semantics (e.g., "pick up" -> direction: upward)
   - Infer typical instruments (e.g., cutting -> knife unless specified otherwise)
   - State results explicitly (e.g., "cup is on shelf" not just "placed")
   - Only populate fields that are semantically present or strongly implied

6. GROUNDED PARAMETERS (when provided):
   - Resolved parameters show the exact names of objects already identified in the world
   - Use the EXACT name from resolved parameters as the role filler for that object
   - e.g. if "object_designator: milk_bottle" is resolved, use "milk_bottle" not "the milk"
   - If no resolved parameters are available, derive names from the instruction text

---

EXAMPLES:

Example 1 - Motion Frame (use THEME):
Instruction: Pick up the bottle from the sink

framenet: picking_up_object
frame: Getting
lexical-unit: pick_up.v
core:
  agent: robot
  theme: bottle
  patient: null
  instrument: robot gripper
  source: sink
  goal: robot's grasp
  result: robot holds bottle
peripheral:
  location: kitchen area
  manner: carefully
  direction: upward
  time: null
  purpose: for transport
  quantity: one bottle
  portion: null
  speed: null
  path: null

---

Example 2 - Change-of-State Frame (use PATIENT):
Instruction: Cut the apple into slices with a knife

framenet: cutting_object
frame: Cutting
lexical-unit: cut.v
core:
  agent: robot
  theme: null
  patient: apple
  instrument: knife
  source: whole apple
  goal: null
  result: multiple apple slices
peripheral:
  location: cutting board
  manner: evenly
  direction: downward
  time: during meal preparation
  purpose: for serving
  quantity: one apple
  portion: whole
  speed: null
  path: null

---

CRITICAL REMINDERS:
- Use "null" for empty fields (not empty string, not omit completely)
- THEME = motion/transfer, PATIENT = modification/change
- Only fill peripheral fields that are explicitly stated or strongly implied
- Be precise with frame names (use official FrameNet frames when possible)

Now analyze this instruction:

Instruction: {input_instruction}

Resolved action parameters (already grounded — use these EXACT names as role fillers \
when the role refers to a resolved parameter):
{resolved_slots}

Provide the complete output with all fields, using "null" for non-applicable elements.
"""

_FRAMENET_PROMPT = ChatPromptTemplate.from_template(_FRAMENET_PROMPT_TEMPLATE)

from llmr_updated_arch.hypotheses import (  # noqa: E402
    FRAMENET_PROMPT_VERSION,
    FRAMENET_REASONER_NAME,
)


# ── Reasoner ──────────────────────────────────────────────────────────────────


class FrameNetReasoner(Reasoner):
    """Populate `semantics.frames` with a FrameNet interpretation.

    The reasoner runs one structured LLM call per backend evaluation. If the
    backend has no instruction, it skips silently.

    :param llm: LangChain-compatible chat model used for FrameNet annotation.
    """

    REASONER_NAME = FRAMENET_REASONER_NAME
    PROMPT_VERSION = FRAMENET_PROMPT_VERSION

    def __init__(self, llm: "BaseChatModel") -> None:
        self._chain = _FRAMENET_PROMPT | llm.with_structured_output(
            FrameNetAnnotation, method="function_calling"
        )

    def annotate(
        self,
        semantics: "ActionSemantics",
        match_data: "MatchData",
        world_context: str,
    ) -> None:
        instruction = semantics.instruction
        if not instruction:
            logger.debug("FrameNetReasoner: no instruction on semantics — skipping.")
            return

        semantics.frames = self._chain.invoke({
            "input_instruction": instruction,
            "resolved_slots": render_resolved_slots(match_data),
        })
