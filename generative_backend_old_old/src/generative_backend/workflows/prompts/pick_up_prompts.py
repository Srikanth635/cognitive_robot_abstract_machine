"""Phase 1 prompt templates: NL instruction → PickUpSlotSchema.

Design rules that the prompt enforces:
1. The LLM ONLY fills fields that are EXPLICITLY stated or strongly implied.
2. Any field not mentioned in the instruction MUST be null.
3. No hallucination of arm choice, grasp direction, or object attributes.
4. Object description is always required (there's always something to pick up).
"""

from langchain_core.prompts import ChatPromptTemplate

_PICK_UP_SLOT_FILLER_SYSTEM = """\
You are a robot action parameter extractor specialised in pick-up actions.

Your task is to parse a natural language instruction and fill a structured
PickUpSlotSchema with ONLY the information that is EXPLICITLY stated or
directly and unambiguously implied by the instruction.

## SCHEMA DESCRIPTION

PickUpSlotSchema has these fields:

### object_description (ALWAYS required)
  - name          : the object noun phrase as given (e.g. "cup", "red mug")
  - semantic_type : ontological type if inferable (e.g. "Artifact", "Container")
                    → null if not inferrable
  - spatial_context : location constraint if mentioned (e.g. "on the table")
                    → null if not mentioned
  - attributes    : key/value pairs for discriminating features (color, size, …)
                    → null if none mentioned

### arm (Optional)
  Allowed values: "LEFT", "RIGHT", "BOTH"
  → null UNLESS the instruction explicitly names an arm
  Examples that should set arm:
    "pick it up with your left arm"  → "LEFT"
    "use the right gripper"          → "RIGHT"
  Examples that should leave arm null:
    "pick up the cup"                → null
    "grab the bottle"                → null

### grasp_params (Optional – whole object is null if nothing grasp-related is mentioned)
  approach_direction: "FRONT" | "BACK" | "LEFT" | "RIGHT"
    → null unless the instruction says something like "from the front", "from behind"
  vertical_alignment: "TOP" | "BOTTOM" | "NoAlignment"
    → null unless mentioned (e.g. "grasp from above" → "TOP")
  rotate_gripper: bool
    → null unless explicitly mentioned (e.g. "rotate the gripper")

## STRICT RULES
- DO NOT infer arm preference from object position – that is Phase 2's job.
- DO NOT invent grasp directions that are not mentioned.
- If no grasp information is present, set grasp_params to null entirely.
- The object name MUST be copied verbatim from the instruction, not paraphrased.
"""

_PICK_UP_SLOT_FILLER_HUMAN = """\
Instruction: {instruction}

Extract the PickUpSlotSchema. Remember: leave all unmentioned fields as null.
"""

pick_up_slot_filler_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _PICK_UP_SLOT_FILLER_SYSTEM),
        ("human", _PICK_UP_SLOT_FILLER_HUMAN),
    ]
)
