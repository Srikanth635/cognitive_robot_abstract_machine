"""Prompt templates for FrameNet semantic annotation."""

from langchain_core.prompts import ChatPromptTemplate


framenet_prompt_template = """
You are a computational linguist specializing in Frame Semantics and FrameNet annotation. Your task is to analyze natural language instructions and produce structured semantic representations following FrameNet principles.

FRAME SEMANTICS FUNDAMENTALS:
A frame represents a conceptual structure describing a type of event, relation, or state and the participants involved. Each frame has:
- CORE elements: Conceptually necessary participants that define the frame
- PERIPHERAL elements: Additional, optional circumstances that modify the event

YOUR TASK:
Analyze the semantic structure of the given instruction, identify the primary frame evoked, and map all relevant frame elements.

OUTPUT FORMAT (YAML-style, no braces):

framenet: <semantic_label>            # Snake_case label for the action type
frame: <FrameNet_frame_name>          # Official FrameNet frame (CamelCase)
lexical-unit: <lemma.pos>             # The verb/predicate that evokes the frame

core:
  agent: <WHO>                        # Volitional entity performing action (typically: robot, user, person, defaults to robot)
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
   - Leave fields empty (null or omit) if not applicable - don't force-fill

3. THEME vs PATIENT (CRITICAL DISTINCTION):
   - **THEME**: Object undergoing MOTION or LOCATION change (pick up, move, place, transfer)
     -> "Pick up the bottle" -> theme: bottle (it's being moved)
   - **PATIENT**: Object undergoing PHYSICAL MODIFICATION or being directly acted upon (cut, break, fill, heat)
     -> "Cut the apple" -> patient: apple (it's being changed/modified)
   - For MOTION verbs -> use THEME (leave patient empty)
   - For CHANGE-OF-STATE verbs -> use PATIENT (leave theme empty)
   - Some frames use BOTH (e.g., "Pour water into glass" -> theme: water, patient: glass)

4. SOURCE vs GOAL vs LOCATION:
   - Source: Specific starting point of motion/transfer
   - Goal: Specific endpoint or destination
   - Location: Static setting where action occurs (use when no clear source/goal)

5. SEMANTIC PRECISION:
   - Extract implicit semantics (e.g., "pick up" -> direction: upward)
   - Infer typical instruments (e.g., cutting -> knife unless specified otherwise)
   - State results explicitly (e.g., "cup is on shelf" not just "placed")
   - Only populate fields that are semantically present or strongly implied

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

Provide the complete YAML-style output with all fields, using "null" for non-applicable elements.
"""

framenet_prompt = ChatPromptTemplate.from_template(framenet_prompt_template)
