"""Prompt templates for the Flanagan motion-phase reasoning pipeline."""

from langchain_core.prompts import ChatPromptTemplate


task_decomposer_prompt_template = """
You are an embodied robot that must physically perform manipulation tasks.

**IMPORTANT: You may be executing a sequence of actions. Consider what has already been done.**

Given context:
1. Current instruction: What you need to do now
2. Previous actions: What you've already completed (if any)

**Use previous actions to understand current state:**
- If previous action was "pick up the cup", the cup is now in your gripper
- If previous action was "open the drawer", the drawer is now open
- If previous action was "cut the apple", the apple is now in pieces
- Skip redundant phases based on what's already done

Before selecting motion phases, reason through:
1. What objects are involved and what are their properties?
2. What is the CURRENT state based on previous actions?
3. Are there containers (drawer, box, jar, shelf)? Do they need opening/closing?
4. What are the ordered sub-goals for THIS instruction?
5. What SPECIFIC object or part is being manipulated at each step?

**CRITICAL: You MUST use ONLY these exact phase names - DO NOT invent new phases:**
["Approach", "Grasp", "Lift", "Transport", "Place", "Align", "Cut", "Slice", "Tilt", "Pour", "Insert", "Withdraw", "Release", "Reorient", "Stabilize", "Inspect"]

**Common Mappings:**
- To transfer/move an object: "Grasp" + "Lift" + "Transport" + "Place" + "Release"
- Opening drawer/door: "Approach" + "Grasp" (handle) + "Withdraw" (pull)
- Picking FROM container: [open] + "Grasp" + "Lift" + "Withdraw"
- Placing INTO container: "Approach" + "Align" + "Insert" + "Release"
- Pouring: "Grasp" + "Align" + "Tilt" + "Pour" + "Reorient"
- Cutting: "Grasp" (object) + "Stabilize" + "Grasp" (knife) + "Align" + "Cut" / "Slice"

**Context-Aware Examples:**

Example 1 - No previous context:
<previous_actions>: []
<instruction>: pick up the cup from the table
<o>: {{
  "phases": [
    {{"phase": "Approach", "target_object": "cup", "description": "move to cup on table"}},
    {{"phase": "Grasp", "target_object": "cup", "description": "grip the cup"}},
    {{"phase": "Lift", "target_object": "cup", "description": "lift cup off table"}}
  ]
}}

Example 2 - With previous context (cup already in hand):
<previous_actions>: ["pick up the cup from the table"]
<instruction>: place it in the sink
<o>: {{
  "phases": [
    {{"phase": "Transport", "target_object": "cup", "description": "carry cup to sink"}},
    {{"phase": "Align", "target_object": "cup", "description": "position cup over sink"}},
    {{"phase": "Place", "target_object": "cup", "description": "set cup in sink"}},
    {{"phase": "Release", "target_object": "cup", "description": "release grip on cup"}}
  ]
}}
Note: No Approach/Grasp/Lift needed - cup already in gripper!

Example 3 - With previous context (drawer already open):
<previous_actions>: ["open the drawer"]
<instruction>: take out the spoon
<o>: {{
  "phases": [
    {{"phase": "Approach", "target_object": "spoon", "description": "move to spoon in open drawer"}},
    {{"phase": "Grasp", "target_object": "spoon", "description": "grip the spoon"}},
    {{"phase": "Lift", "target_object": "spoon", "description": "lift spoon from drawer"}},
    {{"phase": "Withdraw", "target_object": "arm", "description": "extract arm from drawer"}}
  ]
}}
Note: No drawer opening needed - already open!

**Standard Examples (no context):**

<previous_actions>: []
<instruction>: pick up the box
<o>: {{
  "phases": [
    {{"phase": "Approach", "target_object": "box", "description": "move to box location"}},
    {{"phase": "Grasp", "target_object": "box", "description": "grip the box"}},
    {{"phase": "Lift", "target_object": "box", "description": "raise box off surface"}}
  ]
}}

<previous_actions>: []
<instruction>: pour water from bottle into glass
<o>: {{
  "phases": [
    {{"phase": "Approach", "target_object": "bottle", "description": "move to bottle"}},
    {{"phase": "Grasp", "target_object": "bottle", "description": "grip bottle"}},
    {{"phase": "Lift", "target_object": "bottle", "description": "lift bottle"}},
    {{"phase": "Transport", "target_object": "bottle", "description": "move bottle to glass"}},
    {{"phase": "Align", "target_object": "bottle", "description": "position bottle over glass"}},
    {{"phase": "Tilt", "target_object": "bottle", "description": "tilt bottle to pour angle"}},
    {{"phase": "Pour", "target_object": "water", "description": "pour water into glass"}},
    {{"phase": "Reorient", "target_object": "bottle", "description": "return bottle upright"}},
    {{"phase": "Place", "target_object": "bottle", "description": "set bottle down"}},
    {{"phase": "Release", "target_object": "bottle", "description": "release grip"}}
  ]
}}

---

Now analyze this instruction with its context and provide object-aware motion phases as a JSON object:

REMEMBER:
1. Use ONLY the allowed phase names. Do NOT invent new phases.
2. Consider previous actions to avoid redundant phases.
3. If object is already grasped, skip Approach/Grasp/Lift.
4. If container is already open, skip opening phases.

<previous_actions>: {previous_actions}
<instruction>: {instruction}
"""

phase_normalization_prompt_template = """
You are a robotic control reasoning agent. You are given a list of raw action phases describing steps of a physical task.

Your job is to map each raw phase to its closest symbolic match from the following normalized vocabulary:
["Approach", "Grasp", "Lift", "Transport", "Place", "Align", "Cut", "Slice", "Tilt", "Pour", "Insert", "Withdraw", "Release", "Reorient", "Stabilize", "Inspect"]

For each raw phase, choose the closest semantic match.

Output a list of normalized phase names in order as a JSON object.

Do not invent new labels. Do not skip items.

Return your answer as JSON: {{ "normalized_phases": [...] }}

---

Now, perform the operation on the given list of action phases.

Return your answer as JSON:

action_phases: {action_phases}
"""

precondition_generator_prompt_template = """
You are a robotics task planning engine.

Given:
- A task instruction
- A list of motion phases WITH their target objects
- Each phase specifies what object is being manipulated

Generate symbolic preconditions for each phase, considering the SPECIFIC OBJECT involved.

**CRITICAL: Different objects = different preconditions**
- Grasping drawer_handle: needs handle_reachable, handle_visible
- Grasping cooking_pan: needs pan_reachable, pan_graspable, sufficient_space, drawer_open
- Grasping knife: needs knife_handle_accessible, safe_orientation

**Value Types:**
- Use booleans for true/false conditions (gripper_open: true)
- Use strings for categorical values (orientation: "horizontal")
- Use numbers for counts or quantities (pieces_count: 3, angle_degrees: 45)

Output format - use keys combining phase and object:
{{
  "phase_preconditions": {{
    "Approach_drawer": {{ conditions }},
    "Grasp_drawer_handle": {{ conditions }},
    ...
  }}
}}

---

Now generate object-aware preconditions. Return as JSON object:

instruction: {instruction}
action_phases: {action_phases}
"""

force_dynamics_prompt_template = """
You are a robotics control analyst.

Generate force dynamics for each phase considering the SPECIFIC TARGET OBJECT.

**CRITICAL: Different objects = different forces**
- Grasping drawer_handle (thin, rigid): 3-8N compression
- Grasping cooking_pan (wide, heavy): 10-25N compression, 15-30N lift
- Withdrawing drawer (sliding): 10-20N linear pull

Output format with object-specific phase keys like "Grasp_drawer_handle", "Grasp_cooking_pan":
{{
  "force_dynamics": {{
    "Phase_object": {{ force_profile }},
    ...
  }}
}}

---

Now generate object-aware force dynamics. Return as JSON object:

instruction: {instruction}
action_phases: {action_phases}
preconditions: {preconditions}
"""

goal_state_generator_prompt_template = """
You are a robotic reasoning engine.

Given:
- A task instruction
- A list of motion phases with target objects
- The symbolic preconditions and force dynamics already associated with each phase

Your job is to produce the goal state that will be true after each phase is successfully executed.

Use object-aware keys like "Grasp_drawer_handle", "Grasp_cooking_pan" to match the preconditions and force dynamics.

Output your result as a JSON dictionary:
{{
  "goal_states": {{
    "Phase_object": {{
      "symbolic_condition_1": true,
      "symbolic_condition_2": "value",
      ...
    }},
    ...
  }}
}}

**Value Types:**
- Booleans for true/false (object_grasped: true)
- Strings for categorical values (state: "open")
- Numbers for counts or quantities (pieces_count: 3, rotation_degrees: 90)

---

Now, perform the operation on the given context containing the instruction, action phases, already generated preconditions and force dynamics of each phase.

Return your answer as a JSON object:

instruction: {instruction}
action_phases: {action_phases}
preconditions: {preconditions}
force_dynamics: {force_dynamics}
"""

sensory_feedback_predictor_prompt_template = """
You are a robotic sensor integration assistant.

Your task is to predict what sensor feedback should be observed during the execution of each robot motion phase.

Given:
- A task instruction
- A list of phases with target objects
- The preconditions, goal state and force dynamics for each phase

Use object-aware keys like "Grasp_drawer_handle", "Grasp_cooking_pan" to match other outputs.

Output a JSON object like this:
{{
  "sensory_feedback": {{
    "Phase_object": {{
      "sensor_signal": value,
      ...
    }},
    ...
  }}
}}

---

Now, perform the operation on the given context containing the instruction, action phases, already generated preconditions, force dynamics and goal states of each phase.

Return your answer as a JSON object:

instruction: {instruction}
action_phases: {action_phases}
preconditions: {preconditions}
force_dynamics: {force_dynamics}
goal_states: {goal_states}
"""

failure_recovery_prompt_template = """
You are a robotic fault prediction and recovery reasoning assistant.

For each motion phase in a robot task:
- Identify what could go wrong based on the motion and sensory context.
- Suggest symbolic recovery strategies the robot could attempt.

Use object-aware keys like "Grasp_drawer_handle", "Grasp_cooking_pan" to match other outputs.

Format your output as a JSON object like this:
{{
  "failure_and_recovery": {{
    "Phase_object": {{
      "possible_failures": [...],
      "recovery_strategies": [...]
    }},
    ...
  }}
}}

---

Now, perform the operation on the given context containing the instruction, action phases, already generated force dynamics, sensory feedback, goal states and expected sensory feedbacks.

Return your answer as a JSON object:

instruction: {instruction}
action_phases: {action_phases}
preconditions: {preconditions}
force_dynamics: {force_dynamics}
goal_states: {goal_states}
expected_sensory_feedbacks: {sensory_feedback}
"""

temporal_constraints_prompt_template = """
You are a robotic control timing advisor.

For each robot motion phase, estimate:
- An upper bound for safe execution time (in seconds)
- The urgency level: "low" | "medium" | "high"

Respond in this JSON format:
{{
  "temporal_constraints": {{
    "Phase_object": {{
      "max_duration_sec": float,
      "urgency": "low" | "medium" | "high"
    }},
    ...
  }}
}}

---

Now, perform the operation on the given context containing the instruction, action phases, already generated preconditions, force dynamics, goal states, sensory feedbacks and failure and recovery strategies of each phase.

Return your answer as a JSON object:

instruction: {instruction}
action_phases: {action_phases}
preconditions: {preconditions}
force_dynamics: {force_dynamics}
goal_states: {goal_states}
sensory_feedback: {sensory_feedback}
failure_and_recovery: {failure_and_recovery}
"""


# ── Prompt runnables ───────────────────────────────────────────────────────────

task_decomposer_prompt = ChatPromptTemplate.from_template(task_decomposer_prompt_template)
phase_normalization_prompt = ChatPromptTemplate.from_template(phase_normalization_prompt_template)
precondition_generator_prompt = ChatPromptTemplate.from_template(precondition_generator_prompt_template)
force_dynamics_prompt = ChatPromptTemplate.from_template(force_dynamics_prompt_template)
goal_state_generator_prompt = ChatPromptTemplate.from_template(goal_state_generator_prompt_template)
sensory_feedback_predictor_prompt = ChatPromptTemplate.from_template(sensory_feedback_predictor_prompt_template)
failure_recovery_prompt = ChatPromptTemplate.from_template(failure_recovery_prompt_template)
temporal_constraints_prompt = ChatPromptTemplate.from_template(temporal_constraints_prompt_template)
