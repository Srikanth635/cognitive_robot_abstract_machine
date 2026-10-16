# Predicting where a pour ends: design note

The resolver answers a pouring designator without knowing where the pitcher's mouth will
end for the values it chooses: with coraplex's defaults the mouth ends 0.188 m beside the
bowl's opening, and only the simulated pour (`SimulatedPouring`, after the answer) tells
the resolver so. Decided 2026-10-16: the resolver predicts it before answering, with
coraplex's code read and never changed.

## 1. What decides where the mouth ends

coraplex's `PouringAction._pour_poses()` places the tool frame beside the target, moved
sideways by `pour_side_offset + sin(tilt_angle) × the mouth's height above the tool
frame`, back by `pour_approach_offset`, up by `pour_height`, and tilted by `tilt_angle`.
In a trial the poses are computed when the plan is built, before the pitcher is picked
up, so the mouth's height is taken from the pitcher on the counter (0.262 m) and the
sideways shift does not bring the tilted mouth back.

Checked against the seven distinct pours recorded in the phase 4 comparison
(`knowledge_perspectives.md` §13): the tool pose `_pour_poses()` gives in the world before
anything is performed, carried with how the gripper holds the pitcher after the pick-up,
puts the mouth within 5 mm of where every simulated pour measured it, the default miss
(0.188 m) and every corrected landing alike.

## 2. The prediction

`predict = world_T_tool(the values) · tool_T_source · the mouth in the source's frame`

| Part | From | When |
|---|---|---|
| `world_T_tool` | coraplex's `_pour_poses()` for a `PouringAction` with the values, planned in the scene's world before anything is performed | per prediction, in the resolver's process, in milliseconds |
| `tool_T_source`, the robot's heading after it faced the pitcher | a simulated park, torso, navigate and pick-up, as a trial performs them | once per scene and arm, in a process of its own, kept with the run |
| the mouth in the source's frame | the top centre of the source's bounding box, as `PourOutcome` measures it | per prediction |

The prediction is reported as `PourOutcome` reports a pour: the mouth's distance from the
opening's centre, ahead of and right of it as seen from the robot, its height above the
opening, and whether it lands in the opening.

## 3. Offered to the resolver

- llm_reasoner gains a seam beside `OutcomeCheck`: an `OutcomePrediction` offers tools
  predicting what an answer brings about and the rule saying how to use them; the
  backend offers its tools beside the world tools and states its rule. llm_reasoner
  imports nothing of coraplex for it.
- cram_viz's `PourPrediction` implements it with one tool, `predict_pour_end`, taking a
  pour's values as the answer writes them and answering with the predicted outcome. The
  simulated pour stays as the check after the answer.

## 4. Measured

`experiments.frame_comparison` gains the arm "with the prediction": first-try success on
`pour_geometry_open`, against the three runs without it, which all missed.
