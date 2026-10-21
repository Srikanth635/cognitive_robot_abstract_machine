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

## 4. Measured (2026-10-16)

`experiments.frame_comparison`'s arm "with the prediction", three runs of
`pour_geometry_open` against the three recorded without it:

| | Without the prediction | With the prediction |
|---|---|---|
| First pour lands | 0 of 3 (0.188 m off) | 3 of 3 |
| Pours tried | 2, 2, 2 | 1, 1, 1 |
| Final distance from the opening's centre | 0.018, 0.005, 0.014 m | 0.011, 0.005, 0.011 m |
| Resolver cost per run | $0.083–0.133 | $0.046–0.091 |

Every run predicted three pours: the defaults (0.188 m off), the left arm (the pick-up
fails), and a corrected side offset (−0.185 or −0.188 m, landing), which it answered;
each prediction of the final pour was within 5 mm of where the simulated pour measured
it. Two earlier attempts found two faults, both fixed before these runs: the measuring
process wrote its held source as `__main__.HeldSource` (it now runs a module of its
own, `hold_measurement`), and a pick-up the arm cannot make raised instead of being
predicted as a failed motion.

## 5. Measured with the containers open (2026-10-18)

Two runs of `objects_open` with the prediction against the two recorded without it
(`comparison_open`). Its query leaves the source, the target, the arm and the tilt open;
the side offset, approach offset and height keep coraplex's defaults, so the tilt is the
only value that moves the mouth.

| | Without the prediction | With the prediction |
|---|---|---|
| Containers chosen | the pouring cup into the bowl, both runs | the same |
| First pour lands | 0 of 2 | 0 of 2 |
| Any pour lands | 0 of 2 | 1 of 2 (the third, at a tilt of 0.2 rad) |
| Resolver cost per run | $0.243, $0.271 | $0.391, $0.264 |

The prediction did not help here, for three reasons the runs show:

- **Only the tilt is open.** With the default geometry the mouth ends 0.188 m beside the
  opening; the prediction lands it only near 3 rad (the pitcher almost upside down) or
  near 0 rad (barely tilted). The rule still names `pour_side_offset`,
  `pour_approach_offset` and `pour_height` as the values to change, which this query
  does not leave open.
- **The prediction does not know which tilts the arm can reach.** It predicted 2.8, 2.9,
  3.0 and 3.07 rad to land within 0.001–0.053 m; all five pours tried at those tilts
  failed as motions (`MotionDidNotFinish` on the tool's orientation). A pick-up the arm
  cannot make is predicted as a failed motion, but the pour's own motion is not checked.
- **The check accepts a pour that pours nothing.** The one landing pour was at 0.2 rad,
  11°: the mouth ends over the opening, which is all `PourOutcome` measures, but a
  pitcher tilted that little does not flow.

What would make the prediction help here, all on the resolver's side: predict whether
the pour's own motion can be made, not only the pick-up; name in the rule only the
values the query leaves open; and have the check require the tilt the knowledge states
for flow onset.

## 6. The check requires the tilt for flow (2026-10-19)

The third of those is done. cram_viz's `FlowOnsetKnowledge` reads, from the hypotheses the
resolver consulted (not every stored one), the lower bound of the threshold each flow onset
event crosses, converted to radians from whatever angle unit the range states (the lowest if
several), and
`SimulatedPouring` refuses a pour that lands in the opening but is tilted less, stating
`tilt_reaches_flow_onset` and `flow_onset_tilt_radians` in the outcome so the resolver
reads what it needs. The last allowed trial is still accepted, and the tilt is left
unchecked while no consulted hypothesis states a flow onset threshold as an angle.

Measured live on `objects_open` (2026-10-19, Sonnet 5.5, two runs of the default arm and
two with the prediction, the earlier knowledge store reused): no pour landed in any of the
four runs, so the check had no landing pour to refuse. The default arm tried 1.85, 0.6,
1.5708 rad (all missing, or a failed motion with the left arm); the prediction arm tried
2.09 to 3.1 rad, and every pour at 2.8 rad or more failed as a motion. The 0.2 rad landing
of §5 did not recur. Resolver cost per run: $0.254 and $0.191 (default), $0.310 and
$0.283 (with the prediction), $1.04 in all. The check is therefore covered by tests only;
the failed motions of the prediction arm are the second fix of §5, still open.

## 7. The pour's own motion predicted (2026-10-19)

The second fix of §5. After the simulated pick-up, `HeldSource` also keeps the position of
every joint (where the base stands, the torso and the arms), and `PourReach` asks the
semantic digital twin's inverse kinematics, from that state and with the base fixed,
whether the arm can move the tool through the poses from the pre-pour pose to the pour
pose, its orientation turning in equal steps of at most `ApproachTolerances.max_rotation_per_step`
(0.5 rad). An arm that cannot is predicted as a failed motion, as a failed pick-up is
(`PourUnreachable`).

Checked against the simulated pour on the right arm and the pitcher, `objects_open`'s
scene, by tilt (rad): 0.2, 1.85, 1.95, 2.09 and 2.6 finish and are predicted reachable;
2.8 and 3.0 fail and are predicted unreachable. A single solve from the pick-up's end
wrongly refused 2.09, which stepping the orientation fixes. 2.2 and 2.4 fail in the
simulation but are predicted reachable: the joints can get there, so what stops the
motion there is not the kinematics (collisions are not asked about). A pour predicted
unreachable is therefore certain to fail; one predicted reachable may still fail. Not yet
measured live with the resolver.

The step size and the solver's iteration limit were checked on 11 tilts (0.2 to 3.0 rad) with
the iteration limit at 200 and at 1000 and one to twenty steps. With three or more steps the
two limits give the same answers, so the code passes none and the solver's own default
applies. One or two steps give wrong answers whichever the limit, because a single solve over
a large turn stops short; a test pins the same predictions for steps of 0.15, 0.5 and 0.7 rad,
which for a 3 rad pour is 20, 6 and 5 steps. The band comes from this robot, scene and arm.

The tests no longer write which tilts the arm reaches: they try nine tilts spread evenly from
upright to upside down, read the labels from the prediction, and check that every pour
predicted out of reach fails in the simulation (a slow test), that the labels are the same
across step sizes, and that the grid holds both kinds. The converse is not claimed, since
the simulation also fails where only a collision stops it.

## 8. The robot turns to face the bowl before pouring (2026-10-21)

Found in a cram_viz run of `objects_given` (pitcher into bowl, arm and tilt open): the
resolver's third pour, 2.9 rad on the right arm, failed as a motion (`MotionDidNotFinish`)
and its fourth answer was cut off, so the best pour kept missed by 0.114 m. The plan
(`PouringExecution.steps_before_pour`) had the robot stand facing the *pitcher*, pick it up and
pour from that same standing; the pour poses are around the bowl, which lies 0.389 rad to the
left of that heading, 0.66 m away.

Measured by inserting a `NavigateAction` after the pick-up (pitcher, bowl and right arm of
`simple_pouring`, about 80 simulated pours):

- Standing elsewhere changes nothing for the pours that finish at moderate tilts: the mouth
  ends 0.187–0.189 m from the opening's centre at 1.85 rad wherever the robot stands, since the
  sideways miss comes from the pour pose relative to the bowl (§5), not from the base.
- Turning in place changes which tilts the arm can make. With no turn 2.8, 2.9 and 3.0 rad
  all fail; with a turn of 0.3–0.5 rad 2.8 and 2.9 land (0.036–0.053 m) and 3.0 lands from 0.4
  rad (0.016 m).
- Reach is jagged in this region. Facing the bowl exactly (0.389 rad) lands 2.8 rad (0.049 m)
  and 2.9 rad (0.035 m) but fails 3.0, and 2.6 rad, which finishes without the turn (0.094 m
  off), fails with it. Only trying a pour tells.
- The pours that land are nearly upside down (2.8–3.0 rad, 160–172°), not natural pours.

The plan now turns the robot in place to face the target container after the pick-up
(`facing_pose`), in `steps_before_pour`, so the trials, Execute and the pour prediction (which
simulates the same pick-up) all stand there. From the turned base the reach check's solver, at
its default of 200 iterations, stopped short for two recorded pours that the simulation performs
(approach offsets of 0.04 m, predicted out of reach), and a 0.7 rad step refused 1.96 rad; the
check now allows 400 iterations (`ApproachTolerances.max_solver_iterations`), which restores
both. Not done: choosing the stance (other than facing the target) by search.

Live re-run of the same `objects_given` (Sonnet 5.5, every perspective, the liquid knowledge
served from the store, no pour prediction): it did not land. The resolver tried right 1.85 rad
(0.187 m off), left 1.85 (motion failed), right 1.57 (0.199 m) and right 0.6 (0.123 m), kept the
last, and said itself that the right arm probably cannot land the mouth within the opening.
It did not try the near-upside-down tilts the first run reached (2.9 rad), which now land from
the turned base, so facing the bowl widens what can land but does not make the model look
there; $0.60 and 4.4 minutes, all of it the resolver's. The pour prediction (off in cram_viz)
is what names tilts near 3 rad as the ones that land (§5); with it the model would be told.

The same run with the pour prediction on (`predicts_pours`, same store and setup): it did not
land either, and the resolver refused ("no choice of arm and tilt both lands the mouth and
reaches the flow tilt"); $0.47, 4.7 minutes, 3 pours tried and 55 predictions asked. It tried
tilts from -0.6 to 2.8 rad in the prediction and found only 0 to 0.25 rad landing, which is
below the flow tilt, so the pour that landed (0.2 rad, 0.05 m) was refused for not flowing.
It never tried 2.9, and was told by the prediction that 2.0 to 2.8 are out of reach.

That was wrong. From the turned base the reach check calls every tilt from 2.0 rad up
unreachable, whatever the solver's iteration budget (200 to 3000), while the simulation
performs 2.0, 2.8 and 2.9 rad (and fails 2.2, 2.4, 2.6, 3.0 and 3.07):

| tilt (rad) | 1.5–1.96 | 2.0 | 2.2–2.6 | 2.8 | 2.9 | 3.0, 3.07 |
|---|---|---|---|---|---|---|
| simulation | finishes | finishes | fails | finishes | finishes | fails |
| reach check | reachable | out of reach | out of reach | out of reach | out of reach | out of reach |

The check solves with the base fixed, while the pour is a whole-body motion; so §7's "a pour
predicted unreachable is certain to fail" does not hold from the turned base (it held for the
nine-tilt grid its test uses, which has no tilt at 2.0, 2.8 or 2.9). Open: let the reach check
move the base, or stop it from refusing.
