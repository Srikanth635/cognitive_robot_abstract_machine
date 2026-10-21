# cram_viz — CRAM Vizulation

Browser-based visualization for the CRAM architecture — one tool for two modes:

- **Recorded**: run any coraplex demo once through the onboarder and get a
  lightweight, self-contained 3D scene (URDFs + meshes + the real recorded
  giskardpy trajectory) that plays in any browser, no ROS required.
- **Live** (RViz replacement): attach the viewer to a *running* demo — it
  renders the executing world in real time, and dragging objects in the viewer
  writes their pose back into the demo's world.

## Quick start

```bash
cram-viz                                  # serves http://localhost:8711
cram-viz-onboard path/to/demo.py --name my_scene    # record a demo once
cram-viz-live path/to/demo.py             # run a demo with the live bridge
```

Scene bundles are **generated artifacts** (tens of MB per scene) and are not
part of this repository — ready-made demo recordings live in
[sunava/cram-scenes](https://github.com/sunava/cram-scenes), wired as an
**optional** submodule:

```bash
git submodule update --init cram_viz/scenes    # ready-made demo scenes (optional)
```

The viewer looks for bundles in this order: `CRAM_VIZ_SCENES=/path` (env
override) → the initialized submodule `cram_viz/scenes` → `~/.cram_viz/scenes`
(where the onboarder writes by default). Live visualization and freshly
onboarded scenes need none of the ready-made bundles. Select a scene with
`?scene=<name>` or `CRAM_VIZ_SCENE=<name>`.

## Panels — how the UI is composed

The frontend (`src/cram_viz/web/`) is a set of **panels** mounted into layout
slots. Which panel appears where is decided by **one file**:

```js
// web/config.js
window.CRAM_VIZ_CONFIG = {
  layout: {
    left:  ['robot-scene'],
    right: ['eql', 'graph'],
  },
};
```

Removing a visualization = deleting its id here. Adding your own:

1. create `web/panels/<name>/panel.js`:

   ```js
   Panels.define('my-panel', function (root, bus) {
     root.innerHTML = '<div class="panel-head"><h2>My panel</h2></div>…';
     bus.on('entity:highlight', function (p) { /* react */ });
     bus.emit('entity:select', { id: 'x', detail: {…}, relations: [] });
   });
   ```

2. include the script in `web/index.html`,
3. add the id to `config.js`.

Panels **never call each other directly** — they publish/subscribe on the
event bus (`web/core/bus.js`), so any subset of panels works. The contract
between the built-in panels:

| event                | payload                          | emitted by → consumed by       |
| -------------------- | -------------------------------- | ------------------------------ |
| `scene:part-clicked` | `{id}`                           | robot-scene → eql              |
| `scene:step`         | `{step}` (`'__done__'` at end)   | robot-scene → eql, graph       |
| `live:changed`       | `{on, url}`                      | robot-scene → graph            |
| `entity:highlight`   | `{ids, focus?}`                  | eql → robot-scene, graph       |
| `entity:select`      | `{id, detail, relations}`        | graph → eql                    |
| `kb:ready`           | `{payload}`                      | eql → anyone                   |

### Built-in panels

| panel         | shows                                                        |
| ------------- | ------------------------------------------------------------ |
| `robot-scene` | three.js scene: environment, robot, draggable objects, playback + live controls |
| `eql`         | EQL query console, predefined designators + answer panel     |
| `graph`       | tabs: Knowledge / Hypothesis Graph / Ontology Graph / Designator / Kinematics / Plan / Statechart |

Below the EQL query box are predefined, underspecified pouring designators
(`cram_viz.resolution`), each shown as a krrood match query written in a file of its
own under `resolution/designator_queries/`: it states some slots of coraplex's
`PouringAction` and leaves the others open (`...`, or a variable such as
`variable(Tool)`). A query reads krrood's factories, `PouringAction`, `Arm`, `Tool` and
`Body`, and the scene's objects, each by its kind in snake case (`pouring_cup`, `bowl`).
Clicking one copies its query into the search bar, where it may be edited; **Generate**
resolves the query in the search bar with `llm_reasoner`'s resolver in the selected
scene: the server builds the scene's semantic_digital_twin world from its bundle (the
robot on a drive over its surroundings where the recording starts, objects typed by
`resolution/object_annotations.json`, and the robot typed by
`resolution/robot_annotations.json` when a class is kept for it), evaluates the query
on it, the resolver reads the world with the world tools, among them the predicates of
the world model (contact and support between bodies, and, with the robot annotated, who
holds what and what a camera sees) and consults the knowledge component, and the answer
panel shows the slots it chose, why, and what they rest on. A designator is greyed
out, with the reason, when the scene has no object of a name its query reads or no
candidate for a slot it leaves to choose. A new designator is a new query file and a
member of `PouringDesignator`.

Every resolution is a run of its own from the same initial environment: the scene's
world is built anew, and the knowledge is consulted over a copy of the store in
`~/.cram_viz/runs/<run>/hypotheses`, so what one run generates no other run reads; the
run keeps the environment it started from (`environment.json`), its resolution
(`resolution.json`) and a log of what it did (`events.jsonl`: every question to the
model with its prompts and answer, the world and knowledge calls, the attempts and why
one was rejected, the knowledge generated, each pour tried in the simulation) beside it,
and the answer panel names the run and links to the llm_reasoner viewer of it (served at
`http://localhost:8721` unless `LLM_REASONER_VIZ_URL` says otherwise). The link goes both ways: a
page opened with `?scene=<scene>&run=<run>`, as the viewer's *open in cram-viz* link does,
shows what that run resolved (`GET /api/designator/run?name=<run>`, the answer a resolution
gave) without resolving again, with its query in the search bar, the hypothesis it consulted
in the graph tabs, and Execute performing that run. It builds the resolver as resolving does,
so it needs the same credentials. The consulted
hypothesis is what the Hypothesis Graph tab shows; the Ontology Graph tab shows the same
graph with the ontology class of every node and the ontology property of every link
beneath their labels; the Designator tab shows the resolved designator. Resolving needs
an Anthropic API key (read from `.env` as for generation) and costs about ten cents a
click, plus a few minutes and tens of cents for the knowledge each run generates.

Before a pouring resolution is accepted, it is tried in the simulation
(`resolution/trial.py`): the pour is performed from the scene's initial world in a
process of its own and measured by where the pitcher's mouth ends against the bowl's
opening, as seen from the robot. A pour that misses goes back to the resolver with that
outcome as structured data, and the resolver answers again, up to three trials, the last
accepted whatever it brings about; the answer panel lists every pour tried.

**Execute** performs the last resolution in the simulation (`resolution/execution.py`):
in the world of the run's scene, built anew, with its robot annotated as the class
`resolution/robot_annotations.json` keeps for it, the robot parks its arms, lowers its
torso, faces the source container, picks it up with the resolved arm and pours as
resolved, under the live bridge — the 3D scene's Live button shows it running, and the
Plan tab shows the plan of the run (the bridge follows a plan performed through its root
node as well as through `Plan.perform`, and publishes coraplex's `NOT_STARTED` and
`PAUSED` as the `CREATED` and `PAUSE` its legend names). Execute also appends to the
run's `events.jsonl`: that it started and ended or failed, each step of the plan, the
actions below its root, when its status changes (`StepChanged`), and every node of the plan
(`PlanNodeChanged`: the node, the node above it, its kind, what it performs, its status and
whether the bridge derived that status), when it first appears in a snapshot of the plan and
whenever its status changes, so the log holds the tree of the plan and, with the time of each
event, when each motion began and ended. A real Execute of 12 s wrote 65 of them, 15 KB. An
action stays `RUNNING` in the bridge until the plan ends, since the conditions it would turn
`SUCCEEDED` on are not evaluated in these runs, so a step is over when the next starts; its
motions do end, with their own status. **Demo** launches the
fixed demo instead, independent of the designators.

### NEEMs of the simulations

Every plan the simulation performs is kept as a NEEM, the executed plan saved through
coraplex's ORM: each pour tried in the simulation while a resolution is made, and the
run Execute performs, whether the motion finished or failed. They all go into **one
database**, `~/.cram_viz/neems.sqlite` (`CRAM_VIZ_NEEM_DATABASE` names another), and the
run's `events.jsonl` says which plan is which (`NeemRecorded`: whether it is of a trial or of
the execution, the trial or the run, the database, the identifier it was given and how long
saving it took; `NeemNotRecorded` says why one could not be kept, and never hides how the
performance went). The llm_reasoner viewer shows them (see its README).

What it costs: the database's tables, over a thousand, are made when the first NEEM is kept,
which takes about 40 seconds once (a lock keeps two processes from making them together);
after that each NEEM takes 5 to 9 seconds to save, in the process of the trial or of
Execute, and about a megabyte of disk. A resolution tries up to three pours, so it takes
around twenty seconds longer than before. There is no switch to turn the recording off yet.
To read a NEEM, the plan of the NEEM with identifier `N` is the row of the table
`PlanMappingDAO` with `database_id = N`:

```bash
sqlite3 ~/.cram_viz/neems.sqlite "select * from PlanMappingDAO where database_id = 1;"
```

On the Plan and Statechart tabs the node border is its execution status —
running (amber), succeeded/done (green), failed (red), paused (blue),
interrupted (orange), not started (dim, dashed) — streamed live from the
bridge while attached.

Two things to know about those statuses: coraplex performs only the plan
**root** (`Plan.perform` → `root.perform`), while `ActionNode.notify` merely
expands its children into one merged motion statechart. So a *recorded* plan
tree has real status on the root only. Live, the bridge derives per-step
status from the statechart life cycle via `GiskardExecutable.motion_mappings`
(`{MotionNode: Task}`) and propagates it up the tree; those nodes are flagged
`derived`. Statecharts exist only during execution, so the Statechart tab is
live-only.

## Layout

```
src/cram_viz/
  server.py        static frontend + JSON API (/api/kb, /api/eql, /scenes/)
  kb.py            scene-driven EQL knowledge base + graph payloads
  paths.py         all filesystem locations (env-overridable)
  live/
    bridge.py      bridge state + serializers (runs on the sim thread)
    hooks.py       Executor/Plan/GiskardExecutable/mesh hooks
    http.py        the bridge's HTTP endpoints (port 8765)
    __main__.py    cram-viz-live entry point
  onboard/
    demo.py        demo -> scene bundle (record + bundle, one command)
    bundle_urdf.py standalone URDF/xacro asset bundler
  web/
    index.html     shell: topbar + slots + script includes
    config.js      WHICH panels are shown WHERE  ← edit this to swap panels
    core/          bus, panel registry, split/resize helper
    panels/        robot_scene/, eql/, graph/
    vendor/        three.js, vis-network, … (all local, no CDN)
```

## Tests

```bash
pytest test/cram_viz_test
```

The JS core (bus, registry, graph status rendering) is covered by node-based
tests invoked from pytest (skipped when node is unavailable).
