# cramera_resolution

An add-on for the `cramera` viewer. It serves the viewer unchanged and adds what `cramera`
has no place for: the resolution of underspecified actions by `llm_reasoner`, the knowledge
generated for them, the views of that knowledge, and the performance of what was resolved.

`cramera` is not edited. The add-on subclasses its request handler and server, answers its
own routes, serves its own panels under `/addon/`, and inserts their scripts into
`cramera`'s own `index.html` as it is served, so a newer `cramera` page is picked up as it
comes.

```bash
cramera-resolution [port] [--no-browser]
```

## Routes

All under `/api/resolution/`, so they cannot collide with a route `cramera` adds later.
Each answers JSON, and an error as `{"ok": false, "error": "<Type>: <message>"}`.

| Route | `cram_viz` route | What it does |
| --- | --- | --- |
| `GET runs` | (none) | The runs kept, newest first, each with its scene, designator, query, cost and pours tried |
| `GET designators?scene=` | `/api/designators` | The predefined designators the scene offers, and the knowledge to choose from |
| `POST resolve` | `/api/designator/resolve` | Resolves a designator query in a scene with `llm_reasoner` |
| `GET run?name=&scene=` | `/api/designator/run` | Shows a kept run as it was made |
| `POST hypothesis` | `/api/hypothesis/generate` | Generates hypothesis knowledge for an instruction |
| `POST execute` | `/api/execute` | Performs a kept run in the simulation under `cramera`'s live viewer |
| `GET view?name=&scene=` | (part of `/api/kb/view`) | The hypothesis, ontology or designator view |

Resolving needs an Anthropic API key in the environment. Nothing that needs one is built
until a route needs it.

## What is kept, and where

Under `CRAMERA_DATA` (default `~/.cramera`): `runs/`, `hypotheses/` and the NEEM database
`neems.sqlite`. `CRAMERA_HYPOTHESIS_STORE` and `CRAMERA_NEEM_DATABASE` move the last two.
Scenes are found as `cramera` finds them, through `CRAMERA_SCENES`.

Runs store the class paths of their parts, so a run made by `cram_viz` cannot be read here,
and one made here cannot be read by `cram_viz`.

## The panel

`web/addon/designator_panel.js` mounts an *Underspecified Action Designators* panel right
below `cramera`'s EQL panel, through its panel registry. It offers the scene's designators as
chips that copy their query into a box, the knowledge to consult (the phase structure, each
perspective, adjusting the pour geometry), Resolve, what the resolver chose, why, what it
rests on and the pours it tried, the runs kept (a tab; a click shows one again, as does
`?run=<name>` in the address, which the llm_reasoner viewer links to), and Execute. The box
suggests as the EQL box does, through `cramera`'s own suggestion menu: the action designators
by name and, inside one's argument list, the parameters it takes with their types and
defaults (`designators` route, `action_vocabulary`); like the EQL box, it offers once a letter
is typed. The panel publishes the hypothesis a resolution consulted as
`hypothesis:generated` on the bus.

`web/addon/knowledge_tabs.js` adds the hypothesis graph, the ontology graph and the designator
description as tabs of `cramera`'s graph panel, beside Knowledge, Plan, Statechart, Kinematics
and Transforms, once the page has booted its panels. A tab of ours takes the area of the
graph's canvas while it is open, and one of `cramera`'s gives it back. Each view is fetched
from `view` for the open scene. A hypothesis generated while no tab of ours is open marks the
Hypothesis Graph tab with a dot until it is opened; while one is open, it is redrawn.
Clicking a node emits `entity:select`, and `entity:highlight` reveals or spotlights nodes on
the hypothesis graph. The drawing modules (`hypothesis_graph.js`, `ontology_graph.js`,
`designator_view.js`) are copies of `cram_viz`'s.

The panels' logic is tested with node (`test_web_panels.py`), as `cramera`'s own panels are.

## What is not moved yet

- Plan steps and nodes are not written to a run's log as they change (`cram_viz` did, from
  its own bridge's snapshots); a run's start and end are.
- The knowledge is not yet queryable from `cramera`'s EQL panel.
- Performing a run with no run named (`cram_viz`'s hardcoded demo) is not moved.
- The NEEM of a performance is kept only where the generated ORM interface exists
  (`scripts/regenerate_all_orm.py`).

## Scene bundles

`cramera/scenes` holds the bundles of `cram_viz/scenes` in the format `cramera` reads, and
`cramera` finds it without `CRAMERA_SCENES`. Two things differ from `cram_viz`'s bundles:
a `placeTarget` holds `position` where `cram_viz` writes `pos`, and the PR2's `<transmission>`
elements, which describe the real robot's actuators and which `cramera`'s URDF reader
refuses, are left out. Everything else is copied as it is, and `cram_viz/scenes` is not changed.

```bash
python -m cramera_resolution.scene_conversion cram_viz/scenes cramera/scenes
```

Run it again after `cram_viz`'s bundles are regenerated.
