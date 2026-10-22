# CRAMERA

CRAMERA shows CRAM worlds and plan execution in a browser. Its three panels combine
a 3D scene, EQL questions and execution graphs. You can follow a running demo or
inspect a saved recording.

## Install in the CRAM workspace

Use Python 3.12 and follow the [workspace installation instructions](../README.md),
including the ROS setup needed by live demos. From the repository root:

```bash
uv sync --extra dev --active
cramera
```

The viewer is served at `http://localhost:8711`. All browser libraries are packaged
locally. Their versions, checksums and licenses are documented in
[the vendor directory](src/cramera/web/vendor/README.md).

## Follow a live demo

In another terminal using the same environment:

```bash
cramera-live path/to/demo.py
```

The wrapper selects CRAMERA through CRAM's visualization backend. The live bridge
serves the world's geometry and state on port 8765. Open the viewer and choose Live
to follow the robot, objects, plan progress and motion statecharts. The viewer
observes execution; it offers no robot or plan editing controls.

Use one live session per `CRAMERA_DATA` directory. Separate simultaneous sessions
need separate data directories and ports.

`cramera.config.CrameraConfig` holds the live session's robot-root publication key,
world-discovery interval and placeholder object dimensions. Pass it as
`Bridge(configuration=...)` when constructing a custom bridge for `LiveVisualization`.
Each bridge gets its own configuration by default.

The scene supports orbit, pan, zoom, robot following and click-to-inspect. The graph
panel displays the plan, statecharts, robot kinematics and transforms. EQL results
can highlight entities in the scene and replay a recorded time interval.

Attaching a world automatically enables queries against the native `world` object
in the EQL panel. All of its attributes and relationships are available, including
bodies, connections, degrees of freedom, regions and custom semantic annotations.
Collection presets come from the world's public list fields and properties; their
labels use native EQL verbalization. Queries reflect subsequent world changes, and
`world.` completion lists the available members. A demo can register existing
knowledge with `register_query_source` to supply its own scopes, evaluation and
queries. Pass lists of `QueryableKnowledge` and `Preset` objects directly;
`QueryableKnowledge.of_world(world)` exposes a native world. Query domains retain
their original iterable collections, so later records remain queryable. Registered
preset lists also retain later additions. Register again when replacing the source's
knowledge or preset lists.

The EQL editor executes trusted local Python statements against the loaded world.
The Python server listens only on loopback and rejects requests from remote
origins. Run queries you trust in the same local environment as the viewer.

The 3D panel requires WebGL. If the browser cannot create a WebGL context, open the
viewer address in a browser with WebGL enabled; EQL and graph inspection remain
available.

## Record and replay

Live capture starts with the visualization backend. Stop the recording in the
viewer, choose a name and save it. A demo shutdown also finalizes its capture, so
the viewer can save it after the demo has exited. The save dialog can trim the
recording to a selected frame range; discard removes an unsaved capture.

Playback controls provide play/pause, seeking and speed selection. Recordings carry
model descriptions, resolved mesh assets, joint and object trajectories, and the
captured inspection data. Their files can be served by an ordinary static HTTP
server for 3D playback; EQL and the knowledge panels use the Python server.

Use `?scene=<name>` to select a recording explicitly. The scene picker combines
local recordings with an optional shared bundle directory:

| Variable | Purpose | Default |
| --- | --- | --- |
| `CRAMERA_DATA` | Writable capture and cache directory | `~/.cramera` |
| `CRAMERA_SCENES` | Optional directory of existing scene bundles | Local recordings |
| `CRAMERA_ARCHITECTURE` | Repository inspected by the architecture graph | Current CRAM checkout |

Scene bundles and robot-model assets are data, and are not included in this Python
package. New recordings are written under `CRAMERA_DATA/scenes`.

## Resolve underspecified action designators

`cramera.resolution` resolves underspecified action designators with `llm_reasoner`, shows the
knowledge generated for them and performs what was resolved under the live viewer. It needs
ROS's `rclpy` (coraplex builds on it); without it the viewer is served as before and the
routes below say so. Resolving needs an Anthropic API key in the environment.

The panels are in `web/panels/resolution`:
- *Underspecified Action Designators* shares one frame with the EQL panel, as two tabs
  (`web/core/panel-tabs.js`, configured in `web/panel-groups.js`). It offers the scene's
  designators as chips that copy their query into a box (one the scene cannot resolve is
  dimmed, says what the scene lacks, and can still be copied; Resolve then refuses it), the
  knowledge to consult, Resolve, what the resolver chose, why, what it rests on and the pours
  it tried, the runs kept, and Execute. The box suggests the action designators and, inside
  one's argument list, their parameters, through the EQL box's own suggestion menu. A page
  opened with `?run=<name>` shows that run.
- The graph panel has three more tabs, *Hypothesis Graph*, *Ontology Graph* and *Designator*
  (`web/panels/graph/knowledge_views.js`), that draw the knowledge a resolution consulted. A
  freshly generated hypothesis redraws them, and marks the Hypothesis Graph tab until it is
  opened.

The routes, all under `/api/resolution/`, answer JSON, and an error as
`{"ok": false, "error": "<Type>: <message>"}`:

| Route | What it does |
| --- | --- |
| `GET runs` | The runs kept, newest first, each with its scene, designator, query, cost and pours tried |
| `GET designators?scene=` | The predefined designators the scene offers, the action designators and parameters to suggest, and the knowledge to choose from |
| `POST resolve` | Resolves a designator query in a scene |
| `GET run?name=&scene=` | Shows a kept run as it was made |
| `POST hypothesis` | Generates hypothesis knowledge for an instruction |
| `POST execute` | Performs a kept run in the simulation under the live viewer |
| `GET view?name=&scene=` | The hypothesis, ontology or designator view |

Under `CRAMERA_DATA` (default `~/.cramera`) it keeps `runs/`, `hypotheses/` and the NEEM
database `neems.sqlite`; `CRAMERA_HYPOTHESIS_STORE` and `CRAMERA_NEEM_DATABASE` move the last
two. The runs are the ones the `llm_reasoner_viz` viewer shows
(`LLM_REASONER_VIZ_RUNS=$CRAMERA_DATA/runs llm-reasoner-viz`), which the answers link to at
`LLM_REASONER_VIZ_URL` (default `http://localhost:8721`).

Not available yet: plan steps are not written to a run's log as they change (a run's start and
end are), the knowledge cannot be queried from the EQL panel, and the NEEM of a performance
is kept only where the generated ORM interface exists (`scripts/regenerate_all_orm.py`).

`cramera/scenes` holds the bundles of `cram_viz/scenes` in the format `cramera` reads (a
`placeTarget` stores `position`, and the PR2's `<transmission>` elements, which `cramera`'s URDF
reader refuses, are left out). Write it again after `cram_viz`'s bundles are regenerated:

```bash
python -m cramera.resolution.scene_conversion cram_viz/scenes cramera/scenes
```

## Pop out panels

The 3D scene and the graph panel, with all its tabs, each have a *Pop out* button that opens
the panel alone in a window of its own, for a second screen (`?layout=scene` and
`?layout=graph`; press F11 in the window for full screen). The windows are independent: a
node clicked in a popped-out graph is not described by the EQL panel of the other window.
The one thing the windows share is a generated hypothesis, which every window of the viewer
draws in its Hypothesis Graph tab as soon as it is generated.

## Package structure

The frontend mounts independent panels through `web/core/registry.js` and an event
bus. `web/config.js` chooses the layout. The Python `live` package observes existing
world and plan callbacks; `knowledge` answers queries and builds inspection views.
The visualization provider is registered through the `coraplex.visualizations`
entry-point group.

See [core scope](docs/core_scope.md) for the boundaries of this package.

## Tests

```bash
pytest test/cramera_test
```

The browser regressions run through pytest and require Node.js. Python coverage is
measured over the complete `cramera` package; the repository requires at least 85%
coverage for added code. Browser checks cover asset availability, panel contracts,
query rendering, playback helpers and graph behavior.

CRAMERA is licensed under [GPL-3.0-only](LICENSE). Bundled browser dependencies
retain their own licenses.
