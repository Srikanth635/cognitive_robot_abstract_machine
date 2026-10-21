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

## What is not moved yet

- The panels: designators, hypothesis graph and ontology graph. The routes above serve
  their data; the page has only a test panel.
- Plan steps and nodes are not written to a run's log as they change (`cram_viz` did, from
  its own bridge's snapshots); a run's start and end are.
- The knowledge is not yet queryable from `cramera`'s EQL panel.
- Performing a run with no run named (`cram_viz`'s hardcoded demo) is not moved.
- The NEEM of a performance is kept only where the generated ORM interface exists
  (`scripts/regenerate_all_orm.py`).
