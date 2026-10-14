# llm_reasoner_viz

A browser viewer for `llm_reasoner` at work, beside `cram_viz`. cram_viz shows the scene
and the knowledge; this shows the system doing its job: the resolver, the world injector,
the knowledge component and the model, the calls between them, and the prompts and
answers they exchange, as they happen and afterwards.

It is read-only. It shows what the processes working on a resolution run wrote to the
run's log, `events.jsonl` in the run's directory (by default `~/.cram_viz/runs/<run>/`):
the cram-viz server, the process trying a pour in the simulation, and the live runner
performing the resolved action all append to it. A run is followed live while its log
grows and replayed from the same log once it has ended. A run from before the logs has
none, and is not shown.

```bash
llm-reasoner-viz                       # http://127.0.0.1:8721
llm-reasoner-viz --port 8722 --runs ~/.cram_viz/runs
python -m llm_reasoner_viz.server      # without installing the command
```

cram-viz links to the viewer from the answer of every resolution (`LLM_REASONER_VIZ_URL`
sets the address it links to).

The viewer listens on this machine only (`--host` changes that): the events hold prompts
in full. It needs the Python standard library only, and its page is plain JavaScript
modules with no build step and no library loaded from elsewhere.

## The screen

- **Top bar**: the run (a picker of the runs that have a log), how it stands (running,
  finished, failed), the model calls, tokens and cost so far, and whether the viewer is
  following the log. *events* opens a drawer with the raw events.
- **System** (left): the resolver, simulation, world injector, language model and
  knowledge component, with the number of calls over each connection, what a component is
  doing now, and a flash on a connection that has just been used. Clicking a component
  singles out its row on the timeline.
- **Timeline** (middle): a row per component and a bar per piece of work, as wide as it
  lasted. Bars of a run still going are striped and grow; a bar outlined in dashes was
  rejected (an attempt that went back to the model, a pour that missed) and one outlined in
  red failed. *fit* shows the whole run, *−* and *+* zoom, *follow* keeps the newest work in
  view.
- **Inspector** (right): what a bar is. A model call shows who asked, which model, how it
  went, and its system and user prompt as the tagged sections they are made of (each opens
  and closes, and copies), the schema, the tools offered, the tool calls made during it, and
  the answer; a resolution its slots, the knowledge it consulted, its attempts and why it
  answered as it did; an attempt why it was rejected; a hypothesis what was asked and how it
  was served; a pour what was tried and where it landed. The trail above the title leads up
  through what the span happened during; the lists lead down.

The address holds the run and the span (`/?run=<run>&span=<span>`), so a link opens the page
at what it points to.

## The API

| Request | Answer |
|---|---|
| `GET /api/runs` | The runs that have a log, newest first, each summed up: its query, scene, how resolving and performing went, how many events and model calls, tokens and cost |
| `GET /api/runs/<run>/events` | The events of a run as one batch, with `next`, the offset to go on from; `?after=<offset>`, `?limit=<n>` |
| `GET /api/runs/<run>/stream` | The events as server-sent events, those written so far and then those written next; each frame's `id` is where its event ends in the log, so `?after=<offset>` or the `Last-Event-ID` header resumes a stream |

## Layout

```
src/llm_reasoner_viz/
    vocabulary.py   the names of the events and their fields the viewer reads
    paths.py        where the runs are kept
    runs.py         EventLog, RunSummary, RunLibrary
    stream.py       EventStream: following a log as it grows
    server.py       the HTTP server and the console entry point
    web/            the page: index.html, app.css and ES modules
        model.mjs       the spans of a run, built from its events one at a time
        prompts.mjs     a prompt read as its tagged sections
        layout.mjs      spans packed into rows of the timeline
        inspector.mjs   what the inspector shows of a span
        map.mjs         the components, their connections and what each is doing
        *_view.mjs      the elements: map, timeline, inspector, events
        app.mjs         follows a run and keeps the views in step
```

## Tests

```bash
pytest test/llm_reasoner_viz_test
```

The modules with logic have node tests (`test/llm_reasoner_viz_test/js`, run from pytest
and skipped when node is not installed) on events taken from real logs
(`js/fixtures`, shortened); the views that draw elements are checked in a browser.
