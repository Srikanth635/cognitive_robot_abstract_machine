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
in full. It needs the Python standard library only, and its page is plain JavaScript with
no build step and no library loaded from elsewhere.

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
    web/            the page
```
