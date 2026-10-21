# cramera_resolution

An add-on for the `cramera` viewer. It serves the viewer unchanged and adds what
`cramera` has no place for: the resolution of underspecified actions by `llm_reasoner`, the
hypothesis and ontology views, and the execution of what was resolved.

`cramera` is not edited. The add-on subclasses its request handler and server, answers its
own routes, serves its own panels under `/addon/`, and inserts their scripts into
`cramera`'s own `index.html` as it is served, so a newer `cramera` page is picked up as it
comes.

```bash
cramera-resolution [port] [--no-browser]
```

This is the first step: a test panel proving the mechanism. See
`llm_reasoner/doc/status.md` for where the work stands.
