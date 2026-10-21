# Phase 9 release checklist

Checkpoint E is **not yet a supported team release**. Local implementation and evaluation can be useful while the remaining evidence is collected. This checklist records independent gates rather than treating completion of a phase number as deployment authority.

## Implementation and local correctness

- [x] Reproducible comparison of source search, structural queries and curated context on the same pinned source/tasks.
- [x] Independent source checks for returned definition locations and citations; unknown model metrics remain unknown.
- [x] Controlled replacement/rebuild tests and pre-evaluation refusal of mismatched revisions.
- [x] Abrupt-publication, concurrent-reader, disk-capacity, lock-deadline and corrupt-cache recovery checks.
- [x] Real MCP transport checks for corrupt-cache refusal and stale-source recovery after explicit rebuild.
- [x] Shared resource-limited Windows worker and optional job commit-charge measurements.
- [x] Documented cache/evidence retention, worktree limits, rebuilds, schema changes, integration disablement and historical rollback.
- [x] Final package tests, formatting, wheel and real-source evaluation recorded in [the report](phase9-evaluation.md): 269 passed, one Windows-only skip; wheel built; six targets passed through all three retrieval methods.

## Team release and operational evidence

- [ ] Representative generation/debugging/impact tasks pass independently specified acceptance checks in all comparison arms; explain any correctness regression.
- [ ] Instrument actual model tokens, completed-task cost and amortized edited-source maintenance cost. Response size is not a replacement.
- [ ] Agree a team latency target and address any context-assembly cost that exceeds it. Preserve useful bounded navigation if curation does not earn its maintenance cost.
- [ ] Hosted CI validates the final implementation through an authorized workflow trigger. Local results are recorded separately.
- [ ] Deliberately validate and activate a successor checker/runtime/source/artifact tuple before publishing newer protected implementation code as canonical knowledge.
- [ ] Owner-managed retention preserves the canonical artifact and pinned runtime needed for rollback beyond their current expiry.

The configured publication mode remains automated and requires no other person's approval. None of these gates asks an external reviewer to approve ordinary local development. A trust-root update and team deployment remain distinct operations; phase 9 does not change the existing activation pins.

The [model playground](notebook.md) now provides the instrumented path for local Q&A and code-proposal trials. Mocked-provider tests establish integration behavior; they do not close the live generation, token-cost or task-success gates above.
