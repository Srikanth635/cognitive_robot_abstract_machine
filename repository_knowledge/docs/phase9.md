# Phase 9: release validation and operating cost

This phase measures the delivered system and makes failure recovery repeatable. It does not add another graph service, a model router or automatic wiki rewriting. The implementation remains under `repository_knowledge/`; client adapters stay in their owning packages.

## Work and gates

1. Exercise abrupt process termination, concurrent readers, SQLite capacity exhaustion and corrupt-cache recovery in the isolated package tests. Retain the previous complete snapshot on failed publication. A corrupt cache must never report current source.
2. Compare committed Git source search, the structural CLI and the context CLI on the six independent locations in `evaluation.toml`. Pin all three to the same full source revision. Repeat the comparison on controlled fixture replacements; refuse mismatched revisions before scoring.
3. Preserve response-level accuracy, required-path recall, citation verification, returned uncertainty, wall time, response bytes and agent-visible calls. Record the memory cap and measured peak job commit charge separately. Leave model tokens and generated-task success unknown until a separate instrumented generation experiment supplies them.
4. Establish conservative local limits from these measurements. Document cache and evidence retention, full rebuilds, schema changes, optional-client disablement and historical rollback in [operations.md](operations.md).
5. Run the complete isolated package suite sequentially under the resource cap. Keep new tests in the existing package CI job. Record the results and remaining release gates in [phase9-evaluation.md](phase9-evaluation.md).

Checkpoint E remains open until independent model-task results, agreed team latency requirements and hosted validation support a team release. Passing source-navigation tests alone does not establish code-generation improvement or token savings.

## Reproducible retrieval comparison

The evaluation entry point invokes the actual public CLI in a fresh interpreter for each structural/context request. Git search scans Python declaration lines at the selected commit, excluding generated ORM interfaces; it is not given the known answer's file path. The current baseline tasks select classes, so this search is deliberately a declaration lookup rather than a general Python resolver. It can return multiple same-name declarations.

The evaluator first checks the committed-cache revision, then independently captures the committed blobs for scoring. This untimed setup and the additional citation-audit time are separate from per-request latency. Structural excerpts must match source text and snapshot identity. Context citations must match the source commit, snapshot, file digest and valid line range. Search hits must match their exact committed line text. Excluded or unsupported material is not silently accepted as source evidence.

`definition_found` means an exact expected path and declaration line appeared. Required-path recall is the intersection of returned cited paths and independently configured `required_paths`, plus the definition file. The saved six navigation cases require only their definition files; their recall is not a score for complete task context. Additional evidence requirements must be selected before a new experiment. `curated_items` counts local source-validated concepts, not independently reviewed claims. `uncertain_items` preserves the retrieved context's resolution labels.

All observations cost one agent-visible retrieval call. Git subprocesses inside a CLI, evaluator verification calls and interpreter startup are not counted as separate agent tool calls; startup is included in measured latency. A context request does more work and returns more evidence than a declaration search. This experiment measures retrieval, not the cost or correctness of completing a coding task with subsequent source reads and edits.

Run on Windows with the configured interpreter and the shared memory-limited runner:

```powershell
./repository_knowledge/scripts/run_command.ps1 `
  -Module repository_knowledge.interfaces.evaluation `
  -Arguments @('--repository', '.',
    '--database', 'repository_knowledge/.cache/index.sqlite3',
    '--tasks', 'repository_knowledge/docs/evaluation.toml',
    '--bundle', 'repository_knowledge/wiki',
    '--output', 'repository_knowledge/.cache/new-retrieval.csv') `
  -StatisticsPath repository_knowledge/.cache/new-retrieval.resources.csv
```

The task manifest selects the source revision. The cache must have been explicitly built for it. The bundle selection is local draft evidence; this evaluation does not contact GitHub or change publication trust pins. Use a new output name for each run: existing measurements are not overwritten. CSV rows are flushed after each trial so an interrupted run remains visibly incomplete. A response error, invalid citation, missing required path or unexpected declaration result makes the process fail; review partial rows before accepting a run. Same-name hits for an expected absent qualified symbol also fail. A timeout aborts visibly.

The resource CSV records Windows job peak committed bytes, the largest member process commit charge and the configured job limit. Commit charge is not resident memory. The job includes the evaluator and its CLI/Git children, including children that have exited. No concurrent pytest, refresh or benchmark is allowed during a controlled measurement. Source capture warms the filesystem; this is not a cold-disk benchmark.

On Linux, use the same module with the configured interpreter under a process/container memory limit. The existing CI job already applies an address-space cap to the package tests. Do not infer Windows job measurements from a Linux address-space limit.

## Model-task experiment still required

Use the three saved context tasks and a separately selected set of unfamiliar navigation, debugging, generation and impact tasks. Freeze prompts, required sources, acceptance tests, model/version, tool permissions, source revision and wiki revision before running any arm. Execute each arm in an isolated checkout/session, randomize their order, and repeat enough times to expose variation. The direct-source arm may make the source reads it needs; do not handicap it to the evaluator's single search request.

Capture model input/output/cache tokens, actual tool calls, end-to-end latency, successful checks, omitted required sources, invalid citations and stale answers. Repeat after a signature change, rename/deletion and an unresolved dynamic-dispatch case. A name candidate must never be scored as a proven runtime caller. Keep the same independent acceptance checks across arms. Include update cost and failed/retried tasks in the total. No live credentials or model call is introduced into normal CI.

Compute amortized update cost only for an observed workload: total rebuild/refresh time and model spend divided by completed tasks using those snapshots. An unchanged refresh does not measure an edited-service refresh. Payload characters or bytes cannot substitute for model tokens, and a six-symbol lookup sample cannot establish production task-success rates.
