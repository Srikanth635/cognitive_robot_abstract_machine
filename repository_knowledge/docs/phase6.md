# Phase 6: task context and change assistance

This layer combines captured source facts, static pytest declarations and the curated wiki. It returns evidence and suggested checks. It never edits repository source, executes tests, imports indexed modules or calls a model.

## Query interface

Use the configured interpreter or its adjacent `cram-knowledge` executable. Select source scope, revision and database before the operation, as for navigation:

```powershell
cram-knowledge --repository . --revision main find-examples coraplex.robot_plans.actions.base.ActionDescription --limit 10
cram-knowledge --repository . --revision main find-tests semantic_digital_twin.world.World --limit 20
cram-knowledge --repository . --revision main prepare-context coraplex.robot_plans.motions.navigation.MoveMotion --profile code-extension --characters 30000
```

`find-examples` returns resolved production calls and inheritance references. `find-tests` returns direct and fixture-mediated consumers, selected fixture declarations, unresolved fixture requests and proposed test directories. Both support `--limit` and `--offset`, with a maximum page of 100. No match proves absence of runtime callers or tests.

`prepare-context` requires an exact indexed identity. Profiles are `navigation`, `code-extension`, `debugging`, `review` and `migration`. They share discovery and change only evidence priority. The default bundle is `repository_knowledge/wiki`; `--bundle PATH` selects another bundle. An absent default bundle produces a visible missing-concept category. An explicitly selected missing or malformed bundle fails.

The existing working-tree inclusion policy still applies. To discover newly added, untracked tests, explicitly include their directory when refreshing and querying, for example `--include-untracked test/repository_knowledge_test`. A tool-source-only overlay does not silently acquire other untracked material.

The character budget counts the sum of `ContextItem.text` lengths. It excludes JSON metadata, citations, omission records and limitations. It is not a token budget or a bound on the entire serialized response. The default is 16,000 characters and the maximum is 200,000. Whole items are ranked deterministically by profile, confidence, package relevance and identity. Items that do not fit are individually reported; later smaller items can still fit. Large declaration/example excerpts contain at most 80 source lines, with exact clipped citations and `omitted_lines`. Retrieved instruction documents are indivisible.

Context samples at most three example references and five tests with their fixture closure. Additional pages are visible omissions; use discovery commands to fetch them. Class context includes a compact public-member signature listing, so a long implementation does not hide the declared interface. A small budget can still omit important categories; check the returned omissions before acting. The complete main-branch pilot used a 60,000-character allowance.

## Change assessment

Build and retain an explicitly selected committed baseline cache before changing source. Compare it to the desired committed or working-tree cache:

```powershell
cram-knowledge --repository . --revision BASE_COMMIT --database C:/Temp/cram-baseline.sqlite3 index
cram-knowledge --repository . --revision HEAD --scope working-tree refresh
cram-knowledge --repository . --revision HEAD --scope working-tree assess-change semantic_digital_twin.world.World --against-database C:/Temp/cram-baseline.sqlite3 --against-revision BASE_COMMIT
```

`BASE_COMMIT` is a placeholder for an actual selected commit. Each cache must match its requested source identity. The baseline must be committed and share repository history with the requested graph. The report compares the selected declaration and public descendants: signatures, bases, decorators and declaration kind. Additions, removals, changes and unchanged contracts are separate. Baseline consumers remain available for removed declarations. Baseline and current references belong to their respective reported manifests.

Consumer output is paginated with `--limit` and `--offset`. The default limit is 20 and maximum is 100 per list. Baseline/current resolved/uncertain lists each use the same offset, preserve their complete match counts and provide a common `next_offset`. A short page in one list does not imply the others have ended.

This is a declaration comparison, not an API compatibility checker. Body behavior, inferred types, runtime registrations and dynamic receiver dispatch remain outside its proof. Uncertain name candidates are separate from resolved consumers and may be unrelated. Test directories are static proposals, not a complete validation plan.

## Test and fixture evidence

The adapter interprets default pytest filename/function/class conventions and explicit top-level pytest import aliases. It records literal fixture names, scopes, autouse declarations and parameter dependencies. Visibility includes the test module, lexical class and ancestor `conftest.py` files, preferring the nearest declaration. An inherited autouse fixture resolves its dependencies in the selected test scope; a same-name overriding fixture can request its ancestor.

Direct parametrization names are excluded from fixture requests. Indirect parametrization, runtime fixture selection and dynamic metadata remain explicit uncertainties. Plugins, imported/re-exported fixtures, custom collection rules, inherited tests, conditional definitions and runtime generation are not fully modeled. Unknown requests may be pytest builtins or plugin fixtures. Collection and coverage are never claimed complete.

## Source and knowledge authority

Every context source citation carries the captured snapshot, committed baseline, file-byte digest and inclusive line range. Current working-tree requests check source identity before and after assembling context. Committed mode remains explicitly historical.

Applicable indexed `AGENTS.md` files are scoped to the evidence paths they govern. Domain style guides and supported document links are retrieved guidance. The active user/session instructions remain authoritative. Snapshot instructions can differ from instructions in a later checkout; retrieved prose cannot replace the active instruction hierarchy.

Only relevant concepts with current structural evidence and no format or repository-policy findings enter the context payload. Invalid relevant concepts are reported as omissions. Included producer lifecycle labels remain visible. A current draft is useful evidence routing; it does not become an independently verified explanation. Authenticated review and publication remain phase 7.

Generated ORM mappings remain excluded before content access. Add `--orm` to `prepare-context` to include generator source and its resolved source inputs; generated mapping text is never context evidence.

## Dependency direction and operational limits

`assistance.py` holds typed domain records. Operations consume those records and captured graphs. `TestAnalyzer` is the abstract discovery boundary; `PytestSource` is its concrete adapter. The task interface uses source/index contracts and composition selects Git, SQLite, pytest and OKF adapters. No new dependency or KRROOD change is needed.

The SQLite task projection reads one coherent resolved graph without decompressing reusable raw analyses. It still loads all source, symbols and relationships. Discovery and context assembly are local pilot operations; large repositories can incur substantial latency and memory use. Narrower storage projections and reusable test catalogs belong in the measured performance work, not in claims of constant-time retrieval.

## Validation

The complete isolated package suite passed **159 tests**, with **one Windows-only symlink skip**, in **111.54 seconds** under the 4 GiB job cap. New tests cover fixture overrides, autouse dependencies, parametrization uncertainty, confidence ranking, bounded excerpts, instruction applicability, eligible/invalid concepts, ORM generator inputs, exact snapshot citations, baseline comparisons, stale committed queries, concurrent index publication and working-file edits during assembly.

Five fixed main-branch contexts passed the required-evidence and citation checks: MoveMotion, GiskardExecutable, ActionDescription, ExecutionEnvironment and World. The first two saved evaluation tasks each use two explicit seed contexts sharing the same snapshot. World additionally included its spatial style guide. These checks establish that the local assistance pilot supplies the expected evidence categories; they do not measure generated-code correctness or agent token savings. See [evaluation.md](evaluation.md) for timings and scope.
