# Phase 0–3 validation

Measured locally on 19 October 2026 using Python 3.12.13 against main at `7b27424de7768798dc561f447e11c2a8eb86c307`. These are single-run development measurements, not production benchmarks. Both indexing and command trials ran under a 4 GiB Windows job limit.

## Source coverage

All 2,992 committed tree entries were recorded.

| Inventory category | Files |
| --- | ---: |
| Python | 2,281 |
| Markdown/RST | 205 |
| Configuration | 134 |
| Templates | 10 |
| Excluded | 27 |
| Unsupported formats | 335 |

The graph contains 48,321 declarations/sections and 410,781 relationships: 142,022 resolved and 268,759 unresolved. Resolution is lexical evidence, not runtime-call completeness. Inventory diagnostics include 5 symbolic links, 22 excluded local/vendor/build paths and 335 unsupported formats. Extraction also recorded 94 unresolved-source diagnostics. This main revision produced no syntax-failure diagnostic.

The SQLite cache occupies approximately 689 MiB (722,288,640 bytes). It includes committed source text and typed evidence payloads. Full indexing took 123.33 seconds. Cache size reduction and incremental rebuild costs need further measurement.

## Fixed navigation set

The portable task definitions are in [evaluation.toml](evaluation.toml). Each expected path and declaration line was independently checked with committed Git search. All six exact-symbol queries returned the expected source location.

| Symbol | Definition line | Git search, seconds | Selected query, seconds |
| --- | ---: | ---: | ---: |
| GiskardExecutable | 193 | 0.0456 | 0.0515 |
| ActionDescription | 36 | 0.0573 | 0.0334 |
| ExecutionEnvironment | 18 | 0.0839 | 0.0316 |
| MoveMotion | 24 | 0.0577 | 0.0337 |
| World | 544 | 0.0568 | 0.0379 |
| NamedEntity | 14 | 0.0524 | 0.0482 |

These query timings use an already imported Python process. The Git baseline searches a known file and class prefix; it does not perform the same operation as qualified-symbol lookup and can return additional names. Neither column establishes a general speedup.

A separate CLI trial of `GiskardExecutable.execute` returned lines 369–400, its signature and source documentation. Initially it took 89.72 seconds because every query decoded the entire graph. Selective evidence reads reduced the same CLI trial to 1.45 seconds, including process startup. This improvement is an implementation measurement, not an OKF/token-savings claim.

Main-branch relationship checks found 50 incoming relationships to GiskardExecutable (20 returned on the default page) and 19 outgoing relationships from its execute method. Returned records retain their static relationship kinds and resolution states.

## Automated validation

The focused pytest suite covers source identities, exact committed reads despite working-file edits, source-root mapping, generated-ORM exclusion before blob access, large files, syntax failure diagnostics, lexical nesting and aliases, type-only imports, Markdown/RST evidence, persistence, rollback after failed publication, allowed serialized types, bounded excerpts, pagination, stale-commit refusal, query snapshot consistency, import direction and cycles, and CLI workflows.

Selective relationship reads are compared against navigation over the complete fixture graph in both directions. A subprocess test confirms serialization works with EQL unavailable. KRROOD is unchanged.

The final package test result and formatting checks are recorded in [progress.md](progress.md). Hosted CI has been configured but has not been run from this local task.

## Evaluation still required

Phase 0–3 fixtures cover stale commits and working-file isolation; the phase 4 follow-up below adds overlays and mutation equivalence. Curated concepts and trust/publication checks belong to phases 5 and 7. Example selection, code-generation context quality, test recommendations and measured agent token use belong to phase 6. No token reduction, runtime coverage or production trust guarantee is claimed by this release.

## Phase 4 follow-up — 20 October 2026

The focused suite now passes 75 cases with 1 Windows skip for symbolic-link creation; the corresponding Linux CI case remains enabled. Hosted CI has not been executed from this task. Mutation tests compare incremental refresh with a clean rebuild and exercise publication rollback, source races and stale-query refusal.

The real working-tree trial used main as its baseline and explicitly included `repository_knowledge/src` and `repository_knowledge/pyproject.toml`. Other untracked source was outside this trial's selected scope. Both the command processes and their subprocesses were capped at 4 GiB.

| Operation | Measured result |
| --- | --- |
| Initial overlay index | 3,034 inventory entries; 136.11 seconds |
| Unchanged overlay refresh | 2,506 analyses reused; 0 parsed; no publication; 8.53 seconds |
| Query current IndexRefresh.refresh | Correct source lines 41–101; 9.12 seconds |
| Query current GiskardExecutable.execute | Correct source lines 369–400; 8.56 seconds |
| Rebuild separate committed cache in format 2 | Completed in 94.05 seconds |
| Committed freshness check | Matching revision and source; 1.56 seconds |

The format-2 committed cache occupies 739,266,560 bytes; the overlay occupies 745,672,704 bytes. Original per-file analysis is compressed at rest. Cache files are ignored and rebuildable.

These remain single-run development measurements. Working-tree queries re-read selected source to validate its fingerprint before and after navigation. An unchanged refresh avoids analysis decoding and database writes, but still checks source contents. Changed-file parsing is incremental; reference resolution and completed-graph publication remain conservative. Changed-refresh latency on the full CRAM graph has not been benchmarked. Concept invalidation awaits phase 5.

## Phase 5 follow-up — 20 October 2026

The completed package suite passed 132 tests with one Windows-only symlink skip in 98.59 seconds under its 4 GiB job cap. The pilot contains eight concepts tied to the same main baseline. Main validation returned all eight as structurally current, with no format or CRAM policy findings and no authenticated reviews.

The committed wiki check took 9.95 seconds in a development run. Lexical discovery returned the two execution concepts in 9.80 seconds. Export took 1.60 seconds and its output passed the independent OKF CLI v0.4.0 lint with zero errors/warnings. A cross-directory role-concept move took 10.05 seconds and its nine-concept proposal passed validation in 11.43 seconds.

Two completed changed-overlay refresh trials took 191.39 and 294.25 seconds under the 4 GiB cap. Some trials overlapped the package tests, so these are operating observations rather than isolated performance comparisons. Per-file parsing was reused, but the complete reference graph was still resolved and published. A source edit during a later trial triggered the expected snapshot-mismatch refusal; the cache could not be presented as current afterwards.

The Windows overlay initially marked the pilot stale because checkout CRLF bytes differ from Git LF bytes. The corrected policy explicitly proves LF/CRLF equivalence using both digests, reports affected paths, preserves exact-byte comparison as an option and keeps source-index freshness byte-exact. The focused mutation cases cover opposite line endings, unknown byte changes, strict comparison and BOM changes. These checks establish source consistency, not semantic verification of prose or authenticated historical provenance.

The final overlay refresh completed in 293.77 seconds: one file analyzed, 2,511 analyses reused, zero removals, and 3,040 inventory entries. Its completed snapshot is `a157fbbd20d9f4dc76b1155804709bb7a37dcbd15f682b7de93c7d9445b726ef`. Live working-tree wiki validation took 23.64 seconds and returned all eight concepts current with zero findings, while reporting 16 distinct files matched through line-ending equivalence. A final bounded execution query took 10.37 seconds and returned `packages/coraplex.md` and `workflows/execution-environments.md`, with no further matches omitted. All review-authentication values remained false.

Native-CLI interoperability, source-mutation correctness and local navigation are measured here. Agent task success, model tokens and amortized savings remain phase 9 evaluation work; no token-saving percentage is inferred from smaller wiki documents.

## Phase 6 follow-up — 20 October 2026

The fixed task pilot used the same committed-main baseline and one coherent task projection. All source citations were checked for matching snapshot, captured file digest and valid line extent. Each context contained the selected definition, interface, applicable instructions, examples, relevant tests and fixture declarations. The world task also included its spatial style guide. Generation contexts included current draft concepts where their evidence overlapped the selected source; all lifecycle labels remained producer declarations.

| Seed | Payload characters | Context assembly seconds | Selected tests | Fixture declarations |
| --- | ---: | ---: | ---: | ---: |
| MoveMotion | 31,952 | 19.05 | 5 | 7 |
| GiskardExecutable | 35,293 | 18.23 | 5 | 9 |
| ActionDescription | 41,312 | 27.16 | 5 | 13 |
| ExecutionEnvironment | 37,291 | 16.11 | 5 | 10 |
| World | 43,144 | 37.47 | 5 | 7 |

Each allowance was 60,000 payload characters. The first two saved tasks each combine their two explicit seed contexts from the same snapshot. The one-time resolved-graph read and freshness check took **58.02 seconds**, excluded from the assembly timings above. Static pytest discovery took **3.58 seconds**, recognizing **8,323 tests and 485 fixtures**. CLI requests repeat their projection/discovery work; they do not yet share this in-memory pilot session. These are single development observations, partly overlapping bounded package tests and refresh work, not controlled performance comparisons.

The same-main World impact trial compared **115 public contracts**, observed **1,769 resolved references** and **11,740 uncertain name candidates** per snapshot, and took **64.41 seconds** after the shared graph/catalog setup. These are reference occurrences, not distinct runtime consumers. Candidate counts include weak name coincidences. This exposed an unacceptable output size for agents; a failing consumer-page test preceded the addition of bounded resolved/uncertain pages with complete counts and next offsets. The complete static graph is still loaded internally.

Main task artifacts are development outputs outside the repository. The reproducible command contracts and evidence requirements are recorded in [phase6.md](phase6.md), while deterministic local fixtures cover mutation and output bounds in CI. No model was asked to generate code from these contexts, no robotics tests were executed by the assistance service, and no token-saving or generated-code correctness result is inferred.

Final package validation passed **159 tests**, with one Windows-only symlink skip, in **111.54 seconds** under the 4 GiB cap. Black checked all **91** valid Python files, unused-import checks passed, and the wheel built at **82,425 bytes** with SHA-256 `6d2fd768cf178fba8e0eac1445f6a085d1f9b9a1e07ef6ee9d916992785a0c45`. Hosted CI was not executed.

The final overlay refresh took **148.16 seconds**, parsed five changed files, reused **2,512** analyses, removed none and published **3,045 inventory entries**. The resulting snapshot is `1d194c4b36b3b90641fd5170737e256e8de218a9286700011eece1c0efd9026a`. A subsequent working-tree freshness check passed in **4.67 seconds**. The live overlay explicitly includes untracked tool source and its package manifest; other untracked tests/documents remain outside that selected source view. New test behavior is verified in disposable local repositories by the package suite.
