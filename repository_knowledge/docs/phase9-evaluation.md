# Phase 9 local evaluation — 21 October 2026

The six saved CRAM navigation cases passed through source search, structural CLI queries and curated-context CLI queries. Every returned citation passed its independent source check. Rich context is substantially slower than a declaration lookup and remains an explicit operation. This establishes local retrieval correctness for this sample; it does not establish generated-code quality or token savings.

## Inputs and method

The retrieval source is the original main baseline `7b27424de7768798dc561f447e11c2a8eb86c307`, using its existing format-2 committed cache. The common CLI implementation is phase-8 commit `e96d36f662dab307cbfe5d9194b592b81c28e417`, plus the phase-9 development evaluator. The unchanged wiki tree is `2d4f8742c77352f25f9d5729f8a3b970707b9a53`. The configured runtime is Python 3.12.13 on Windows. The [input hashes](evidence/phase9/phase9-input-hashes.csv) identify the measured development scripts and task manifest before final formatting and additional negative-case gates. [Final input hashes](evidence/phase9/phase9-final-input-hashes.csv) identify the delivered evaluator and guarded runner.

Each method used the same exact source revision and independently specified definition path/line from [evaluation.toml](evaluation.toml). Source search scanned committed Python declaration lines without being given the answer file. Every structural/context query used a fresh CLI interpreter. Context had a 60,000-character payload allowance and the local draft bundle. The evaluator captured source independently before timing requests and audited responses afterward. Both costs are outside the per-request latency; audit time has its own CSV column.

This is one serial trial per method and target, with a warm filesystem on a development machine. No pytest, extraction or other benchmark overlapped the comparison. It is not a cold-start distribution or a statistically established service-level target. The existing baseline cache's modification time and size remained unchanged.

## Retrieval results

| Target | Source search seconds | Structural CLI seconds | Curated context seconds | Context response bytes |
| --- | ---: | ---: | ---: | ---: |
| GiskardExecutable | 0.083 | 1.680 | 74.882 | 76,110 |
| ActionDescription | 0.082 | 1.489 | 83.090 | 89,859 |
| ExecutionEnvironment | 0.079 | 1.737 | 75.886 | 78,619 |
| MoveMotion | 0.080 | 1.611 | 72.205 | 69,120 |
| World | 0.076 | 1.518 | 77.687 | 81,283 |
| NamedEntity | 0.084 | 1.516 | 92.941 | 64,354 |

Median latency is **0.081 seconds** for source search, **1.564 seconds** for the structural CLI and **76.786 seconds** for rich context. All 18 requests located the expected definition. Definition-file recall was 1/1 in every case. These cases require only the definition file, so this is not a measure of complete source recall for a coding task.

Source search returned eight cited hits, including three same-name World declarations. The structural CLI returned six excerpts; context returned 143 citations. All **157/157** passed their source checks. Counts include repeated citations. Context retained source-resolution labels, included draft concepts for four targets and included none for World or NamedEntity. A valid source citation does not establish that every sentence of curated prose is correct or independently reviewed.

Every row records one agent-visible request. CLI-internal Git calls and evaluator audits are excluded from that count. The source arm returns a declaration line, the structural arm returns a bounded declaration excerpt, and context supplies interfaces, instructions, examples, tests and fixtures. They perform different amounts of work; the latency comparison is not a claim about completed-task efficiency. Token usage and generated-task success remain blank rather than estimated from bytes.

The complete [retrieval observations](evidence/phase9/phase9-retrieval.csv) and [resource receipt](evidence/phase9/phase9-retrieval.resources.csv) are retained. The whole job, including query/Git subprocesses, peaked at **552,538,112 committed bytes** (about 527 MiB), below its 4 GiB cap. This is job commit charge, not resident memory.

## Rebuild and unchanged maintenance

A separate clean build used the full phase-8 source commit `e96d36f662dab307cbfe5d9194b592b81c28e417`. It captured 3,218 inventory entries into a new ignored cache of **768,667,648 bytes**, with snapshot `b4ffe5746159cf296f43507491502bcef8669cc0b1f0d6e0aa8ae5c4b6c99f0d`. The existing baseline and overlay caches were preserved.

The clean build took **98.70 seconds**, including guarded-worker startup, and peaked at **1,000,566,784 committed bytes** (about 954 MiB). Existing source syntax warnings were visible; this operation parsed source without importing the indexed robotics packages. See the [rebuild time](evidence/phase9/phase9-rebuild.seconds.txt) and [resource receipt](evidence/phase9/phase9-rebuild.resources.csv).

Two unchanged refreshes took **2.52 and 2.65 seconds**, reused all **2,660** supported source/document analyses, parsed zero files, removed zero paths and did not publish or change the database write time. Their [first resource receipt](evidence/phase9/phase9-refresh.resources.csv), [confirmed resource receipt](evidence/phase9/phase9-refresh-verified.resources.csv) and corresponding [times](evidence/phase9/phase9-refresh-verified.seconds.txt) are retained. The confirmed full typed refresh receipt remains an ignored development artifact.

These are full-build and no-change costs, not the cost of an edited service. An observed completed-task workload is still needed to amortize update time and model spend. Reusing unchanged extraction does not imply that changed-file graph resolution and publication scale only with the diff.

## Failure and mutation evidence

The new tests exercise a real process exit after replacement starts, reads during an unfinished publication, actual SQLite page-cap exhaustion, lock-deadline failure, corrupt bytes, explicit quarantine/rebuild, and real MCP refusal/recovery. They retain the previous completed snapshot or return a visible error. They do not simulate power loss, filesystem corruption or arbitrary hardware failure.

Retrieval tests repeat the three methods after a committed source replacement and rebuild, refuse a different source revision before evaluation, preserve absent-symbol results and reject incorrect citation digests/text. An expected absent qualified symbol with same-name search hits fails evaluation. A correct definition with missing independently required paths also fails. Normal CI uses disposable repositories and mocked/offline boundaries, without model calls or credentials.

Final local package validation passed **269 tests**, with **one Windows-only symlink skip**, in **208.68 seconds** under the 4 GiB cap. Its [resource receipt](evidence/phase9/phase9-tests.resources.csv) records **760,553,472 peak job committed bytes**. The required formatter, Black and unused-import checks passed for all seven changed Python files, and the wheel built successfully. Details are recorded in [progress.md](progress.md). The new tests are collected by the existing `test/repository_knowledge_test` CI job. No hosted phase-9 run or full ROS/robotics suite is claimed.

## Release decision

Keep bounded structural navigation available and request rich context explicitly. Its 72–93-second measured latency is a material limitation, even though its memory use fits the current cap. [operations.md](operations.md) records conservative resource limits, recovery and retention rules.

Checkpoint E remains open. Representative generated-task success, actual model tokens, complete required-source recall, changed-service amortization, agreed team latency requirements and hosted validation remain unmeasured or unconfirmed. The deployed phase-7 checker/runtime/source/artifact pins are unchanged; this branch's newer protected code is not thereby certified for canonical publication. See the [release checklist](phase9-release.md) and [model-task experiment](phase9.md) before expanding deployment or optional infrastructure.
