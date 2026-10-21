# Delivery progress

Baseline: main at `7b27424de7768798dc561f447e11c2a8eb86c307`.

Implementation branch: `cram_wiki`, in an attached managed worktree. Phases 0–8 are on the fork; phase 9 adds local release-validation evidence and operating guidance below. The original checkout's staged deletions and untracked replacement files were preserved.

| Phase | Delivered |
| --- | --- |
| 0 | Saved assessment and phased plan; isolated Python 3.12 environment; approved existing dependencies; fixture identity approval; fixed navigation tasks and local baseline measurements; memory-capped package test runner |
| 1 | Dataclass source/snapshot/query model; explicit source and persistence contracts; qualified symbol identities; custom errors; import-boundary and cycle checks |
| 2 | Exact Git-commit inventory; manifest-derived import roots; Python declarations and lexical references; alias/re-export/generic-base handling; type-only context; Markdown/RST headings and explicit local Markdown links; visible exclusions and parser diagnostics |
| 3 | Atomic SQLite publication; selective evidence reads; bounded symbol/reference/relationship queries; coherent freshness checks; typed JSON CLI; CI package entry |
| 4 | Content-bound working-tree capture; explicit untracked inclusion; checkout/index identity; reusable file analysis; conservative target invalidation; stale-query refusal; refresh/reconciliation commands; mutation-equivalence tests |
| 5 | Curated OKF import/export; eight source-backed pilot concepts; separate format, policy and evidence checks; reverse invalidation; lifecycle/tombstone proposals; bounded wiki CLI; explicit LF/CRLF equivalence policy |
| 6 | Bounded task context, examples, tests and change assessment with source citations and explicit uncertainty |
| 7 | Automated trusted publication, authenticated artifact consumption and verified historical restoration on the fork |
| 8 | Optional MCP client, source-bound canonical context, bounded basstler startup, shared CRAMERA architecture and opt-in refresh requests |
| 9 | Local retrieval comparison, storage/client recovery checks, memory receipts, retention/recovery guidance and an explicit checklist of remaining team-release gates |

Phases 0–3 validation: 40 focused pytest cases passed under the 4 GiB Windows job cap; the requested rerun also passed all 40 cases in 23.92 seconds. The required formatting script was applied to the new Python files. Black and unused-import checks passed, and the package built as a wheel. Main-branch indexing and six independently checked navigation tasks passed; measurements are in [evaluation.md](evaluation.md).

Final shared-fixture cleanup was verified with all 3 storage tests passing. The main-branch freshness command returned a matching commit and exited successfully in 1.43 seconds. Black's final check left all 44 valid Python files unchanged; the deliberately invalid syntax fixture is excluded from that check.

## Phase 4 validation — 20 October 2026

The final focused-suite rerun passed **75 tests**, with **1 Windows-only symlink test skipped**, in 71.80 seconds under the 4 GiB job cap. Linux CI is configured to exercise the symlink case. The required formatting script was applied; Black left all 61 valid Python files unchanged, unused-import checks passed, and the updated wheel built successfully. No KRROOD changes or new dependencies were introduced.

Mutation tests cover incremental/clean equivalence, changed exports and consumers, source-root changes, staged/untracked additions, deletions, renames, branch changes and parse failures. Capture interleaving and query-time edits are rejected. Generated mappings are tested with a filesystem-read trap. Scope replacement, legacy-cache refusal, explicit untracked inclusion and ignored-file policy are covered.

A real overlay of main plus explicitly selected tool source captured 3,034 entries in 136.11 seconds. An unchanged refresh reused 2,506 analyses, parsed zero files, skipped publication and took 8.53 seconds. Live queries returned the expected current refresh implementation and CRAM execution method. The separate committed cache was rebuilt in format 2 and its freshness check passed. See [evaluation.md](evaluation.md) for the measured scope and limits.

## Scope boundaries

Committed mode still reads only the selected revision. Working-tree mode captures current tracked files and explicitly included untracked source in a separate overlay. See [phase4.md](phase4.md) for the capture, invalidation and publication contract.

`wiki/` contains the root and package/workflow/convention entry points and eight source-backed draft concepts. Phase 5 adds OKF exchange and separate format/policy/evidence validation. Generated structural facts remain in the ignored SQLite cache.

Static resolution is conservative in scope but is not a Python type checker or control-flow analysis. Dynamic dispatch, conditional bindings, overloads, decorator effects, enum-base aliases, string annotations and unsupported document syntax require care. Repeated declarations retain source locations; lexical resolution currently uses the final collected binding and does not establish which branch executes.

The initial reader expects well-formed project manifests and Python encoding declarations. Malformed metadata/encoding can abort capture rather than produce a per-file diagnostic. That failure is visible and cannot publish a partial replacement; broader input recovery needs a separately tested parser boundary.

SQLite JSON functions support selective queries in the tested Python environment. Lexical search is implemented; optional full-text indexing has not been added. Broad queries and deep relationship traversals can still load substantial evidence. Context token budgeting and example/test selection are phase 6 work.

The package uses the approved KRROOD, typing-extensions and PyYAML dependencies. KRROOD source remains unchanged. The root UV configuration issue is recorded in [architecture.md](architecture.md); no unrelated packaging cleanup was included.

## Phase 5 validation — 20 October 2026

The final package run passed **132 tests**, with **1 Windows-only symlink test skipped**, in **98.59 seconds** under the 4 GiB job cap. This includes the original navigation suite and new interchange, source-evidence, lifecycle, CLI, input-boundary and LF/CRLF mutation cases. Hosted CI was not executed from this task.

The required formatting script was applied. Black left all 74 valid Python files unchanged, unused-import checks passed, and the wheel built successfully. The original KRROOD serializer remains unchanged; the new boundary also carries nested metadata mappings through its dataclass serializer.

All eight pilot concepts passed committed-main format, CRAM policy and structural evidence checks. They remain drafts without authenticated review. The third-party OKF CLI v0.4.0 read both the pilot format and its metadata, returned zero lint errors/warnings, and found the generic-dataclass concept through lexical search. It is not a dependency and no hook was installed. See [phase5.md](phase5.md) for the interface, evidence contract and limits.

Exporting and re-importing the full bundle preserved its supported fields and navigation. The exported bundle also passed the independent CLI's lint. A real cross-directory role-concept move produced a nine-concept proposal containing the old tombstone and the new draft; that proposal passed validation against main.

The Windows working-tree trial exposed byte differences from LF/CRLF checkout conversion. Validation now proves recognized text equivalence using both original and selected byte digests and reports those paths explicitly. Tests also preserve exact-byte policy, reject arbitrary byte changes/BOM changes, and retain byte-exact source-index freshness.

Final live working-tree validation passed all eight concepts with zero format, policy or evidence findings, explicitly reporting 16 distinct text-equivalent source files. The overlay contains 3,040 inventory entries and its current snapshot is `a157fbbd20d9f4dc76b1155804709bb7a37dcbd15f682b7de93c7d9445b726ef`. The final refresh analyzed one file, reused 2,511 analyses and took 293.77 seconds. Working validation took 23.64 seconds; the bounded execution query returned the two expected concepts. These costs remain explicit limits of the conservative full-graph maintenance design.

## Next phase

Phase 9 local validation is recorded below. Supported team release remains conditional on the open gates in [phase9-release.md](phase9-release.md). Optional phase-10 capabilities require measured justification.

## Phase 6 validation — 20 October 2026

Implemented `find-examples`, `find-tests`, `prepare-context` and `assess-change` as read-only operations and CLI commands. Domain records and abstract test discovery remain independent of adapters. Profiles share discovery while ranking definitions, public interfaces, instructions, guidance, concepts, examples and tests for their task. Character budgets, sampled pages, partial source excerpts, invalid concepts and absent categories remain explicit.

Static pytest evidence includes fixture names, declared scope, autouse dependencies, nearest conftest overrides, class visibility and parametrization/runtime uncertainties. Proposed test directories are never labeled complete coverage. ORM tasks select generators and resolved source inputs without reading generated interfaces. Change reports retain baseline consumers for removed APIs and compare declared public contracts against a separately pinned source snapshot.

The complete package suite passed **159 tests**, with **one Windows-only symlink skip**, in **111.54 seconds** under the 4 GiB cap. The required formatter was applied; Black left all **91** valid Python files unchanged, unused-import checks passed, and the wheel built successfully. No new dependencies or KRROOD changes were introduced.

The fixed five-seed main pilot returned the required evidence categories and validated every included source citation. World also included spatial guidance. Initial live evaluation exposed weak lexical test candidates displacing useful context; a failing ranking test preceded the correction to prefer resolved consumers and relevant packages. Full graph decoding and repeated test discovery remain material latency limits, recorded in [evaluation.md](evaluation.md). The capability meets local pilot checkpoint C; model task-success, token-use and production trust claims remain unvalidated.

The impact trial additionally exposed a large consumer result for World. Change reports now preserve complete match counts while paging the baseline/current resolved/uncertain lists. A failing output-budget test preceded that correction. The final overlay refresh completed in 148.16 seconds and a subsequent freshness check passed in 4.67 seconds. Its snapshot is `1d194c4b36b3b90641fd5170737e256e8de218a9286700011eece1c0efd9026a`; the selected scope includes current tracked source plus explicitly included untracked tool source and package metadata.

## Phase 7 implementation — 20 October 2026

Added committed-bundle capture, trusted policy and clean checker custody, deterministic extraction fingerprints, admission/lifecycle checks, authenticated source and optional review adapters, immutable artifact validation, publication/consumption commands and fork-only CI workflows. The configured default is automated with zero required reviews. Another person's approval is not required. Source consistency and human review remain separate.

The workflows require owner-selected immutable checker/runtime pins before activation. The fork Actions service is enabled. Hosted runtime, admission, initial publication and authenticated consumption have passed; see [the deployment checkpoint](phase7-deployment.md). Existing local caches are not canonical evidence. Phase 8 client integration remains separate. After the authorized workflow-test syntax correction, the complete package run passed **206 tests**, with **one Windows-only skip**, in **163.94 seconds** under the 4 GiB job cap. Subsequent focused tests cover publication/source-identity serialization and refusing policy reads before checker custody. The wheel built successfully; Bash syntax validation passed. Black and unused-import checks pass for all valid files.

The workflow assertions now pass. Activation preparation also adds a dedicated Python 3.12 runtime recipe with hash-pinned existing dependencies and a fork-only image-build workflow; its focused checks pass. The existing ROS image does not install the checker dependency closure. Hosted activation results will be recorded separately.

The final activation preparation includes five workflows: admission (PR or manual source/base selection), runtime build, shared activation-pin reading, publication and explicit artifact consumption. The new hosted-consumption workflow and dispatch checks passed all four focused workflow tests. The runtime installs only the existing import closure, with hash-pinned Linux wheels; checker and KRROOD source remain outside the dependency image.

Remote operation uses owner-managed activation pins from `main` plus optional dedicated trigger refs. Source proposals cannot select the executable checker or acquire publisher-ref identity. Registry credentials stay outside the admission container. The newly added activation-file tests reject shell injection.

Final local validation after remote-activation changes passed **211 tests**, with **one Windows-only skip**, in **152.81 seconds** under the 4 GiB cap. Black left all 113 valid Python files unchanged; unused-import and Bash syntax checks passed. The new setup correction selects Git Bash on Windows and keeps the existing assertions unchanged.

## Hosted phase 7 checkpoint

The runtime build, real-source admission, first publication and first artifact-consumption check passed on the fork. The first canonical artifact is `12292221849`, bound to source/checker `08b0d94cc33ed8b16cc6a1c2404874cb496805a4`. See [phase7-deployment.md](phase7-deployment.md) for immutable identities and run links. The final pair and historical-restoration result are retained in the owner-managed verification report on `main`.


## Phase 8 validation — 20 October 2026

The optional MCP stdio adapter and existing CLI share one application dispatcher. The tool surface contains bounded read-only queries, with source/cache/baseline/authority fixed at startup. Canonical context delegates to the independently pinned checker process, refuses working-source or wrong-revision prose and never replaces a refused artifact with local drafts. Context assembly pins the verified source even if HEAD and the cache advance during provider verification.

Basstler startup uses a standard-library-only client and the explicitly selected CRAM interpreter, with a five-second deadline and 16 KiB response allowance. It neither imports the robotics stack nor installs/rebuilds anything. CRAMERA translates the shared snapshot into its existing entities and bypasses its old class memoization on that path. The original scanner remains an optional fallback; migrated reads create no second disk cache. The optional commit helper only records the latest refresh request; it is not installed automatically and performs no LLM call or commit.

Final local validation passed **247 package tests**, with **one Windows-only skip**, in **181.37 seconds** under the 4 GiB cap. A separate six-check run covered basstler, the session-process round trip and the unchanged legacy generated-file exclusion check with the approved view-registration fixture. All 32 modified Python files passed formatting; unused-import and Bash syntax checks passed. The three affected packages built as wheels with their new modules and optional dependency metadata.

Real committed-cache inspection had a 1.461-second median across three interpreter launches and correctly reported the old cache as stale without writing it. The complete basstler path returned in 1.469 seconds. The optional queue helper's median was 0.171 seconds. These are startup/request costs, not full graph maintenance or model-task results.

See [phase8.md](phase8.md) and the configuration templates for setup and limits. Host configuration, viewer opt-in, commit-hook installation and a background queue worker were not enabled automatically. The phase-7 checker/runtime/source/artifact selection on main remains unchanged; new protected code needs a separately validated trust-root update before newer canonical publication. No phase-8 hosted run or full ROS/EQL test suite is claimed.

## Phase 9 local validation — 21 October 2026

Added a reproducible operator-only evaluation module for committed source search, structural CLI and local curated context. It pins one source, audits returned evidence against independent committed blobs and preserves CSV observations without inventing model metrics. Incorrect line text/digests, a mismatched revision, unexpected same-name matches for an absent qualified symbol and missing independently required paths are covered by failing-first tests. Controlled source replacements repeat all three retrieval methods after rebuild.

Recovery tests cover abrupt writer termination, concurrent readers, SQLite capacity exhaustion, lock refusal, corruption, quarantine/rebuild and real MCP refusal/recovery. Existing storage transactions already preserve complete snapshots in these cases; no storage rewrite, new dependency or KRROOD change was needed. The shared guarded command worker now also supports benchmark/extraction commands and optional peak Windows job commit-charge receipts.

The complete isolated package suite passed **269 tests**, with **one Windows-only symlink skip**, in **208.68 seconds** under the **4 GiB** job cap. Peak job commit charge was **760,553,472 bytes**. A final fixture-entry cleanup was followed by all six recovery tests passing again in 4.20 seconds. All seven changed Python files passed the required formatting script, Black and unused-import checks. The wheel built at **124,178 bytes**, SHA-256 `f4b09e4af975542e1a8404fc4bed0d6332bff76cc60670a6e86cc866baa5a884`.

All 18 real retrieval trials found their expected definition and all **157/157** returned citations passed independent source checks. Median source/structural/context latencies were **0.081/1.564/76.786 seconds**. Context is explicitly opt-in because the six measured calls took **72–93 seconds**. The comparison job peaked at about **527 MiB** of commit charge. Source recall here covers definition files, not complete coding-task evidence.

A clean phase-8 source index captured **3,218 entries** in **98.70 seconds**, peaking at about **954 MiB**, and its freshness and exact LocalApplication query passed. Two unchanged refreshes reused **2,660 analyses**, parsed none and skipped publication in **2.52/2.65 seconds**. Cache write time was unchanged. The original baseline and overlay caches were preserved; no optional integration, hook, background worker or publication activation was enabled.

See [phase9-evaluation.md](phase9-evaluation.md) for the raw evidence, [operations.md](operations.md) for limits/retention/rebuilds/rollback, and [phase9-release.md](phase9-release.md) for open gates. Model generation success, actual token use, edited-service amortization, team latency targets, hosted phase-9 CI and a successor canonical trust-root activation remain unconfirmed. Checkpoint E is open; local validation does not certify a supported team release.

## Model playground and local setup — 21 October 2026

Added the optional provider-selectable [notebook](../notebooks/knowledge_playground.ipynb), read-only MCP query adapter, credential-free typed trial records, and packaged model instructions. The notebook supports source-grounded Q&A, code proposals, query receipts, reported token usage and optional ignored local reports. Default execution needs no provider account and makes no model request. The existing LiteLLM tool loop uses an empty scratch sandbox and receives no repository editing or command-execution tool. Source changed before or during a turn is refused. KRROOD remains unchanged.

The complete isolated package run passed **284 tests**, with **one Windows-only skip**, in **277.95 seconds**, under the **4 GiB** job cap. Peak job commit charge was **1,210,077,184 bytes**. Tests use mocked provider responses with the actual SDK loop and real MCP subprocesses, and nbclient executes every notebook cell against fixture evidence with the configured Python. A subsequent failing-first inspection check now also refuses mismatched discovery/declaration snapshots; all **16** final notebook checks passed in **50.27 seconds**. All eight changed Python files passed the required formatter, Black and unused-import checks. The optional modules and prompt are included in the final built wheel (**131,314 bytes**, SHA-256 `b6bbc63da58fbb9b1323b296b171789d7d67f9bbcd1023358788ce16c1a2438b`). PowerShell launcher syntax checks passed. Package CI and the local runner disable import-time cost-map fetching; CI installs the optional notebook extra.

The setup uses a separate current working-tree cache, a read-only Codex MCP entry with a 300-second query deadline, and the **CRAM Knowledge** kernel through the existing Windows resource guard. See [notebook.md](notebook.md) for refresh commands, credentials, an ignored interactive copy and limits. Host configuration needs a fresh session/reload before its tools appear. Local activation is checked separately after the source commit; these tests make no hosted-CI, live-provider quality, token-saving or successor canonical-publication claim.
