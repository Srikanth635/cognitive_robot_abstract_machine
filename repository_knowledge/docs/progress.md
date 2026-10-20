# Delivery progress

Baseline: main at `7b27424de7768798dc561f447e11c2a8eb86c307`.

Implementation branch: `cram_wiki` (phases 0–6 pushed to the fork at `ba38cd3033762774c7c940cf714516a5e6dc0747`), in an attached managed worktree. The original checkout's staged deletions and untracked replacement files were preserved.

| Phase | Delivered |
| --- | --- |
| 0 | Saved assessment and phased plan; isolated Python 3.12 environment; approved existing dependencies; fixture identity approval; fixed navigation tasks and local baseline measurements; memory-capped package test runner |
| 1 | Dataclass source/snapshot/query model; explicit source and persistence contracts; qualified symbol identities; custom errors; import-boundary and cycle checks |
| 2 | Exact Git-commit inventory; manifest-derived import roots; Python declarations and lexical references; alias/re-export/generic-base handling; type-only context; Markdown/RST headings and explicit local Markdown links; visible exclusions and parser diagnostics |
| 3 | Atomic SQLite publication; selective evidence reads; bounded symbol/reference/relationship queries; coherent freshness checks; typed JSON CLI; CI package entry |
| 4 | Content-bound working-tree capture; explicit untracked inclusion; checkout/index identity; reusable file analysis; conservative target invalidation; stale-query refusal; refresh/reconciliation commands; mutation-equivalence tests |
| 5 | Curated OKF import/export; eight source-backed pilot concepts; separate format, policy and evidence checks; reverse invalidation; lifecycle/tombstone proposals; bounded wiki CLI; explicit LF/CRLF equivalence policy |

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

Phase 7 adds owner-operated machine validation and canonical publication, with independent review optional. See [phase7.md](phase7.md). Phase 6 delivery is recorded below; hosted CI and production deployment are not claimed.

## Phase 6 validation — 20 October 2026

Implemented `find-examples`, `find-tests`, `prepare-context` and `assess-change` as read-only operations and CLI commands. Domain records and abstract test discovery remain independent of adapters. Profiles share discovery while ranking definitions, public interfaces, instructions, guidance, concepts, examples and tests for their task. Character budgets, sampled pages, partial source excerpts, invalid concepts and absent categories remain explicit.

Static pytest evidence includes fixture names, declared scope, autouse dependencies, nearest conftest overrides, class visibility and parametrization/runtime uncertainties. Proposed test directories are never labeled complete coverage. ORM tasks select generators and resolved source inputs without reading generated interfaces. Change reports retain baseline consumers for removed APIs and compare declared public contracts against a separately pinned source snapshot.

The complete package suite passed **159 tests**, with **one Windows-only symlink skip**, in **111.54 seconds** under the 4 GiB cap. The required formatter was applied; Black left all **91** valid Python files unchanged, unused-import checks passed, and the wheel built successfully. No new dependencies or KRROOD changes were introduced.

The fixed five-seed main pilot returned the required evidence categories and validated every included source citation. World also included spatial guidance. Initial live evaluation exposed weak lexical test candidates displacing useful context; a failing ranking test preceded the correction to prefer resolved consumers and relevant packages. Full graph decoding and repeated test discovery remain material latency limits, recorded in [evaluation.md](evaluation.md). The capability meets local pilot checkpoint C; model task-success, token-use and production trust claims remain unvalidated.

The impact trial additionally exposed a large consumer result for World. Change reports now preserve complete match counts while paging the baseline/current resolved/uncertain lists. A failing output-budget test preceded that correction. The final overlay refresh completed in 148.16 seconds and a subsequent freshness check passed in 4.67 seconds. Its snapshot is `1d194c4b36b3b90641fd5170737e256e8de218a9286700011eece1c0efd9026a`; the selected scope includes current tracked source plus explicitly included untracked tool source and package metadata.

## Phase 7 implementation — 20 October 2026

Added committed-bundle capture, trusted policy and clean checker custody, deterministic extraction fingerprints, admission/lifecycle checks, authenticated source and optional review adapters, immutable artifact validation, publication/consumption commands and fork-only CI workflows. The configured default is automated with zero required reviews. Another person's approval is not required. Source consistency and human review remain separate.

The workflows require owner-selected immutable checker/runtime pins before activation. No remote settings were changed and no hosted publication has been performed. Existing local caches are not canonical evidence. Phase 8 client integration remains separate. After the authorized workflow-test syntax correction, the complete package run passed **206 tests**, with **one Windows-only skip**, in **163.94 seconds** under the 4 GiB job cap. Subsequent focused tests cover publication/source-identity serialization and refusing policy reads before checker custody. The wheel built successfully; Bash syntax validation passed. Black and unused-import checks pass for all valid files.

The workflow assertions now pass. Activation preparation also adds a dedicated Python 3.12 runtime recipe with hash-pinned existing dependencies and a fork-only image-build workflow; its focused checks pass. The existing ROS image does not install the checker dependency closure. Hosted activation results will be recorded separately.

The final activation preparation includes five workflows: admission (PR or manual source/base selection), runtime build, shared activation-pin reading, publication and explicit artifact consumption. The new hosted-consumption workflow and dispatch checks passed all four focused workflow tests. The runtime installs only the existing import closure, with hash-pinned Linux wheels; checker and KRROOD source remain outside the dependency image.

Remote operation uses owner-managed activation pins from `main` plus optional dedicated trigger refs. Source proposals cannot select the executable checker or acquire publisher-ref identity. Registry credentials stay outside the admission container. The newly added activation-file tests reject shell injection.

Final local validation after remote-activation changes passed **211 tests**, with **one Windows-only skip**, in **152.81 seconds** under the 4 GiB cap. Black left all 113 valid Python files unchanged; unused-import and Bash syntax checks passed. The new setup correction selects Git Bash on Windows and keeps the existing assertions unchanged.
