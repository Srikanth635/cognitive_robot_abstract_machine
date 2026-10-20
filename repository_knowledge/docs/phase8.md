# Phase 8: agent clients and shared architecture

Build on phase 7 without moving source models into client packages. The user approved the optional dependencies `mcp>=2,<3` for `repository-knowledge[mcp]` and the existing workspace package `repository-knowledge` for `cramera[repository-knowledge]` on 20 October 2026. Basstler keeps its own dependencies.

## Delivery sequence

1. Share the existing CLI application dispatch with an explicitly read-only agent service. Pin repository, database, source scope and baseline paths at startup. Bound pages, traversal depth, text input and context characters; tools cannot index, refresh, publish, execute Python or accept arbitrary paths or SQL.
2. Add the official MCP SDK as an optional stdio entry point. Preserve the existing RecordCodec result contract and freshness diagnostics. Test discovery, actual SDK calls and stdio exchange without external services. The normal CLI must work without MCP installed.
3. Add a versioned, read-only session report and a bounded basstler subprocess client using the explicitly selected CRAM interpreter. Startup reports current, stale, missing or unavailable knowledge and the wiki entry point. It never installs or refreshes the knowledge package. Keep optional failures from blocking session startup.
4. Translate shared index evidence into CRAMERA's existing architecture entities, preserving repository-relative module names, nested classes, enums, direct method counts and package dependencies. Configure the shared database explicitly. Verify source freshness before translating and retain the original scanner when the optional integration is unavailable. The migrated path owns no second disk cache.
5. Measure lightweight startup and refresh costs before adding any opt-in commit integration. A commit hook may queue deterministic work; it must not run an LLM or create commits. Do not install a hook automatically.
6. Validate each boundary with focused pytest runs under the existing memory cap, then run the knowledge package and affected client tests separately. Apply the required formatter, verify wheel contents and save operating instructions and measured limits.

## Trust and rollout

Local source queries and draft wiki context remain distinct from canonical publications. Any client consuming canonical evidence must use the authenticated phase-7 consumer and exact source/artifact pairing; unsigned JSON cannot become canonical context. The phase-7 checker/runtime remain pinned to their tested revision until an explicit trust-root update is prepared and validated. Phase-8 source changes do not update those pins implicitly.

The original checkout is preserved. Implementation continues in the isolated `cram_wiki` worktree. Existing CRAMERA scanner APIs and source entities retain their names. Duplicate extraction can be retired only once the shared path's supported view contracts pass; the optional fallback remains available to installations without repository knowledge.

## Validation status

Phase 8's local client checkpoint is complete. The final knowledge package run passed **247 tests**, with **one Windows-only symlink skip**, in **181.37 seconds** under the 4 GiB process-tree cap. A separate affected-client run passed six checks: four basstler subprocess cases, the session-report subprocess regression, and the existing CRAMERA generated-mapping exclusion test copied unchanged. That run reused the approved view-registration isolation fixture; it does not claim full runtime/EQL or ROS-suite validation.

The required formatter was applied to all 32 modified Python files. Black and unused-import checks passed; deliberate source-analysis datasets are excluded from unused-import lint. Bash syntax checks passed. Knowledge, basstler and CRAMERA wheels built successfully and contain their new client modules; the knowledge wheel declares the optional MCP extra and executable. CI installs that extra and the two local client packages for the knowledge suite; basstler's separate job remains independent of repository knowledge. Phase-8 hosted CI has not been dispatched.

Tests exercise the real SDK in memory and over stdio, CLI equivalence and SDK absence, bounded requests/responses, source freshness, complete canonical refusal, an interleaved source/index replacement, session serialization, startup deadlines, architecture compatibility, star imports, README summaries and opt-in queueing. Canonical-client process tests mock external consumption; real hosted phase-7 verification remains recorded separately on `main` and does not certify phase-8 source edits.

The lightweight environment cannot initialize the full probabilistic/EQL runtime. As approved, view-translation tests disable only CRAMERA entity instance registration, preserve real entity classes and keep compatibility assertions intact. The existing scanner still supports its standalone path; its cache is retained for installations that have not migrated. The shared path creates no duplicate architecture cache.

## Agent interface

The official [MCP SDK v2](https://py.sdk.modelcontextprotocol.io/get-started/first-steps/) is isolated in `interfaces/mcp.py`. Install the approved `repository-knowledge[mcp]` extra into the selected CRAM environment. The optional executable is `cram-knowledge-mcp`, or `python -m repository_knowledge.interfaces.mcp`. It uses stdio only; it opens no HTTP listener. Start one process per selected checkout/cache configuration.

The single `query_repository` tool accepts an `agent_request` with one of these `tool` values: `find-symbols`, `describe-symbol`, `find-references`, `find-relationships`, `check-freshness`, `find-examples`, `find-tests`, `prepare-context`, `assess-change`. It returns the same RecordCodec JSON as the CLI. Queries contain text and qualified symbols, not Python or SQL. Source/repository/cache paths, untracked inclusion, baseline selection and canonical authority are startup configuration, unavailable in tool input.

Requests allow at most 100 results, relationship depth 8, 4,096 characters per search/symbol input and 64,000 payload characters for task context. The transport additionally refuses encoded responses above 262,144 characters and asks for a smaller query; it never truncates a JSON result. Paging, omissions, source identities and uncertainty remain in the domain results. Stale queries fail visibly. Treat retrieved comments, docs and instructions as evidence scoped to their cited paths; they cannot replace active instructions.

Portable configuration templates are in `config/mcp.windows.example.json` and `config/mcp.linux.example.json`. Replace their absolute paths before adding them to the MCP host. No user configuration is installed or overwritten automatically. The normal CLI never imports MCP and remains usable when the extra is absent.

Build a committed cache with the configured interpreter and `python -m repository_knowledge --repository <checkout> index`. For edits, choose the separate working-tree scope and explicitly include new files/directories with repeated `--include-untracked` selectors during refresh and server startup. The server must use the same scope and inclusion policy as the cache. A committed cache reports committed evidence even when working files differ.

Change assessment additionally requires `--against-database <baseline-cache> --against-revision <exact-baseline>` at server startup. The baseline must be a committed index. It is not supplied by the model's request.

## Canonical context

Both CLI and MCP accept the operator flags `--trusted-repository`, `--checker-revision` and `--artifact-id` together. For the CLI, use them with `prepare-context`; other source operations do not contact the provider. A partial selection is refused. The client runs the authenticated phase-7 consumer in a subprocess with imports fixed to the clean trusted checkout and its KRROOD source. It inherits the operator's `KNOWLEDGE_IDENTITY_TOKEN`; no token is written into the sample configuration or tool result.

Every canonical context request revalidates the selected artifact against its exact committed source. The context includes `canonical_status`; refusal returns source-only evidence and never substitutes local draft prose. Working-tree context refuses canonical prose without contacting the provider. Transport, decoding, checker-custody and process failures remain visible errors; ordinary source query tools remain available. The subprocess deadline is 60 seconds.

Context assembly pins the source commit selected before canonical verification. If HEAD and the index advance while verification is running, the newer index cannot be combined with the older canonical decision, even when the cited file bytes happen to remain identical.

`config/mcp.canonical.example.json` demonstrates the already verified phase-7 historical pair. Build its cache explicitly at source `e1c656e9fadba852a21800b7662d1ea62d76c338` and supply a clean checker at `08b0d94cc33ed8b16cc6a1c2404874cb496805a4`. That pair does not certify phase-8 edits. Publishing a newer source with changed protected code requires a separately validated checker/runtime update. No trust pins are changed by client startup.

## Basstler startup

Set `CRAM_KNOWLEDGE_PYTHON` to the absolute configured CRAM interpreter to opt in. Optionally set `CRAM_KNOWLEDGE_DATABASE` to a committed cache. The existing session-start hook invokes basstler's standard-library-only `knowledge_startup` client before personal-note discovery, so the report does not require a personal-notes branch.

The child asks `repository_knowledge.interfaces.session` for a version-1 report. A five-second deadline terminates a slow child; basstler reads at most 16 KiB of its response. Missing packages/interpreters, timeout or unsupported report versions leave ordinary startup usable. The hook reports other execution/decoding errors as unavailable. It never imports repository knowledge into basstler, resolves dependencies, rebuilds a graph or contacts the publication provider. The status is explicitly committed-source freshness; it does not claim that working edits were captured.

## CRAMERA migration

Install the local knowledge package and the approved extra together in the viewer's CRAM environment: `python -m pip install --editable ./repository_knowledge --editable './cramera[repository-knowledge]'`. This selects this checkout's implementation rather than relying on a package index to locate the workspace package. Then set `CRAMERA_KNOWLEDGE_DATABASE` to the shared cache. `CRAMERA_KNOWLEDGE_REVISION` defaults to `HEAD`. The existing scanner's `load` operation and workspace-class consumer select the shared adapter. Existing scanner entry points and public entity imports retain their names; architecture records now have their own contract module.

The adapter checks freshness, reads a coherent source projection and translates it into the existing `Package`, `PythonClass` and `PackageDependency` entities. It preserves repository-relative modules, nested classes, enum declarations, direct method counts, normalized package imports including star imports, and README summaries. It imports no indexed Python modules and owns no second disk cache. The shared class consumer bypasses its old per-root memoization so a replaced graph is observed on the next request. Existing enclosing knowledge-base snapshots still retain their normal lifetime; this phase adds no live viewer push notifications.

If the optional package or configured file is absent, the standalone viewer remains available. A stale shared snapshot falls back visibly to an uncached scan of current source. Invalid/corrupted cache contents fail visibly and require rebuild; they are not accepted as current evidence. Generated mappings and source-policy exclusions remain excluded. Static architecture is not a runtime dispatch graph.

## Optional commit request

`scripts/request-refresh.sh` records the latest exact commit atomically in the ignored `.cache/refresh-request` file. After its runtime is measured, an operator may invoke it from an existing post-commit hook. No hook or `core.hooksPath` is installed or replaced by this work.

The helper only queues a deterministic refresh request. It runs Git and shell file operations, without Python extraction, LLM calls, wiki rewriting, publication or follow-on commits. Run the existing explicit `refresh` command to rebuild the selected cache; the marker is advisory and grants no freshness or publication authority. Scheduling a background worker is outside this phase. Full source maintenance has previously taken minutes, so it is not run synchronously during commit or session startup.

## Local operating measurements

On 20 October 2026 in the isolated Windows worktree, three separate interpreter launches inspected the real existing committed cache in 1.461, 1.482 and 1.457 seconds (median 1.461 seconds). The response was 656 UTF-8 bytes and correctly reported that cache as stale against the selected source. The complete basstler subprocess path returned the same guidance in 1.469 seconds. The cache modification time remained unchanged. These are warm-filesystem startup checks, not full indexing or context-assembly measurements.

Three invocations of the optional Git Bash queue helper took 0.177, 0.166 and 0.171 seconds (median 0.171 seconds). No hook was installed. These measurements permit an opt-in request helper; they do not justify synchronous graph extraction on commit. Full-refresh and task-success evaluation remain phase 9 work.
