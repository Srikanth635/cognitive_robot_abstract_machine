# Repository knowledge operations

The wiki and durable plans live in `repository_knowledge/wiki/` and `repository_knowledge/docs/`. Rebuildable graphs, refresh requests, wheels and temporary reports live in ignored `repository_knowledge/.cache/`. Source, local draft prose and authenticated publication evidence have different recovery paths. Ordinary query clients never repair or publish canonical knowledge.

## Local operating envelope

Use the configured Python 3.12 interpreter. On the measured Windows machine, require 8 GiB free physical memory before extraction, evaluation or pytest and cap the entire process tree at 4 GiB with `scripts/run_command.ps1` or `scripts/run_tests.ps1`. They share one guarded worker; the child begins execution only after job attachment. Run one extraction, test run or evaluation at a time. A cap termination is a failed operation; it cannot certify freshness.

Keep the existing read limits: at most 100 navigation results, relationship depth 8, and 80 excerpt lines. MCP permits at most 64,000 context payload characters and 262,144 encoded response characters. Larger CLI context allowances do not expand the MCP transport limit. Check omissions and uncertain relationships before generating a change.

For agent navigation, start with a name search or an exact qualified declaration query, then request the relevant relationship page. Use `find-examples` and `find-tests` before extending an API. Request rich `prepare-context` explicitly when the additional contracts, instructions, examples and fixtures justify its latency. Before editing, check the actual source files and the report's source scope; historical committed evidence does not include working edits. Validate the resulting change with the suggested checks plus any task-specific tests, then refresh separately. An unresolved candidate never proves a caller or complete test coverage.

Startup remains limited to five seconds and 16 KiB. Canonical verification has a 60-second subprocess deadline. The evaluation runner uses a 300-second deadline per retrieval. These are fail-visible ceilings, not latency promises. Full extraction and graph context assembly do not belong in synchronous commit hooks or startup. The measured six-task report determines whether context is practical for a particular workflow; ordinary source search remains available.

Use 2 GiB per cache and 4 GiB total local cache storage as investigation thresholds for this pilot, not established scalability limits. Two measured caches total about 1.4 GiB before temporary files. Keep at least 4 GiB additional free disk space for a candidate rebuild and SQLite rollback journals at this scale. If growth crosses a threshold, stop automatic refresh, inspect scope and reserve more space before explicitly rebuilding. These disk thresholds are operator checks, not an automatic quota or garbage collector. Memory and subprocess deadlines are enforced.

## Retention and worktree bounds

Retain one current committed cache, one working overlay if used, and at most one explicitly needed comparison cache. After a replacement passes freshness and consumer checks, remove its superseded rebuildable cache with clients stopped. Never remove SQLite journal/WAL/SHM companions from a database that might still be open. Do not delete a hot rollback journal: SQLite needs it to recover an interrupted publication.

Keep evaluation CSVs, resource CSVs and the exact source/implementation identities for accepted measurements in version-controlled documentation. Raw repeated trials, temporary wheels and failed rebuilds stay in `.cache/`; review and remove them after their report is accepted. Keep the last failed cache in quarantine only until diagnosis and successful rebuild, subject to the same disk budget. Source and plans are never cache-cleanup targets. No automatic deletion is installed.

Evaluation reads committed Git blobs and needs no new worktree. Reuse the active source checkout and, for canonical consumption, one separately pinned clean checker. Limit a single operator's knowledge workflow to these two working checkouts plus one temporary recovery checkout while recovery is active. Inspect `git worktree list --porcelain` before adding one. Never archive another chat's checkout or the user's original checkout. Managed worktrees are archived through the app's recoverable worktree operation after needed artifacts are saved; ordinary disposable worktrees are removed only after confirming their absolute path, ownership and clean state. Concurrent independent users need their own explicit capacity budget.

Published Actions artifacts currently expire after 90 days. The phase-7 canonical artifacts were published on 20 October 2026 and require retention action before 18 January 2027 if historical consumption must remain available. Local copies of artifact JSON are useful for audit but cannot replace authenticated provider identity. Before expiry, retain the receipt/digest, workflow run, exact source/checker commits, policy, pinned runtime image and verification result, and arrange owner-managed provider retention or a separately verified new publication. Keep the last known good source/artifact/checker/runtime tuple while deploying a successor. Never delete or repoint trust refs as cache cleanup. The pinned runtime image and Git source must remain retrievable too.

## Diagnose before changing data

Read the session report or run `check-freshness` with the same source scope, database and revision as the failing client. Missing cache means use source navigation and build explicitly. Stale committed evidence may still describe its exact historical commit; it does not describe HEAD or working edits. A working overlay captures only tracked paths plus explicitly included untracked paths.

A corrupt SQLite file, unknown serialized type or unsupported schema fails visibly. Stop affected optional clients and preserve the file and any journals for diagnosis. A full disk, lock deadline or interrupted writer must not be treated as a successful refresh. The recovery tests demonstrate transaction rollback and process restart for the current storage implementation; they do not simulate power loss, filesystem damage or arbitrary hardware faults.

## Full rebuild and schema changes

Prefer a new candidate file rather than overwriting the only usable cache. Pin the source once, check free memory/disk, then build under the resource cap. This example is for committed source in the active worktree:

```powershell
$repository = (Get-Location).Path
$revision = (& git -C $repository rev-parse --verify 'HEAD^{commit}').Trim()
$candidate = "repository_knowledge/.cache/rebuild-$revision.sqlite3"
if (Test-Path -LiteralPath $candidate) { throw 'Select a new candidate path.' }
./repository_knowledge/scripts/run_command.ps1 -Module repository_knowledge `
  -Arguments @('--repository', $repository, '--database', $candidate,
    '--revision', $revision, 'index')
if ($LASTEXITCODE -ne 0) { throw 'The candidate rebuild failed.' }
./repository_knowledge/scripts/run_command.ps1 -Module repository_knowledge `
  -Arguments @('--repository', $repository, '--database', $candidate,
    '--revision', $revision, 'check-freshness')
if ($LASTEXITCODE -ne 0) { throw 'The candidate does not match the selected source.' }
```

Run an expected symbol query before switching a consumer to the candidate. Stop/reconfigure/restart clients together so their source revision and database remain paired. Keep the previous file until the consumer check passes. If HEAD advanced, choose whether to use the explicit historical revision or build again; do not label that candidate current for a newer revision.

For an overlay, select `--scope working-tree` and a separate database, using the same explicit `--include-untracked` values on indexing, refresh and querying. After checkout-wide changes or suspected incremental drift, use `reconcile` on a compatible cache or build a new one. Neither operation repairs wiki prose. Validate source-backed concepts separately with the documented wiki checks.

For a schema/extractor incompatibility, keep the previous implementation/cache pair until rollback is unnecessary, install the selected implementation in its configured environment and perform a clean rebuild. Do not edit SQLite payloads, downgrade its format number, or migrate a corrupt cache by hand. Generated ORM files remain excluded; regenerate them only through the existing project generator if an actual ORM task requires it.

## Corruption, interrupted publication and capacity failures

Stop writers before quarantine or replacement. An ordinary abrupt writer exit leaves SQLite to recover its journal on the next open; let that complete before assessing corruption. Verify freshness and a known symbol afterward. If opening still fails, retain the database and companions under a clearly identified quarantine location and build a new candidate from exact source. Do not use a copied main file without its active journal as proof of a coherent backup.

On SQLite capacity exhaustion, free or reserve space and retry explicitly. The old snapshot survives the tested transaction failure; its commit still needs checking. A new database that failed during initial construction has no successful publication and should be quarantined or discarded after diagnosis. Do not repeatedly restart a refresh worker against a full disk. Query lock timeouts are visible failures; avoid long-lived external database transactions and stop competing writers before retrying.

## Disable integrations and roll back

Disable basstler inspection by unsetting `CRAM_KNOWLEDGE_PYTHON` and its optional database selection. Disable the shared CRAMERA source by unsetting `CRAMERA_KNOWLEDGE_DATABASE` and `CRAMERA_KNOWLEDGE_REVISION`; the standalone scanner remains available. Remove the selected MCP server from the host configuration and restart that host. Remove only the knowledge refresh invocation from an existing commit hook, preserving other hook commands. The advisory refresh-request file can then be removed with writers stopped. There is no installed queue worker to disable.

For a client regression, select the previous tested implementation in a separate environment and pair it with a matching source/cache format. Do not reset a dirty checkout. For canonical rollback, select the complete older source/artifact/checker/policy/runtime tuple and run authenticated consumption. This returns historical evidence; it never resets source or authorizes mixing older prose with newer code. Expired artifacts or changed trust policy can prevent rollback and must be restored through owner-managed trust configuration, not bypassed with local draft files.

Phase 9 does not change the deployed phase-7 pins. A newer checker or protected runtime change requires its own clean custody, runtime build, admission, publication and consumption validation before activation. No other person's approval is required in the configured automated mode.
