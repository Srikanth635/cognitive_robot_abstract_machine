# Repository knowledge

This directory contains the source-navigation tool, task assistance, development records and OKF wiki.

| Directory | Purpose | Version controlled |
| --- | --- | --- |
| `src/repository_knowledge/` | Models, contracts, operations, adapters and CLI | Yes |
| `docs/` | Saved design, phased plan, decisions and validation | Yes |
| `wiki/` | OKF bundle: package, workflow and convention concepts | Yes |
| `scripts/` | Memory-capped Windows test runner | Yes |
| `config/` | Optional MCP host configuration examples | Yes |
| `.cache/` | Rebuildable SQLite indexes | No |
| `.venv/` | Local Python 3.12 environment | No |
| `../test/repository_knowledge_test/` | Tests and source datasets | Yes |

The wiki is a directory tree of Markdown files. Its root `index.md` is the agent entry point. The development plan belongs in `docs/`, outside the OKF bundle. Generated source facts live in the database, so the wiki does not need a document per method.

Read [the build plan](docs/plan.md) and [the architecture decisions](docs/architecture.md).

Phases 0–6 provide source navigation, working-tree overlays, curated knowledge and task context. See [delivery progress](docs/progress.md), [measured validation](docs/evaluation.md) and [the overlay design](docs/phase4.md).

## Install and query

Use the configured CRAM Python 3.12 environment with KRROOD installed. From the repository root:

```powershell
python -m pip install --no-deps --editable ./repository_knowledge
cram-knowledge --repository . --revision main index
cram-knowledge --repository . --revision main find-symbols GiskardExecutable
cram-knowledge --repository . --revision main describe-symbol coraplex.plans.executables.GiskardExecutable.execute
cram-knowledge --repository . --revision main find-references coraplex.plans.executables.GiskardExecutable --limit 10
cram-knowledge --repository . --revision main find-relationships coraplex.plans.executables.GiskardExecutable.execute --depth 1
cram-knowledge --repository . --revision main check-freshness
```

The task's focused interpreter is `repository_knowledge/.venv/Scripts/python.exe`; the console executable is beside it. This environment and its installed packages are local, ignored state. The tool is separately installable and is not added to the root workspace dependencies.

Use `--database PATH` before the command to select another disposable index. Committed mode defaults to `repository_knowledge/.cache/index.sqlite3`. A new index replaces the previous snapshot atomically. Committed queries refer to the requested **commit**.

## Working-tree overlays

Use working-tree scope for current tracked files, including staged additions, unstaged changes and deletions:

```powershell
cram-knowledge --repository . --scope working-tree index
cram-knowledge --repository . --scope working-tree describe-symbol coraplex.plans.executables.GiskardExecutable.execute
# After editing source:
cram-knowledge --repository . --scope working-tree refresh
# Rebuild all analysis and republish for reconciliation:
cram-knowledge --repository . --scope working-tree reconcile
```

The overlay defaults to `repository_knowledge/.cache/working-tree.sqlite3` within the selected checkout. Its identity binds the current bytes, actual HEAD, Git index, baseline revision, checkout path and inclusion policy. A shared explicit cache path cannot mix committed and working-tree scopes.

New untracked files require explicit inclusion, for example `--include-untracked repository_knowledge/src --include-untracked repository_knowledge/pyproject.toml`. Repeat the same selections before the command when indexing, refreshing and querying. Select a directory to include future untracked files under it. Git-ignored files, generated mappings, symlinks/junctions and excluded directories remain outside ordinary content reads.

Working-tree mode represents current disk contents; it does not expose a separate staged-only view. The selected revision supplies the committed baseline. A branch switch, staging change or relevant file edit invalidates the old overlay. Queries check current source before and after selecting evidence and refuse stale results. Their snapshot identifies what was checked; an agent must refresh/recheck again after further edits.

Refresh reuses unchanged per-file analysis, then re-resolves all relationships against the current declaration inventory. Deleted or broken files cannot retain old definitions. Renames are treated as removal and addition, without guessed identity preservation. Unchanged refreshes skip analysis decoding and publication. Changed refreshes still resolve and publish the complete graph; their cost is not proportional only to changed lines.

Cache format 2 adds raw file analysis and working-tree identities. Run `index` to rebuild earlier format-1 caches. `refresh` and `reconcile` require an existing compatible cache. The committed database remains separate; the overlay stores the latest completed view rather than an archive of past editing states.

## Agent contract

Successful commands emit one JSON document using KRROOD's typed serialization. The result includes its snapshot identity, source scope, exact source locations, source excerpts, diagnostics, result count and next page offset. The Python `RecordCodec` adapter restores the result dataclasses.

Start with `find-symbols`, then use a qualified identity with `describe-symbol`. Follow references or relationships only as needed, with explicit page and depth bounds. The maximum default result page is 100 records, traversal depth is 8, and each source excerpt contains at most 80 lines. Excerpts report omitted lines; pagination reports omitted records. A zero-match response is not proof that no runtime caller exists.

Relationships describe static lexical evidence. Unresolved imports, receiver dispatch, dynamic calls, star imports and document anchors may remain unknown. A resolved call target identifies a declaration; it does not prove that a runtime execution reaches it. Declared configuration/template files are inventoried as source evidence; their deeper semantics are deferred.

`check-freshness` returns exit code 2 when the selected commit or working snapshot differs. Navigation refuses that mismatch with exit code 1. Other invalid requests or unavailable indexes fail with a diagnostic on stderr. Rebuild an incompatible cache with `index`; there is no migration requirement for disposable indexes.

## Curated wiki

Phase 5 adds eight source-backed draft concepts under [wiki/index.md](wiki/index.md), with separate format, repository-policy and source-evidence checks. Use `cram-wiki --repository . --revision main validate`, `find "execution"` or `show conventions/roles.md`. The source index must already exist and match the requested scope. These documents have no authenticated review.

`cram-wiki --repository . export C:/Temp/cram-wiki-export` writes a portable bundle into a new directory. Move proposals retain deprecated tombstones. See [phase5.md](docs/phase5.md) for the evidence schema, full command examples, interoperability assessment and limitations.

## Task context and changes

Phase 6 adds `find-examples`, `find-tests`, `prepare-context` and `assess-change` to `cram-knowledge`. For example:

```powershell
cram-knowledge --repository . --revision main prepare-context coraplex.robot_plans.motions.navigation.MoveMotion --profile code-extension --characters 30000
```

Results preserve source citations, fixture dependencies, uncertain candidates and omitted context. Character budgets apply to payload text, not tokens or the entire JSON response. Suggested test directories never imply complete coverage or executed tests. See [the phase-6 contract](docs/phase6.md) for profiles, explicit baseline comparisons, instruction scope and limitations.

## Tests

On Windows, with the focused environment populated:

```powershell
./repository_knowledge/scripts/run_tests.ps1
```

The runner checks for at least 8 GiB free memory and attaches pytest and its subprocesses to a Windows job capped at 4 GiB before permitting execution. It runs only this package with `--orm-build never` and avoids the robotics root fixtures.

CI includes a dedicated package matrix entry in the existing CRAM environment. It checks available memory, imposes an address-space limit and runs the same isolated pytest directory. Tests use local disposable Git repositories and the explicitly approved synthetic fixture identity; no credentials, model calls or network services are required.


## Validated publication

Phase 7 adds `cram-publication` (also `python -m repository_knowledge publication`) for admission, publication and authenticated artifact consumption. The default fork policy is automated: no other person's approval or pull request is required. Machine validation does not label prose human-reviewed. See [phase7.md](docs/phase7.md) for commands, trust boundaries, workflow activation and rollback. Hosted activation requires explicit immutable checker/runtime pins.

## Agent clients

Phase 8 adds the optional `cram-knowledge-mcp` stdio server, bounded basstler startup inspection, canonical context through the independently pinned consumer, and CRAMERA's shared architecture adapter. See [phase8.md](docs/phase8.md) for setup, query bounds, source-only fallback, opt-in queueing and validation. MCP and CRAMERA integration dependencies are optional; basstler does not import the robotics stack. Configuration examples are templates and are not installed automatically.

## Evaluation and recovery

Phase 9 adds reproducible source/structural/context comparisons, storage and client recovery tests, and optional Windows job memory receipts. See [the measured report](docs/phase9-evaluation.md), [operations guide](docs/operations.md) and [release checklist](docs/phase9-release.md). Rich context remains an explicit operation because measured CLI requests took 72–93 seconds. Local navigation correctness does not establish model task success or token savings; the team-release gate remains open.
