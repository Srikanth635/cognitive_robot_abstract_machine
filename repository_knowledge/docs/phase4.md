# Phase 4: working-tree source

The committed cache remains distinct from a per-worktree overlay. The overlay represents current file contents, including staged additions and unstaged modifications/deletions. It does not provide a separate index-only view. Untracked source requires explicit repository-relative file/directory selection; ignored files remain excluded.

Capture identifies the base revision, actual HEAD, Git index content, canonical checkout identity, selected file bytes and inclusion policy. Files are read into memory before parsing. Repeated capture must agree; concurrent mutations abort rather than publish a mixed snapshot. Generated mappings, symlinks, submodules and excluded directories remain outside content reads.

Refresh retains raw per-file extraction alongside resolved graph evidence. Identical file contents and import context reuse that extraction. All references are then resolved against the current declaration/export inventory. This conservative invalidation covers callers in unchanged files, source-root changes and unresolved dependencies. Renames remove old paths and introduce new paths without guessing semantic identity.

Publication remains transactional. Committed and working-tree scopes cannot replace one another at a database path. Queries capture one saved graph, compare the current source fingerprint before navigation, and recheck working-tree state before returning evidence. Parse failures replace old facts with diagnostics. Failed capture or publication cannot make historical evidence appear current.

Mutation tests compare refresh with a clean rebuild for changes, additions, deletion, rename, branch change, import/export changes, source-root changes and parser failure. Reconciliation is an explicit clean rebuild. No-change refresh must preserve semantic identity and avoid publication.

The first implementation makes parsing incremental and re-resolves/publishes the graph conservatively. It does not claim that database publication or reference resolution is proportional to the Git diff. Curated-concept invalidation follows when concepts exist in phase 5.
