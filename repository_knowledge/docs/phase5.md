# Phase 5: curated knowledge

The wiki is a separate layer above the structural source index. Its eight pilot concepts live in [wiki/](../wiki/index.md), under package, workflow and convention directories. The saved [plan](plan.md) remains the reference for later phases. The initial source revision and citation selectors are recorded in [pilot.toml](pilot.toml).

`concepts.py` contains plain concept, lineage and validation records. Application operations check evidence, propagate invalidation, search concepts and prepare lifecycle proposals. `adapters/okf.py` owns YAML, Markdown interpretation and directory exchange. The source cache remains independently replaceable. Static boundary and cycle checks cover the new modules.

## Three separate results

| Result | What it establishes |
| --- | --- |
| Format | Concepts have a parseable delimited metadata mapping and a nonempty string type; reserved indexes/logs follow the supported structural conventions. Unknown types, extra metadata, missing optional fields and broken links remain readable. |
| CRAM policy | Titles, summaries, supported pilot types, source lineage, attribution labels, explicit dependencies and tombstone replacements satisfy the local requirements. Declared review events are unauthenticated findings. |
| Source evidence | Recorded files, whole-file SHA-256 digests, inclusive line ranges and optional qualified symbols agree with one source snapshot under the reported content-comparison policy. Expiry and reverse dependencies are checked independently of lifecycle. |

`current` means matching structural evidence. It does not verify the meaning of every sentence or establish an architectural rationale. `authenticated_review` remains false in every phase-5 result, including documents declaring a human verifier or stable status. Authenticated review and promotion belong to phase 7.

The adapter accepts the official open-ended type vocabulary. Local pilot types are `Package`, `Workflow` and `Convention`. All need source evidence; packages additionally need a qualified declaration binding, and conventions need a document source. Workflows use source contracts, examples or fixture evidence appropriate to the explanation.

## Evidence representation

Official OKF metadata includes `type`, `title`, `description`, `status`, `generated`, `sources`, `verified` and `stale_after`. Review declarations are preserved as data. Bare `verified` mappings and lists are both readable.

The `cram` family is a repository extension, not part of the OKF specification:

```yaml
cram:
  source_commit: <immutable source commit>
  evidence:
    - source_id: action-base
      path: coraplex/src/coraplex/robot_plans/actions/base.py
      start_line: <inclusive first line>
      end_line: <inclusive last line>
      content_digest: <whole-file SHA-256>
      symbol: coraplex.robot_plans.actions.base.ActionDescription
  depends_on: [workflows/world-fixtures.md]
  replacement: ''
```

Evidence labels must appear in `sources[].id` and in per-claim body footnotes. The adapter preserves additional top-level metadata and additional keys inside `cram`; evidence entries use the documented schema. Safe YAML scalar/list/mapping values are supported, with date/time scalars normalized to ISO text. Cyclic aliases and excessive nesting are refused. YAML comments and layout are not preserved.

The original drafting commit is retained as declared lineage. A validation result separately identifies the selected source commit, scope, snapshot and check time. Matching file digests can remain current across unrelated commits. This checks current content consistency; it does not authenticate a producer's historical provenance or external source URL.

The default comparison recognizes uniform UTF-8 LF/CRLF differences. It reconstructs both byte representations of the captured normalized text and requires *both* the original and selected file digests to match that pair. It reports affected paths in `text_equivalent_paths` and the selected `content_comparison` policy. Unknown bytes, encoding-marker changes and actual text edits cannot pass this equivalence check. The Python operation also supports `ContentComparison.EXACT_BYTES`. Source-index freshness remains byte-exact: changing line endings still requires refreshing the working-tree index.

## Freshness and lifecycle

Changed or missing files, invalid source spans, missing/moved symbols and expiry invalidate evidence. Invalid timestamps make evidence unusable rather than silently disabling expiry. Missing evidence is unknown. Explicit `depends_on` paths and supported local body links propagate invalidation conservatively; cycles with unchanged evidence remain current. The complete small bundle is checked on each query, without persisting a separate concept cache.

Imported lifecycle values remain distinct from these results. New concepts default to draft. Producer operations can propose drafts or replaced tombstones, remove claimed review/generation events when changing lifecycle, and cannot promote a concept to stable. A move creates a new draft and retains the old path as a deprecated tombstone with a successor and historical evidence. The CLI adjusts supported body links for directory changes; unknown metadata and path-valued extension fields still require review.

Source removal does not delete wiki files. Validation reports stale evidence until an explicit correction or deprecation proposal is reviewed. No automatic rewrite of prose or refreshed digest can make an unchecked explanation trusted.

## Commands

Install the package with the configured interpreter to expose `cram-wiki`. It uses the same repository, database, revision, scope and explicit untracked-inclusion policy as `cram-knowledge`. Options precede the command.

```powershell
cram-wiki --repository . --revision main validate
cram-wiki --repository . --revision main find "execution"
cram-wiki --repository . --revision main show conventions/roles.md
cram-wiki --repository . --revision main export C:/Temp/cram-wiki-export
cram-wiki --repository . --revision main move conventions/roles.md conventions/role-pattern.md C:/Temp/cram-wiki-proposal
```

`--bundle` selects a different local bundle. Find/show results include concept documents and their separate validation, total matches and next offset. Search checks exact paths first, then lexical path/title/summary/body matches. Show requires the exact document path. Default pages contain at most 20 concepts, with a maximum request of 100. Character/token budgeting is phase 6.

Working-tree queries require a current overlay and repeat its `--include-untracked` selections. Queries check source freshness before processing and recheck working source afterwards. The database selection reads source/declaration evidence without loading all relationships or cached analyses.

Validation/search output is one typed KRROOD JSON document. Exit 2 indicates format/policy findings or non-current returned concepts; malformed documents, invalid requests and stale source indexes fail visibly. Export is format exchange, requires no index, and makes no validity or trust claim. Export/move write only to a new directory and preserve the existing bundle. Output is a proposal, not canonical publication.

## Interoperability and limits

The [official v0.2 specification](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md) was checked on 20 October 2026. The independent `superops-team/okf` CLI v0.4.0 was evaluated from its Windows amd64 release. Archive SHA-256: `888a9b863a37f193c0c104885abea62ff3ac5e00c3a266d64af3c2e54c75a0f1`, matching the release checksum list. It read/linted all eight concepts with zero errors and found the generic-dataclass concept through lexical search. Its mixed-case and verification suggestions are advisory. No hook, watcher, model provider or global installation was configured.

This is read interoperability, not proof that its generator/update commands preserve CRAM evidence or meet freshness guarantees. The third-party CLI is not a dependency. It offers additional MCP and semantic-search features, which would need the same source-identity and mutation evaluation before adoption.

Supported links follow the shared document analyzer's explicit Markdown-link subset. Reference-style links, HTML links, arbitrary anchors and formal relationship semantics are not fully analyzed. Reserved-document checks cover the documented basic index/log layout, not all Markdown constructs. Malformed YAML or malformed CRAM extension shapes abort import with a visible error. They are not silently admitted or published.

KRROOD remains unchanged. The JSON boundary delegates owned fields to its dataclass serializer; explicit typed wrappers also carry nested dictionaries, avoiding runtime serializer discovery. Only known evidence types may be instantiated when reading typed results. PyYAML is the only newly declared phase-5 dependency, approved by the user and already used elsewhere in CRAM.

Phase 6 will combine these explanations with relevant interfaces, implementations, instructions, examples and tests into bounded task context. Phase 7 will establish independent review and publication authority.
