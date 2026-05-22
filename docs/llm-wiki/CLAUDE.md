# CLAUDE.md — LLM Wiki schema for cognitive_robot_abstract_machine

This file is the **schema and rule set** for an LLM Wiki (Karpathy pattern) covering the
three primary packages in this monorepo: `pycram`, `semantic_digital_twin` (SDT), and
`giskardpy`. Any agent ingesting, querying, or maintaining the wiki must read this file
first and follow the rules below.

The wiki is a *persistent, compounding artifact*: every ingest **updates existing pages**
in preference to creating new ones. Coverage grows when a concept earns a page, not
upfront per-file.

---

## 1. Vault layout

```
doc/llm-wiki/
  CLAUDE.md            # this file — schema and rules
  index.md             # catalog of every wiki page (one line each)
  log.md               # append-only chronological ingest/lint log
  raw/
    snapshots/         # commit-pinned code excerpts the wiki cites
    notes/             # hand-written design notes fed in as input
    pr-digests/        # extracted PR descriptions + diffs that drove updates
  wiki/
    packages/          # one page per top-level package (pycram, sdt, giskardpy)
    concepts/          # cross-cutting ideas (Designator, World, MotionStatechart, ...)
    entities/          # one page per "thing": class, key module, important function
    bridges/           # interfaces between the three packages (pycram<->sdt, ...)
```

**Source code is NOT mirrored under `raw/`.** Code lives where it lives. `raw/` only
holds excerpts the wiki actually cites (with commit SHA), design notes, and PR digests.

---

## 2. What counts as a "source" for ingest

An ingest event is the agent reading a bounded input and updating the wiki accordingly.
Valid sources:

- A **subpackage** (e.g. `pycram.designators`, `sdt.world_description`).
- A **single module** when narrow enough to fit one page touch.
- A **PR diff** — what changed and why, captured in `raw/pr-digests/`.
- A **design note** the user wrote and dropped in `raw/notes/`.
- A **user question** that revealed a gap. The answer-finding *is* the ingest.

Each ingest must produce a `log.md` entry (see §7).

---

## 3. Page schema (frontmatter — mandatory)

Every page in `wiki/` starts with YAML frontmatter:

```yaml
---
id: <fully-qualified.id>         # globally unique; see §4
kind: package | concept | entity | bridge
package: pycram | sdt | giskardpy | cross
source_paths:                    # optional, only if the page cites code
  - path: pycram/src/pycram/plans/designator.py
    lines: [19, 82]
    commit: <40-char SHA>
uses:                            # IDs this page depends on
  - pycram.plans.PlanNode
used_by:                         # IDs that depend on this page (inverted)
  - pycram.plans.factories.make_node
values:                          # optional — enum entity pages only
  - VALUE1
  - VALUE2
fields:                          # optional — action/motion entity pages
  field_name:                    #   flat key = single-class page
    type: <wiki-id-or-primitive> #   wiki page ID, or bool/float/str
    domain: [VAL1, VAL2]         #   optional; subset of valid enum values
    default: <value>             #   optional; value when argument is omitted
    derived_from: <path.expr>    #   optional; how this field is auto-filled
    description: one-line gloss  #   optional
  ClassName:                     #   nested key = bundled page (no "type:" here)
    field_name:
      type: <wiki-id-or-primitive>
status: stable | experimental | deprecated | stub
tags: [designator, plan, action, motion]
last_ingest: 2026-05-17
---
```

Rules:
- `id` is fully qualified: `<package>.<dotted-path>.<Name>`. Example: `pycram.plans.Designator`.
- `uses` and `used_by` are *page IDs*, not free-text. The lint pass enforces symmetry:
  if A.uses contains B, B.used_by must contain A.
- `source_paths` is the provenance contract. The lint pass flags pages whose `commit`
  is no longer reachable on `main`.
- `kind: stub` (or `status: stub`) marks pages that exist as anchors for backlinks but
  have not yet been written out — fine, intentional, just track them.
- `values:` is a flat list of valid enum member names. Present only on enum entity pages
  (where the Python class subclasses `Enum` or `IntEnum`). Enables an agent to enumerate
  choices from frontmatter without reading source code. When a field on another page
  restricts its enum via `domain:`, the allowed values must be a subset of the target
  enum's `values:` list.
- `fields:` is a typed field map for action and motion entity pages. **Single-class pages**
  use a flat map where each top-level key is a field name. **Bundled pages** (covering
  multiple classes) use a nested map where the top-level key is the class name and its
  value is a flat field map — distinguished from field entries by the absence of a `type:`
  sub-key. Field entries support: `type` (wiki ID or primitive `bool`/`float`/`str`),
  `domain` (subset of the enum's `values:` list), `default` (value when the argument is
  omitted from the constructor), `derived_from` (path expression for auto-filled fields,
  e.g. `plan_history.PickUpAction.grasp_description` or `ActionConfig.navigate_keep_joint_states`),
  and `description` (one-line gloss). Abstract base classes (`ActionDescription`,
  `BaseMotion`) do not carry `fields:` — their subclasses carry all concrete definitions.

---

## 4. ID and link conventions

- IDs use full dotted Python paths with class names where applicable:
  - Package: `pycram`, `sdt`, `giskardpy`
  - Concept: `concept.designator`, `concept.motion-statechart`
  - Bridge: `bridge.pycram-sdt`, `bridge.pycram-giskardpy`
  - Entity: `pycram.plans.Designator`, `sdt.world.World`, `giskardpy.qp.QPController`
- **Filename rule (deterministic path resolution):** the file for an ID lives at
  `wiki/<kind-plural>/<id>.md`. Folders are always plural: `packages/`, `concepts/`,
  `entities/`, `bridges/`. No subdirectories inside — files are flat per kind.
  Example: `pycram.plans.Designator` → `wiki/entities/pycram.plans.Designator.md`.
- Internal links use Obsidian wiki-link syntax with the **ID** as target:
  - `[[pycram.plans.Designator]]` — preferred
  - `[[concept.designator|Designator]]` — with display alias
- Cross-package references always use fully qualified IDs (no collisions).
- An agent can resolve any `[[id]]` to a file path without searching: try each of
  `wiki/{packages,concepts,entities,bridges}/<id>.md` — exactly one will exist (or
  zero, which is a lint failure per §9).
- **Per-package entity indexes (schema v3):** entity listings are maintained in
  `index-pycram.md`, `index-sdt.md`, `index-giskardpy.md` at the wiki root.
  `index.md` carries only Packages/Concepts/Bridges and pointers to these files.
  When adding or updating an entity, update the matching per-package index file
  (and `index.md` only if the kind is package, concept, or bridge).

---

## 5. Body sections (recommended order)

Every page body follows this order. Sections may be omitted when empty; do not invent
content to fill a section.

1. **One-line summary** — italicized, single sentence, what this is.
2. **Purpose** — 2-5 sentences: why this exists in the codebase, what problem it solves.
3. **When to use** — bullets. When to reach for this; when *not* to.
4. **Construction / dependencies** — minimal example or required collaborators.
5. **Key attributes / parameters** — table when the page is an entity.
6. **Subclasses / implementations** — for base classes.
7. **Related** — explicit `[[wiki-link]]` list, grouped by relationship (uses, used by,
   sibling concepts).
8. **Open questions** — anything contradictory, unverified, or noted for follow-up.
   Never silently resolve contradictions; log them here.
9. **Provenance** — bullet list of `source_paths` with brief description of what each
   citation supports.

**Size budget.** Entity pages stay ≤ 200 lines; concept pages ≤ 300; package and
bridge pages ≤ 150 (they're indexes, not deep dives). If a page outgrows the budget,
split it: promote a subsection to its own entity page and replace it with a
`[[link]]`. Stubs (§14) are exempt and typically ≤ 30 lines.

---

## 6. Ingest workflow

When asked to ingest a source X:

1. **Identify scope.** What package(s), what concept(s) does X touch?
2. **Search the existing wiki.** Read `index.md` and any page IDs that overlap.
   Default action is to **update** those pages, not create new ones.
3. **Read the code or note.** Ground every claim in `source_paths` with commit SHA.
4. **Decide on new pages.** Create a new page only if a concept has no existing home.
   Even then, link it from at least one existing page in the same ingest.
5. **Maintain symmetry.** When updating `uses` on one page, update `used_by` on the
   target. If the target page does not exist yet, create a stub.
6. **Update `index.md`.** One line per page touched (added or modified).
7. **Append to `log.md`.** Single entry summarizing the ingest (see §7).
8. **Aim for ~5-15 pages touched per ingest.** Significantly fewer = ingest probably
   too narrow; significantly more = probably doing too much in one pass.
9. **Final check (no dangling links).** Before finishing, scan every `[[id]]` link
   in the pages you touched and verify each target file exists (§4 path rule). For
   each missing target, **create a stub** per §13 in the same ingest. Add the stub
   to `index.md` and the `created:` line of the `log.md` entry.

---

## 7. `log.md` entry format

Append-only. Newest entries at the bottom. One block per ingest:

```markdown
## [YYYY-MM-DD] ingest | <source identifier>
- **scope:** what was read (path, PR, note title)
- **created:** [[id1]], [[id2]]
- **updated:** [[id3]], [[id4]]
- **findings:** 1-3 bullets on surprising facts or design decisions worth preserving
- **open questions:** any contradictions or follow-ups added to pages
```

For lint passes, use `lint` instead of `ingest`. For demand-driven Q&A ingests, use
`query`.

---

## 8. Update-vs-create decision rule

Create a new page only if **all** are true:
- The concept doesn't fit under any existing page (not even as a subsection).
- It has or will have at least one `used_by` (no orphans).
- It is non-trivial enough to warrant its own entry — single-line facts go inside an
  existing page, not a new one.

When in doubt, prefer subsections inside an existing page; promote to its own page
only when in-degree justifies it (≥3 backlinks from independent ingests).

---

## 9. Linting rules

A lint pass (`log.md` entry kind = `lint`) checks:

1. **Symmetry.** Every `uses: B` on page A implies `used_by: A` on page B.
2. **No orphans.** Every page is reachable from `index.md`.
3. **Stub coverage.** Stubs older than 30 days are listed in the lint log for
   prioritization.
4. **Stale provenance.** `source_paths.commit` SHAs are checked against `git
   rev-list --all`; missing SHAs are flagged.
5. **ID hygiene.** Filename matches `id` field exactly.
6. **Contradictions.** Pages with non-empty `Open questions` sections are listed.

Lint is **report-only** by default. Fixes happen in subsequent ingests.

---

## 10. Anti-patterns to avoid

- **Pre-generating one page per source file.** That recreates the codebase; it does not
  add knowledge. Pages must justify themselves with concept-level content.
- **Auto-summarizing docstrings as page bodies.** Docstrings are already there — the
  wiki captures *why* and *when to use*, which docstrings rarely cover.
- **Silently rewriting contradictions.** If two sources disagree, both go in
  `Open questions` with a note.
- **Backfilling provenance later.** A claim without `source_paths` is a claim the lint
  pass cannot verify. Cite as you write.
- **Dropping content from `log.md`.** It is append-only; corrections are new entries.

---

## 11. Bootstrapping order (recommended)

1. Three package overview ingests: `pycram`, `sdt`, `giskardpy`.
2. ~7 concept ingests: Designator, Plan/Language, World, MotionStatechart, QPSolver,
   CollisionChecking, SemanticAnnotation. (Adjust as the repo reveals them.)
3. Three bridge ingests for pairs that actually share runtime interfaces.
4. Demand-driven from here on — every gap is an ingest.

---

## 12. Navigation / lookup workflow (how to *use* the wiki)

An agent answering a user question against this wiki should follow this recipe.
Do **not** read source code from the repo until the wiki has been exhausted; the
wiki's job is to be the first-class answer surface.

1. **Read `CLAUDE.md`** (this file) — for ID/path conventions and page schema.
2. **Read `index.md`** — for Packages, Concepts, Bridges, and pointers to the
   per-package entity indexes (`index-pycram.md`, `index-sdt.md`,
   `index-giskardpy.md`).  Read the per-package index that matches your question's
   package to discover entity IDs.
3. **Select a starting page** based on the user's question:
   - If the question is about a concept ("what is a designator?"), start at the
     matching `concept.*` page.
   - If it's about a specific class or function, start at the matching `entities/`
     page.
   - If it's about cross-package interaction, start at the matching `bridges/` page.
   - Otherwise, start at the relevant `packages/` page and follow its outbound
     links.
4. **Follow `[[id]]` links on demand.** Each link resolves to a file path via §4.
   Open a linked page only when the question pulls you toward it; do not eagerly
   fan out.
5. **Use frontmatter `uses` / `used_by` for neighborhood queries** ("what depends on
   X?", "what does X require?"). These fields are structured and reliable; do not
   reparse prose.
6. **Stop when the answer is reached.** Cite the page IDs you consulted in the
   response.
7. **Fall through to source code only if** the wiki is silent on the question or
   a page's `source_paths` is the most direct citation. When you do, treat the
   discovery as a candidate ingest: log the gap in `log.md` as a `query` entry so
   the next ingest can close it.

This workflow is also what makes the wiki self-bootstrapping: a new agent only needs
to be pointed at `doc/llm-wiki/` to operate competently.

---

## 13. Stub policy (no dangling links)

A **stub** is a placeholder page that exists so an `[[id]]` link from somewhere
else resolves to a real file. Stubs are normal and expected — they are how the
wiki defers depth without losing structure.

**Rule.** Every ID that appears in any `uses`, `used_by`, or `[[wiki-link]]` MUST
have a corresponding file. If a target doesn't exist when an ingest references it,
the same ingest creates a stub for it (logged in §6 step 9).

Minimal stub:

```markdown
---
id: pycram.exceptions.ContextIsUnavailable
kind: entity
package: pycram
source_paths:
  - path: pycram/src/pycram/exceptions.py
    commit: <SHA>
uses: []
used_by:
  - pycram.plans.Designator
status: stub
tags: [exception]
last_ingest: 2026-05-17
---

_Stub. Raised when a designator's plan/robot/world is accessed before the designator
is wrapped in a [[pycram.plans.DesignatorNode]]._

To be expanded on the next ingest that touches `pycram.exceptions`.
```

Stub body conventions:
- One italicized sentence prefixed with `_Stub._` stating purpose.
- One sentence on **when this will be expanded** (which ingest theme is likely to
  cover it).
- Do not invent content. If you don't know what it is, the stub body says so.

A stub still participates in `used_by` symmetry — when the stub is created, every
page that references it gets its `uses` updated in the same ingest.

When a stub is promoted to a full page, `status` changes to `stable` /
`experimental` / `deprecated`, and the body fills out per §5. No new file is
created — the stub IS the future page.

---

## 14. Versioning this schema

This `CLAUDE.md` is itself versioned by git. Significant changes are noted in `log.md`
as `## [YYYY-MM-DD] schema | summary`. Pages do **not** need to be re-ingested just
because the schema changed; they are migrated lazily on the next ingest that touches
them.
