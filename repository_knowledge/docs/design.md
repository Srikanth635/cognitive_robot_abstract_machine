CRAM codebase knowledge system — repository assessment and proposed design

Assessment date: 19 October 2026.
Inspected revision: 7b27424de7768798dc561f447e11c2a8eb86c307.

Local main, origin/main, the fork's advertised main, and upstream cram2/main all resolved to that revision during inspection. Findings below come from committed Git objects, not the working tree. The checkout currently reports 2,992 staged deletions alongside untracked replacement paths; no attempt was made to alter or explain that state. This document is a proposal, not an implemented or benchmarked system.

**Recommendation**

Build a Python-aware source index plus a curated OKF knowledge bundle, exposed through bounded query operations. Reuse the repository's extraction, graph and workflow facilities where their contracts fit. Use the source index for structural facts and the OKF documents for explanations, conventions, and workflows.

The main value for CRAM is connecting symbols across packages to their tests, fixtures, examples, documentation and development rules. Token reduction is a possible measured consequence, not the only acceptance criterion.

**What main already provides**

The committed tree contains 2,992 files, 1,316 Python files beneath real package src directories, 924 Python files beneath test (including fixtures and datasets), and 206 Markdown/RST files. There are 11 UV workspace members, plus basstler as a separately configured developer-tooling project. These are path-based inventory counts, not counts of tested behaviours.

| Existing component | Evidence | Reuse and boundary |
| --- | --- | --- |
| Workspace declarations | [Root project configuration](https://github.com/Srikanth635/cognitive_robot_abstract_machine/blob/7b27424de7768798dc561f447e11c2a8eb86c307/pyproject.toml) | Discover distribution names, source roots, Python version and declared dependencies. Distinguish declared dependencies from observed imports. |
| Static architecture scanner | [ArchitectureScanner](https://github.com/Srikanth635/cognitive_robot_abstract_machine/blob/7b27424de7768798dc561f447e11c2a8eb86c307/cramera/src/cramera/knowledge/architecture_scan.py#L87) | Already parses Python without importing the scanned modules, collects classes and package imports, and excludes generated ORM interfaces. Extend or extract shared behaviour instead of independently duplicating it. |
| Architecture entity model | [PythonClass](https://github.com/Srikanth635/cognitive_robot_abstract_machine/blob/7b27424de7768798dc561f447e11c2a8eb86c307/cramera/src/cramera/knowledge/architecture_entities.py#L62) | Existing classes have module, base-name and method-count information. Agent navigation additionally needs canonical import paths, nested qualified names, signatures, source spans and evidence. |
| Source extraction | [Source extraction helpers](https://github.com/Srikanth635/cognitive_robot_abstract_machine/blob/7b27424de7768798dc561f447e11c2a8eb86c307/krrood/src/krrood/code_generation/source_extraction_utils.py#L33) | Existing definition spans and import extraction merit reuse. Their current APIs do not constitute a complete symbol-reference index. |
| Rich runtime relationships | [ClassDiagram](https://github.com/Srikanth635/cognitive_robot_abstract_machine/blob/7b27424de7768798dc561f447e11c2a8eb86c307/krrood/src/krrood/class_diagrams/class_diagram.py#L403), [SymbolGraph](https://github.com/Srikanth635/cognitive_robot_abstract_machine/blob/7b27424de7768798dc561f447e11c2a8eb86c307/krrood/src/krrood/symbol_graph/symbol_graph.py) | Can provide optional registered-type, role and instance evidence. Runtime object graphs and static source graphs have different identities and completeness. |
| Agent workflow tooling | [basstler](https://github.com/Srikanth635/cognitive_robot_abstract_machine/blob/7b27424de7768798dc561f447e11c2a8eb86c307/basstler/README.md), [session hook configuration](https://github.com/Srikanth635/cognitive_robot_abstract_machine/blob/7b27424de7768798dc561f447e11c2a8eb86c307/.claude/settings.json) | Suitable for orchestrating freshness checks and reporting the knowledge entry point. Keep its independent environment boundary explicit. |
| Query and visualization UI | [CRAMERA](https://github.com/Srikanth635/cognitive_robot_abstract_machine/blob/7b27424de7768798dc561f447e11c2a8eb86c307/cramera/README.md) | Useful future human-facing consumer. Its existing query runner executes trusted Python; it is not a ready-made restricted agent query endpoint. |
| Quality and domain rules | [AGENTS.md](https://github.com/Srikanth635/cognitive_robot_abstract_machine/blob/7b27424de7768798dc561f447e11c2a8eb86c307/AGENTS.md), [spatial style guide](https://github.com/Srikanth635/cognitive_robot_abstract_machine/blob/7b27424de7768798dc561f447e11c2a8eb86c307/semantic_digital_twin/doc/style_guide.md) | Retrieve applicable rules with code-generation context, without replacing or rewriting their authority. |
| Tests and fixtures | [Shared fixtures](https://github.com/Srikanth635/cognitive_robot_abstract_machine/blob/7b27424de7768798dc561f447e11c2a8eb86c307/test/conftest.py), [motion-chart tests](https://github.com/Srikanth635/cognitive_robot_abstract_machine/blob/7b27424de7768798dc561f447e11c2a8eb86c307/test/coraplex_test/test_plan/test_motion_state_chart_building.py) | Support examples, fixture selection and conservative test recommendations. A static test reference does not prove coverage. |
| Ownership and CI | [CODEOWNERS](https://github.com/Srikanth635/cognitive_robot_abstract_machine/blob/7b27424de7768798dc561f447e11c2a8eb86c307/.github/CODEOWNERS), [CI](https://github.com/Srikanth635/cognitive_robot_abstract_machine/blob/7b27424de7768798dc561f447e11c2a8eb86c307/.github/workflows/ci.yml) | Reuse package ownership and package-level validation. Ownership currently covers only some packages. Branch-protection settings were not audited. |

Two concrete limitations matter before using the existing scanner as editing context. Its cache acceptance checks root path, cache version and cached contents, but does not compare source commits or content hashes. Its architecture view resolves base classes using short names, preferring the same package. This can reuse old information or ambiguously resolve an inheritance edge. These are source-inspection findings, not test-confirmed regressions.

**Proposed responsibilities**

1. Repository inventory discovers committed source files, manifests, tests, docs, templates and declarative configuration. It explicitly reports excluded, unsupported and unparseable files. Generated ormatic_interface.py files, virtual environments, downloaded datasets, meshes, recordings, vendored libraries and personal state are excluded from ordinary source indexing. Record generator relationships from handwritten model and generator code instead.

2. Deterministic extraction produces modules, classes, methods, functions, fields, enums, import bindings, signatures, source spans and resolvable references. Parse with the project's configured Python interpreter. Do not import every scanned package or run demos to build the index. Preserve generic base expressions, aliases, re-exports, nested definitions and type-only imports. Recognize unresolved dynamic dispatch rather than inventing targets.

3. Relationship storage keeps typed edges with evidence. Start with SQLite tables for entities, references, sources and invalidation; use full-text search if supported by the configured runtime. Existing rustworkx facilities can support richer in-process traversal when appropriate. A dedicated graph database is not required by the initial workload; measure before adopting one.

4. OKF enrichment produces a small set of linked package, subsystem, workflow, convention and extension-point documents. Every factual passage links to relevant source or documentation. Structure can be generated deterministically; synthesis stays draft until its required verification is satisfied. Avoid creating and maintaining one prose page for every method.

5. Query operations return bounded context with source revision, file locations, evidence, unresolved relationships and omitted-result counts. The same service can support a CLI and, later, an MCP adapter and CRAMERA views.

6. Validation and publication separately enforce source consistency, organisational policy and authenticated review. Parsing frontmatter is not authentication. Ordinary extracted facts can be machine-checked; architectural rationale requires an appropriate owner.

**Placement and reuse**

A focused repository-knowledge package is the preferred long-term boundary. Keep source extraction, the entity model, storage and queries together; put CRAM-specific enrichment in explicit adapters. Basstler should orchestrate refreshes and session integration, and CRAMERA should consume the resulting graph rather than own the indexing service.

Reuse krrood's source extraction, serializers and exception conventions where compatible. Do not introduce imports from krrood back into the higher-level workspace packages. Optional runtime enrichment may depend on the robotics stack in its existing environment; static indexing must not initialize worlds, ROS or simulators.

Before relocating an existing public method or renaming its API, obtain the permission required by AGENTS.md. New dependency declarations also require approval before implementation. No packages or dependencies were added during this assessment. Do not assume a lightweight basstler environment already contains krrood or the full CRAM stack.

**Knowledge to model**

| Record | Useful information |
| --- | --- |
| Repository revision | Repository identity, source commit, tree/content digest, extractor version, policy version, configuration digest and extraction coverage. |
| Symbol | Qualified import identity, kind, declaration path/span, signature, fields, docstring and owning package. A locator includes its revision. |
| Relationship | Source, target, kind, source span, evidence category and resolution status. |
| Test or fixture | Test location, referenced symbols, fixture dependencies, fixture visibility/scope and available execution prerequisites. |
| Concept | Stable OKF path, subject, supporting source identities/digests, generation details and lifecycle. |
| Verification | Authenticated actor, checked claim or artifact digest, source revision, check result and review evidence. |

Initial edge kinds: defines, imports, inherits, references, annotates, documents, uses_fixture, generates, depends_on and supersedes. Add calls only where the target is resolved. Keep observed runtime edges and inferred semantic links distinguishable from static evidence.

Preserve logical concept identity across a confirmed source rename where appropriate. Source-symbol names can change independently. If equivalence is unclear, expose the ambiguity and require reconciliation; do not silently transfer approvals.

For freshness beyond the standard date field, use explicitly documented organisational extensions such as source_commit, source_paths and source_digest, or a publication manifest. Keep these separate from official OKF keys.

**Agent operations and practical tasks**

The following are proposed contracts, not existing commands.

| Operation | Result | Example CRAM task |
| --- | --- | --- |
| find_symbols | Ranked exact, qualified and text matches with source locations. | Locate the current motion-execution entry point. |
| describe_symbol | Signature, fields, bases, relevant source and linked concepts. | Understand GiskardExecutable before changing execution behaviour. |
| find_references | Resolved references, unresolved candidates and reference kinds. | Find consumers of a changed method or spatial type. |
| find_relationships | Bounded typed traversal with supporting source spans. | Trace plan construction into motion statecharts. |
| find_examples | Current tests and examples using the relevant API. | Follow an existing action or serializer pattern. |
| find_tests | Candidate tests plus required fixtures and environment constraints. | Choose relevant package-level checks for a World change. |
| assess_change | Changed public contracts, dependent symbols, stale concepts and candidate tests. | Review a generic-field or role-registration change. |
| prepare_context | A task-specific selection of rules, definitions, examples, tests and uncertainties. | Generate a new action using current CRAM conventions. |
| check_freshness | Revision compatibility, outdated sources and extraction gaps. | Reject a cached answer from another worktree. |

For code generation, prepare_context should gather the relevant interface, one suitable current implementation, tests and fixtures, applicable AGENTS rules and domain conventions. The agent then checks exact source before editing, adds a failing test, changes the code and runs the required checks. Knowledge retrieval guides the work; it does not establish that generated code is correct.

Further applications include onboarding, reviewer routing, public-API migration planning, documentation drift detection, fixture discovery, duplicate-implementation discovery and identifying dependency-policy violations. Later, read-only runtime snapshots can connect a recorded failure to its implementation revision. Do not mix a live robot's state with source-architecture facts.

**Concrete pilot**

Index structural information across all packages, but begin curated enrichment in four connected areas:

- coraplex plans, actions, motion-chart construction and execution environments.
- giskardpy motion statecharts and the interfaces consumed by coraplex.
- semantic_digital_twin worlds, spatial types and the spatial naming guide.
- krrood generics, roles, serializers and ORM generation conventions.

Useful first concepts include the plan-to-motion path, simulated versus real execution, adding an action, choosing a world fixture, spatial transform conventions, extending generic dataclasses, and the handwritten-model-to-generated-ORM workflow.

A concrete cross-package query is: "What needs review if GiskardExecutable.execute changes?" The answer should point to its declaration, chart construction, execution-mode selection, relevant tests, and unresolved dynamic users. Another is: "How do I add a role without violating krrood isolation?" The answer should include the role implementation, existing isolated mimics, registration tests and the applicable rule. Neither query should require loading every package into the agent's context.

**Freshness across branches and worktrees**

Use three distinct states:

- Published baseline: immutable knowledge associated with an exact source revision and publication evidence.
- Working-tree overlay: added, edited and deleted files for this checkout. Compute content identities, not only branch names or timestamps. Uncommitted changes must never inherit human verification automatically.
- Session observations: temporary findings that may become draft concepts, but cannot silently modify the published baseline.

On source changes, replace the changed files' structural records and invalidate affected links and concepts through reverse dependencies. Changes to manifests, import bindings, bases, generators, fixtures or policy may require a broader rescan. A scheduled or explicit full rebuild must reconcile the incremental result.

Atomically publish a completed index. Query readers retain a coherent previous snapshot or receive an explicit unavailable/partial result; they must not see half of an update. A parse failure invalidates the old facts for that file and is visible in extraction coverage.

Revalidate selected source hashes before returning editing context. A concept can remain valid after an unrelated commit if its complete evidence set and policy still match. Whole-repository commit equality alone is unnecessarily restrictive, while matching a date alone is too weak.

**Git and publication workflow**

Use an explicit refresh command and a session freshness check for the first pilot. A commit hook can queue or perform bounded deterministic refreshes, but should not synchronously invoke an LLM and write a second commit on every developer commit.

CI on pull requests validates source links, extraction completeness, changes in public contracts, organisational policy and affected concept freshness. Include source, docs, extractor, workflow and policy changes in triggers. A trusted post-merge publisher can produce the canonical artifact tied to the merged source revision.

Store curated concepts, concept policy and index entry points in Git. Keep large rebuildable indexes and local overlays outside tracked source. Publication manifests associate the source commit, concept-tree digest, validation results and authenticated approvals, avoiding self-referential "commit contains its own hash" schemes.

Protect policy, workflow and ownership definitions along with knowledge. Read current authenticated approval state rather than producer-authored verified fields. Bind approval to the reviewed content and source revision. Local drafts may still support navigation when clearly labelled; stale or unsupported claims cannot determine a patch without checking source.

A CLI/MCP query interface should offer structured operations, not expose the existing EQL runner's arbitrary Python execution as its default query surface. Retrieved prose is evidence, not a source of new authority over the agent.

**Current OKF tooling assessment**

The [official specification](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md) currently identifies itself as v0.2. It defines portable concepts and optional provenance/lifecycle signals; repository-specific acceptance policy remains separate.

The [current CLI README](https://github.com/superops-team/okf) now documents MCP support and v0.2 handling, so several details in the articles have aged. Evaluate its bundle/search/serving capabilities rather than rebuilding them by assumption.

Its [Git scanner source](https://github.com/superops-team/okf/blob/main/pkg/git/git.go) uses regex extraction for Python and defaults to a 100 KiB file limit; the default include list also omits RST. At the inspected CRAM revision, world.py is 110,611 bytes and adapters/multi_sim.py is 151,525 bytes. Both exceed that limit. Several large tests do too. This is why default initialization is an inadequate completeness guarantee for CRAM.

Use the CLI as a candidate adapter after testing compatibility, exclusions, Python resolution and worktree freshness. It has not been installed or benchmarked here. Graphify was not evaluated in this assessment; there is no evidence yet that adopting it would outperform the existing CRAM building blocks.

**Delivery sequence and acceptance**

1. Establish an inventory and evaluation set. Capture known answers for navigation, code extension, test discovery and change-impact tasks at the inspected revision. Include same-named classes, generic bases, unresolved dynamic dispatch, RST documentation and files larger than 100 KiB.
2. Deliver deterministic symbol queries and a source-revision manifest. Reuse compatible extraction code, add meaningful failing tests before fixes, and report all unsupported/unparseable input.
3. Add a small OKF concept set and prepare_context. Separate copied structural facts from generated interpretation. Confirm that every citation resolves and every example belongs to the intended revision.
4. Add worktree overlays, reverse invalidation, deletion tombstones and atomic snapshots. Compare incremental output with a clean full rebuild after edits, additions, deletions, renames, generic-base changes and fixture changes.
5. Add CI policy and authenticated publication. Test forged verification metadata, changed policy, stale approvals and rollback. The trusted publisher must not execute untrusted proposal code with its credentials.
6. Add an MCP adapter and optional CRAMERA view after query contracts stabilise. Retain CLI/file access so agents are not dependent on one client.
7. Extend runtime evidence and richer retrieval only where evaluation identifies a gap.

Compare the same tasks at the same revisions using direct source search, structural queries, and structural queries plus curated concepts. Measure task correctness, required-source recall, invalid citations, stale answers, latency, tool calls, tokens and amortized update cost. Repeat after source mutations. Do not set a token-savings target from the articles' anecdotes.

Development validation must follow this repository's rules: configured interpreter, pytest, meaningful test-first coverage, separate package/directory invocations, free-memory checks and a memory cap, explicit ORM regeneration when required, --orm-build never for applicable test runs, isolated krrood mimics and the docstring formatter for modified Python. Existing basstler CI deliberately excludes the root conftest, so its option-registration contract needs to be reconciled with the required ORM flag before adding a new test job there.

No tests were run during this read-only architectural assessment. Branch protections, installed runtime availability, actual query quality and operational cost remain to be validated during implementation.

