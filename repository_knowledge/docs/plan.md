CRAM repository knowledge system — phased build plan

Prepared: 19 October 2026.
Design baseline: main at 7b27424de7768798dc561f447e11c2a8eb86c307.
Status: saved implementation plan. See [delivery progress](progress.md) for completed work, validation and remaining gates.

This plan turns the [repository assessment](design.md) into an ordered build. It targets useful agent capabilities while keeping source analysis, knowledge curation, publication authority and user interfaces independently changeable.

**1. Intended result**

Agents can locate code, inspect contracts, find relevant examples and fixtures, prepare a change, assess likely effects, and retrieve architectural explanations. Every answer identifies its source snapshot, evidence and limitations. Changes in the repository invalidate affected knowledge. Published concepts carry independently established verification when required by policy.

The system supplies context and evidence. Code generation remains the coding agent's responsibility, followed by CRAM's test-first development and validation requirements.

The first useful release is a committed-source query tool. Working-tree use arrives only after freshness and overlay tests pass. Automatic publication is a later capability, not a prerequisite for useful navigation. The query service initially means local application operations exposed by a CLI; it does not require a continuously running server.

**2. Stable dependency rules**

Stable dependencies concern which modules must change when another module changes. They are not merely pinned package versions. Put shared meaning and contracts near the centre; put Git, database, model-provider, client and hosting details outside it.

| Layer | Responsibility | Permitted project dependencies | Must not depend on |
| --- | --- | --- | --- |
| Domain model | Symbols, source locations, snapshots, relationships, concepts, evidence and policy values | Standard library; narrow KRROOD foundation where repository rules require it, such as DataclassException | Git commands, SQLite, YAML/OKF serialization, LLM providers, MCP, CRAMERA entities, ROS |
| Contracts | Small abstract interfaces required by implemented operations | Domain model | Concrete adapters or client SDKs |
| Operations | Indexing, queries, invalidation, context preparation, verification and publication decisions | Domain model and contracts | Adapter implementations, transport payloads, provider-specific response objects |
| Adapters | Git/filesystem reads, Python parsing, SQLite persistence, OKF encoding, authenticated reviews, optional generation/runtime evidence | Contracts and domain model; explicitly approved dependencies | Client UI internals or other adapters' private implementation |
| Interfaces | CLI, MCP and human-facing integrations | Operations and their request/result dataclasses | Direct database access or duplicated business rules |
| Composition | Instantiate concrete implementations and supply them to operations | All layers needed by that executable | Being imported by lower layers |

The domain model uses plain dataclasses, not CRAMERA NamedEntity or KRROOD Symbol subclasses. Those existing entities participate in runtime registration; the source index must not acquire that behaviour merely by representing a class.

KRROOD's current JSON serializer has NumPy, SciPy and additional imports. Reuse it at the owned-JSON boundary as required by AGENTS.md, rather than making every domain object inherit from it. Keep the associated dependencies and startup costs explicit. Do not describe the whole application as dependency-free.

Use explicit abstract superclasses or mixins, not Protocol. Introduce a contract when a concrete operation first needs it; do not create speculative interfaces for every possible backend. Separate reading from writing where authority or client needs differ.

Enforce the dependency direction with a static import-boundary check that includes type-only imports. Reject cycles and forbidden cross-layer imports. An outer composition entry point is the deliberate place that knows concrete classes.

For external tools, pin the version actually evaluated, document its contract, and add it only after the required dependency approval. Keep the source index usable when optional search, generation, MCP or viewer integrations are absent.

**3. Proposed package boundaries**

Create a focused workspace package with the proposed import name repository_knowledge. This is a proposed name, not an existing package. Package registration and direct dependency declarations are Phase 0 decisions.

Its source is divided by responsibility:

- model: domain dataclasses, enums and invariants.
- contracts: small source, persistence and evidence interfaces introduced as needed.
- operations: index construction, queries, freshness, context and publication.
- adapters: Git/filesystem, Python analysis, SQLite, OKF, reviews, and optional generation/runtime integrations.
- interfaces: CLI and later MCP entry points.
- composition: explicit object construction.
- exceptions.py files near the code that raises their exceptions.

Keep curated knowledge under repository_knowledge/wiki/, with its entry index and package, workflow and convention subdirectories. Keep development plans and design records in repository_knowledge/docs/. Keep extraction caches, databases and worktree overlays in repository_knowledge/.cache/, ignored per worktree. Publish immutable bundles as artifacts associated with exact source revisions.

Integration boundaries:

- KRROOD remains lower-level. Reuse its extraction and serialization facilities where their contracts fit; never make KRROOD import repository_knowledge or other prohibited workspace packages.
- CRAMERA becomes a consumer. Adapt source-index results into its existing view entities at the integration boundary.
- Basstler remains an orchestrator in its own environment. Invoke the service through a versioned process interface using the configured CRAM interpreter; do not quietly install the robotics stack into basstler.
- Runtime adapters may depend on CRAM runtime packages, but they are separate optional entry points. Static indexing does not initialize ROS, robot worlds or simulations.

Where extraction behaviour is already implemented, reuse or carefully refactor it rather than maintaining a second implementation. Reconcile CRAMERA's existing scanner during integration. Any public method rename requires permission before the change. Compatibility adapters must perform a real translation or implement an abstract contract, not become an unexplained chain of forwarding methods.

**4. Delivery dependencies**

The core path is:

Phase 0 -> Phase 1 -> Phase 2 -> Phase 3 -> Phase 4 -> Phase 5 -> Phase 6.

Phase 7 requires Phases 4 and 5, including the policy contracts introduced when knowledge validation first needs them.
Phase 8 requires the query/context contracts from Phases 3 and 6; access to canonical publication also requires Phase 7.
Phase 9 validates the enabled features together.
Phase 10 contains optional extensions and does not delay the core release.

These are technical prerequisites, not instructions to create parallel agents. Each phase should be delivered in small reviewable changes, with passing prerequisites retained.

**Phase 0 — Establish the implementation baseline**

Purpose: settle package, environment and reuse choices before building a new dependency graph.

Work:

- Record the baseline source revision, package layout, rules and existing extraction contracts.
- Resolve development-checkout readiness before editing. The inspected checkout reported staged deletions across tracked files; do not reset or restage them automatically. If still necessary, use an isolated checkout based on the agreed main revision.
- Identify the configured Python interpreter and supported package-test commands. Record how to check available memory and impose a process-memory cap on the actual development and CI operating systems.
- Audit direct and transitive dependencies for source extraction, exceptions, serializers and candidate OKF tooling.
- Decide the minimal new package declarations. Prepare the exact changes for approval where AGENTS.md requires it; no blanket approval is needed for this plan.
- Decide whether early tests use the existing root harness. A separate lightweight harness must explicitly support --orm-build never without duplicate option registration or silently importing the robotics test setup.
- Define a fixed evaluation set for navigation, examples, code-extension context, impact analysis and stale answers. Use real source revisions and repository-local mimic datasets.
- Record initial measurements for direct source search. Keep the tasks and answer evidence fixed for later comparisons.

Deliverables: architecture decisions, approved dependency proposal when applicable, evaluation fixtures, baseline measurements and a documented test runner.

Exit gate: the interpreter, test isolation and memory limits are verified; package boundaries and initial fixture reuse are explicit. No upstream mutation is needed.

Suggested change split: baseline/evaluation data; architecture and dependency configuration.

**Phase 1 — Define the minimum domain and contracts**

Purpose: create the stable vocabulary that later storage and tools share.

Work:

- Introduce snapshot, source-location, symbol, relationship, extraction-diagnostic and query-result dataclasses.
- Keep concept and verification models minimal until the operations using them arrive; record their required identity rules now.
- Define enums for fixed kinds, resolution states, exclusion reasons and result status.
- Distinguish logical symbol identity from a source locator at a revision. Include module and nested qualification; short names are never sufficient identity.
- Give repository snapshots an explicit repository identity, source revision/content digest and configuration identity. Keep extractor/parser/schema identity in the index manifest.
- Treat concept identities separately from code-symbol names so a verified rename can preserve a concept path.
- Define a typed query envelope with snapshot, evidence, diagnostics, completeness, result limits and pagination. A bounded answer must state when results were omitted.
- Add only the source and storage contracts required by the first operation. Contracts are owned by the consuming inner layer.
- Create the dependency-boundary test and document schema compatibility rules.

Deliverables: domain model, first abstract contracts, versioned result schema, invariants and boundary checks.

Exit gate: same-named and nested symbols remain distinct; the model imports no adapters or runtime entities; invalid records fail clearly; dependency cycles fail CI.

Suggested change split: identities and invariants; request/result contracts and boundary tests.

**Phase 2 — Build reproducible source snapshots and extraction**

Purpose: create a complete, inspectable structural record from a chosen commit.

Work:

- Discover source roots and distribution/import-name mappings from project manifests, including the root package and separately configured basstler.
- Read committed files from the requested Git revision, not whichever working files happen to exist. Handle paths and subprocess arguments structurally.
- Account for every selected file as indexed, intentionally excluded, unsupported or failed.
- Exclude generated ORM interfaces before opening them. Exclude caches, vendored code and binary assets from ordinary source analysis.
- Parse Python with the configured interpreter without importing the scanned modules.
- Reuse existing definition-span and import facilities where appropriate. Add missing qualified nesting, signatures, fields, decorators, enum members, annotations and generic-base expressions.
- Resolve aliases, relative imports, re-exports and unambiguous bases. Label candidate or unresolved relationships honestly.
- Distinguish type-only references, declarations, imports and possible runtime uses.
- Index Markdown and RST headings/source references. Record templates and declarative files as source evidence even when deeper semantic analysis is deferred.
- Handle large source files deliberately; do not silently inherit the candidate CLI's file-size exclusions.
- Define parse-failure behaviour within AGENTS.md. Prefer a parser process whose nonzero result becomes a diagnostic if that fits the implementation; a local exception-handler exception to the rules needs explicit approval before coding it.

Deliverables: revision manifest, extraction records, coverage report and source-analysis fixtures.

Exit gate: repeated extraction at identical inputs yields identical semantic records; generated mappings are never opened; large files and RST are accounted for; same-named classes are not guessed together; parsing starts no ROS/world runtime.

Suggested change split: committed inventory; Python declarations/imports; relationship resolution and document references.

**Phase 3 — Deliver persistent queries and the first useful CLI**

Purpose: make the structural index useful before adding LLM-generated knowledge.

Work:

- Implement SQLite persistence behind the read/write contracts, with atomic completed-snapshot publication.
- Preserve source evidence and extraction diagnostics with records. Record schema, parser, extractor and configuration versions.
- Implement find_symbols, describe_symbol, find_references, find_relationships and check_freshness.
- Support exact qualified names and lexical lookup; add SQLite full-text search only after confirming availability in the configured interpreter.
- Return source slices or locations from the indexed snapshot, with deterministic ordering and bounded traversal.
- Expose CLI operations with structured output suitable for an agent or basstler process call.
- Use the required KRROOD serializer for owned dataclass round-trips. Parse externally controlled data into explicit schema dataclasses; never import arbitrary classes named by untrusted index input.
- Specify cache rebuild/version refusal behaviour. Rebuild disposable indexes when appropriate instead of requiring premature migration machinery.
- Initially support committed-source mode explicitly. Editing assistance must not claim to include uncommitted changes yet.

Deliverables: queryable local index, first CLI, API contract tests and operating instructions.

Exit gate: navigation evaluation returns the expected definitions and references with valid citations; an interrupted refresh cannot expose half an index; mismatched versions are visible; missing data is not presented as proof that no reference exists.

Release checkpoint A: useful read-only navigation for a pinned commit.

Suggested change split: storage and snapshot transactions; bounded query operations; CLI and structured serialization.

**Phase 4 — Add working-tree freshness and incremental maintenance**

Purpose: make the index reliable during actual development.

Work:

- Build a separate overlay for staged, unstaged, added and deleted source files, including explicitly in-scope untracked source.
- Identify overlays by content and checkout identity; branch names and timestamps alone are insufficient.
- Snapshot file contents before parsing and detect a file changing during capture.
- Prevent cross-worktree cache reuse unless source and configuration identities match.
- Replace changed-file records and invalidate incoming relationships and dependent concepts. Re-resolve references when exports, aliases, bases or manifests change.
- Use conservative scope expansion for dynamic or insufficiently resolved dependencies.
- Handle renames as evidence-based reconciliation. Ambiguous mappings stay ambiguous.
- Ensure parse failures remove old facts from the current view or mark them unusable; historical snapshots remain distinguishable.
- Pin each query to one coherent snapshot and recheck selected source before serving editing context.
- Add explicit full reconciliation and compare its semantic output with incremental rebuilds.

Deliverables: overlay support, invalidation engine, reconciliation command and mutation-test dataset.

Exit gate: edits, additions, deletions, renames, branch switches, fixture changes and interrupted updates produce results equivalent to a clean full rebuild within the supported analysis scope. No test returns stale source as current. Unchanged inputs produce no semantic churn.

Release checkpoint B: reliable local source navigation during editing.

Suggested change split: overlay snapshots; invalidation and reverse edges; atomic refresh and reconciliation tests.

**Phase 5 — Add curated OKF knowledge**

Purpose: explain the code's architecture and extension rules with traceable evidence.

Work:

- Define repository-specific concept types and their evidence requirements.
- Implement an OKF import/export adapter while keeping YAML and Markdown structure out of the core model.
- Keep format conformance, repository policy and factual verification as separate results.
- Link concepts to source paths, identities and content digests. Distinguish official fields from repository extensions.
- Keep freshness independent from OKF lifecycle: expiry or source invalidation is not the same state as deprecated.
- Create root/package indexes and a small initial concept collection.
- Cover plan-to-motion execution, execution environments, adding an action, world fixtures, spatial conventions, generic dataclasses, roles and ORM generation.
- Start with deterministic material and reviewed source-backed explanations. An LLM provider is optional for this release and stays behind a drafting contract.
- Add draft, stable and deprecated transitions, source-change invalidation, replacement links and tombstones.
- Keep machine generation separate from review. A second model response is not authenticated human verification.
- Test candidate OKF CLI interoperability before deciding to make it a dependency.

Deliverables: initial knowledge bundle, import/export adapter, validation results and concept lifecycle tests.

Exit gate: every pilot claim has source evidence; edits invalidate the right concepts; round-tripping preserves supported extensions; syntax compliance cannot grant review status; relocated/deleted concepts retain meaningful links.

Suggested change split: OKF adapter and policy checks; pilot concepts; invalidation and tombstones.

**Phase 6 — Build task context and change assistance**

Purpose: make the combined system directly useful for code generation and review.

Work:

- Implement find_examples, find_tests, assess_change and prepare_context.
- Model test references, fixture scope and dependencies. Preserve uncertainty for indirect parametrization and dynamic test construction.
- Retrieve applicable AGENTS instructions and domain guidance without granting retrieved prose authority to override them.
- For generation tasks, assemble the interface, a current implementation example, constraints, source-backed concepts and relevant tests.
- For changes, compare public contracts, list resolved consumers and uncertain candidates, and recommend relevant test directories.
- Include relevant generator inputs for ORM tasks rather than reading generated interface files.
- Rank and budget context deterministically. If using character budgets or token estimates, label them accurately; a tokenizer dependency is a separate decision.
- Define task profiles for navigation, code extension, debugging, review and migration while sharing the same query operations.
- Keep the service read-only: it returns context and proposed checks, not automatic repository edits or test execution.

Deliverables: task-context API, example/test links, impact reports and end-to-end task evaluations.

Exit gate: pilot tasks receive the definitions, rules, examples and tests needed to act; all citations resolve against their snapshot; omitted context is visible; a suggested test is not misrepresented as complete coverage.

Release checkpoint C: the complete local agent-assistance pilot.

Suggested change split: examples/fixtures/tests; change comparison; context preparation and task evaluations.

**Phase 7 — Establish trusted validation and publication**

Purpose: share canonical knowledge without allowing its producer to authenticate itself.

Work:

- Default to owner-operated, machine-validated publication without another person's approval. Keep optional independent human review distinct from machine checks.
- Bind verification to exact concept content, relevant source evidence and policy version.
- Audit actual fork ownership and branch protection; CODEOWNERS contents alone are not proof of enforcement.
- Add ownership for knowledge, policy, extraction and publication components where absent.
- Run unprivileged PR checks for source consistency, policy, lifecycle transitions and deterministic extraction.
- Evaluate proposal data with trusted checker code; proposed policy changes cannot silently weaken their own gate.
- A separate pinned publisher authenticates the source commit/tree, validates source/content state and creates the canonical artifact. Independent-review mode additionally obtains authenticated review evidence.
- Do not carry an approval over changed content after merge conflict resolution or later source edits.
- Define artifact verification for the chosen distribution: authenticated commit/artifact identity and digest are mandatory; portable signing/attestation is added where consumers need it.
- Publish source identity, bundle digest, validation and the explicit publication mode together. Include human review evidence only when required by policy. Keep artifact hashes outside their own hashed payload.
- Add consumer refusal/fallback behaviour and rollback to a complete known publication.
- Mock external identity APIs in CI. Live credential-dependent tests require the repository's explicit approval and skip rules.

Deliverables: automated default policy, source/review adapters, fork-side CI workflows, publisher, consumer checks and rollback procedure. See [phase7.md](phase7.md) for implementation and activation.

Exit gate: forged frontmatter, stale approvals, altered bundles, policy weakening and wrong-revision artifacts fail the relevant trust gate. A rollback restores the associated code/knowledge evidence.

Release checkpoint D: a shared, verifiable knowledge baseline after immutable checker/runtime pins and hosted publication/consumption checks pass. See [the deployment checkpoint](phase7-deployment.md). No external reviewer is required by the default mode.

Suggested change split: admission policy and mocked evidence; trusted CI checks; publication and rollback.

**Phase 8 — Integrate agent clients and CRAMERA**

Purpose: expose one mature service through several consumers without duplicating its logic.

Work:

- Add a read-oriented MCP adapter for the existing operations after choosing and approving the required SDK or evaluated external tool.
- Keep the initial tool surface bounded; avoid raw Python execution or arbitrary SQL.
- Make CLI and MCP return equivalent domain results and diagnostics.
- Integrate basstler session startup with a bounded freshness check and entry-point report. Use the selected CRAM interpreter through the process contract.
- Add opt-in commit hooks only after their runtime cost is measured. Hooks queue or perform deterministic refreshes; they do not synchronously run an LLM and create follow-on commits.
- Adapt CRAMERA's architecture view to the shared source index. Preserve existing behaviour through meaningful boundary translation.
- Remove duplicate extraction/cache ownership once consumers have migrated and required API permissions are settled.
- Keep clients operational when an optional transport, viewer or generator is unavailable.

Deliverables: agent adapter, session integration, optional hooks and CRAMERA consumer integration. See [phase8.md](phase8.md) for the implementation contract, configuration and validation.

Exit gate: client contract tests agree; startup does not import or install the full robotics stack into basstler; tools cannot write canonical publication state; CRAMERA migration tests preserve existing supported views.

Suggested change split: MCP interface; basstler integration; CRAMERA migration and duplicate-code removal.

**Phase 9 — Validate the release and control operating cost**

Purpose: establish that the whole system improves real tasks and remains maintainable.

Work:

- Compare direct source search, structural queries, and structural-plus-curated context on the same revisions and tasks.
- Repeat after controlled mutations and include unresolved cases rather than scoring only easy queries.
- Measure task success, required-source recall, citation validity, stale answers, latency, memory, tool calls, agent tokens and amortized update cost.
- Add crash/restart, concurrent-reader, disk-write failure and corrupted-cache tests for the implemented interfaces.
- Define retention for rebuildable caches and published audit evidence. Bound worktree growth.
- Document recovery, full rebuild, schema changes, disabling optional integrations and rollback.
- Select latency and resource limits from baseline measurements and intended machine capacity. No unsupported throughput or token-saving percentage is promised.
- Verify new tests are in CI, use the configured interpreter, and obey package-by-package execution and memory limits.

Deliverables: evaluation report, operational limits, recovery guide and release checklist. The local implementation and experiment are described in [phase9.md](phase9.md), [the evaluation report](phase9-evaluation.md) and [the operations guide](operations.md).

Exit gate: no unexplained correctness regression from the baseline; no false-current results in the defined mutation tests; the agreed resource budget holds. If savings are absent, preserve useful navigation and revise curation/retrieval before expanding infrastructure.

Release checkpoint E: supported team release. It remains open until the unmeasured team-task and operating gates in [the release checklist](phase9-release.md) are satisfied.

Suggested change split: evaluation; resilience/resource controls; release documentation.

**Phase 10 — Add optional capabilities when measured gaps justify them**

These are independent follow-on decisions:

- Read-only runtime exports from KRROOD and CRAMERA, tagged with environment and source revision. Keep observed objects separate from source symbols.
- Better resolution through a language-server or other analysis adapter when static ambiguity materially harms tasks.
- Semantic retrieval if lexical and graph queries miss important concepts.
- JavaScript and C++ semantic extraction for parts currently represented only as source/document records.
- Broader multi-repository or organisational deployment when one-repository operations are reliable.

Each extension needs an explicit use case, evaluation cases, dependency review and its own failure policy. A vector database, graph server, model-router or distributed queue is not a mandatory prerequisite.

**5. Acceptance rules applying throughout**

- Source evidence, generated interpretation, validation and authenticated review remain distinguishable.
- A missing reference is not proof of no caller; a test reference is not proof of coverage.
- Every answer is associated with one coherent source snapshot.
- No generated ORM file is opened by ordinary indexing.
- No phase requires a live model or GitHub credential for its normal CI tests.
- No source index imports robot packages solely to discover declarations.
- Known exclusions and unsupported input remain visible.
- Canonical publication cannot be written by an ordinary query client or drafting producer.
- Every phase extends working capabilities; speculative framework scaffolding is avoided.
- New features and fixes follow test-first development. Existing failing tests are not rewritten to make implementation pass.
- Follow dataclasses, explicit abstract interfaces, type hints, section markers, exception and structured-data conventions. Long queries/templates belong in their own files.
- Use the configured interpreter, check free memory and enforce a memory cap before pytest. Run one package or a few directories at a time, with --orm-build never. Regenerate ORM explicitly only when needed.
- Run scripts/format_docstrings.py for modified Python files.
- Preserve user changes; author commits with the configured human identity. Push only to the fork after checking full remote output. Upstream PR mutation is outside this plan's authorization.

**6. Work to start first**

Begin with Phase 0 and the smallest Phase 1/2 slice: a pinned source snapshot, precise definitions and source locations for the plan-to-motion path, and a dataset proving that duplicate names, large files and generated-ORM exclusions are handled correctly. Then deliver describe_symbol through the persistent query layer.

This sequence produces a useful source-navigation capability before investing in LLM enrichment, publication credentials or a new interface. The first implementation review should contain the tested contracts and exact dependency proposal, not a large unverified wiki.



## Robot plan generation development

The requested simulation-plan use case has its own [PG1–PG6 plan](plan-generation.md). It builds on the source, context and notebook capabilities above, with source evidence first, then action selection, complete proposals, simulation validation, bounded repair and evaluation. Its phase numbers do not replace the platform phases.
