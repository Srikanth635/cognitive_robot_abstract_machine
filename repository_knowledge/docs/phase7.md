# Phase 7: validated publication and consumption

The default in this fork is **automated publication**. Srikanth can publish without another person's approval, a pull request or an environment approval. Machine validation establishes source consistency and artifact identity; it does not establish that every explanation is semantically correct. `authenticated_review` remains false in this mode. Independent human review is an optional, stricter policy mode.

## Layout and dependency direction

Everything continues under `repository_knowledge/`: source code in `src/`, wiki concepts and navigation in `wiki/`, the trusted policy in `policy/publication.toml`, and plans/contracts in `docs/`. Disposable SQLite graphs stay in ignored `.cache/`. Canonical publication is a GitHub Actions artifact containing one `publication.json`; it is not another tracked generated wiki or cache.

`publication.py` defines publication, policy, identity and refusal records. Operations depend on abstract `KnowledgeReader`, `ReviewReader`, `SourceIdentityReader` and `RecordDigester` contracts. Git, captured Markdown, provider APIs, digest encoding and artifact archives remain adapters. CLI composition selects adapters. Existing dependency-direction and cycle tests cover these additions. There are no new dependencies or KRROOD modifications.

## Admission and publication

1. Select a full source commit and a separately trusted full checker commit. The checker checkout must be clean, and both the running knowledge package and KRROOD must come from that checkout.
2. Read proposal Git objects as data. Do not install, import or test proposed Python modules. Do not load a proposal-provided cache or policy.
3. Extract the full source graph twice in separate interpreters, with a specified verification hash seed. Compare framed digests of the manifest, sources, declarations, references, diagnostics and raw analyses.
4. Compare protected checker, scripts, dependency, workflow and policy paths against the trusted checker. Changes require an explicit owner-managed trust-root update; proposal data cannot authorize its own new checker.
5. Validate captured wiki documents against exact committed bytes, attribution, links, source evidence and lifecycle rules. New concepts must be drafts. Removed concepts require retained tombstones; changed concepts remain draft or deprecated. Existing pilot concepts remain drafts.
6. For automated publication, authenticate the configured fork, branch and exact commit/tree through GitHub. The commit must be in the configured branch's history. Branch protection and review APIs are not required in this mode.
7. Bind the bundle, every concept, source snapshot, policy and checker revision. Include cited source files/declarations and validation in one payload. Upload only that payload as an immutable artifact.

The admission baseline is the supplied PR base; publication compares lifecycle against the source commit's first parent. This does not compare every historical intermediate commit or establish compatibility with every older publication. A trust-root update may deliberately change policy; the owner must select the resulting new checker pin explicitly.

Full extraction remains necessary for this trust check. Incremental local refresh from phase 4 remains useful for navigation, but its cache is not trusted as canonical admission evidence. Time, memory and source-size limits can refuse a run; refusal cannot produce a partial publication.

## Commands

Use the configured interpreter with a clean trusted checkout containing this phase. The checker revision and policy must be selected independently of the proposal. Place output outside the checker checkout.

```powershell
$checker = 'C:/path/to/clean/trusted-checkout'
$source = 'C:/path/to/source-checkout'
$checkerCommit = '<full trusted checker commit>'
$sourceCommit = '<full source commit>'
$baseCommit = '<full baseline commit>'
$python = '<configured Python interpreter>'

& $python -m repository_knowledge publication --repository $source --trusted-repository $checker --checker-revision $checkerCommit check --revision $sourceCommit --base-revision $baseCommit

& $python -m repository_knowledge publication --repository $source --trusted-repository $checker --checker-revision $checkerCommit publish --revision $sourceCommit --output C:/Temp/new-canonical-publication

& $python -m repository_knowledge publication --repository $source --trusted-repository $checker --checker-revision $checkerCommit consume --revision $sourceCommit --artifact-id 12345
```

The interpreter must load `repository_knowledge/src` and `krrood/src` from `$checker`, rather than an editable installation from another checkout. `KNOWLEDGE_IDENTITY_TOKEN` supplies a read-only provider token for publication/consumption; offline admission requires no credential. The module entry and `cram-publication` expose the same interface. `--request` is optional in automated mode.

## CI activation

The workflows are fork-only. `knowledge-admission.yml` evaluates PR data with a pinned checker under `pull_request_target`, or an owner-selected source/base pair through manual dispatch; it never executes a proposed script, installs proposed dependencies or passes credentials into the checking container. `knowledge-publish.yml` is manually dispatched by the owner with a full source commit. Its executing workflow revision must equal the checker pin. There is no human approval environment. `knowledge-verify.yml` consumes an explicit artifact/source pair with a read-only token and uploads the typed verification result. Dispatch it for an earlier pair after publishing a later snapshot to exercise historical restoration.

Activation reads `repository_knowledge/policy/activation.env` from the fork's owner-managed `main` branch through `knowledge-trust.yml`. It is outside proposal data and is never sourced as executable shell code. Its five fields are `checker_revision`, `runtime_image`, `source_commit`, `base_commit` and `artifact_id`. The parser rejects unknown/duplicate keys and malformed values. The checker and source/base use full Git commits; the runtime uses an image digest; the artifact ID is zero until a publication exists.

Register the five knowledge workflows on the fork's default branch before dispatch. Build `knowledge-runtime.yml` at the selected checker revision, then commit its reported `ghcr.io/srikanth635/cram-knowledge-runtime@sha256:...` digest and the source/checker identities in the activation file. The runtime recipe uses a digest-pinned Python base and hash-pinned Linux wheels for the existing checker dependency closure. Checker and KRROOD source come from the selected checkout. Registry login happens on the runner using its package-read token, outside the network-isolated admission container.

Manual dispatch remains available. Dedicated refs also support remote operation: `codex/knowledge-runtime`, `codex/knowledge-admission`, `codex/knowledge-publisher` and `codex/knowledge-verification` trigger their respective workflows. Publication/admission/verification refs must point at the configured checker revision. A source-branch push cannot acquire publisher identity. Select a stable publisher ref at the checker revision; source changes continue on `cram_wiki`. The immutable checker pin remains an explicit trust-root decision. The activation file selects the default source/artifact pair; manual source/artifact inputs can select a historical pair explicitly.

The runner checks for 5 GiB available memory and caps the entire container, including the second extraction process, at 4 GiB. Checker/proposal mounts are read-only; admission has no network; publication receives only the read-only provider token. Dependencies come from the pinned runtime. Actions are pinned to full commits. Artifact retention is 90 days; artifacts must still exist when consumed.

Hosted activation results are recorded in the delivery progress. No remote variables, branch protection, environments or secrets are needed for the activation-file setup. The inspected fork branch had no enforced branch protection/rulesets; the automatic policy does not pretend otherwise.

## Consumption, fallback and rollback

The consumer selects an explicit artifact ID and expected source commit. It verifies the provider's repository, successful dispatch or authorized publisher-ref run, workflow path, pinned workflow revision and SHA-256 archive digest before decoding a bounded single-file archive. API authorization is removed on artifact redirects. Owned JSON uses the existing KRROOD serializer and its explicit type allowlist.

It then rechecks payload identity, policy, concept/bundle hashes, exact source identity and current evidence validity. An altered payload, wrong source, changed policy, invented human-review flag or stale evidence yields `source-only`, no canonical payload, and exit code 2. Invalid provider replies, unavailable artifacts and transport/decoding failures abort visibly; clients must retain source navigation rather than treating failure as validation. No unsigned local JSON file can substitute for the authenticated artifact boundary.

Rollback means selecting an older artifact together with its matching source commit and trusted checker/policy. Acceptance returns the complete historical bundle and its cited source facts. It does not reset Git, overwrite local files or silently mix old prose with current source. If the old source is no longer in the configured branch history, the artifact expired, the trusted policy changed or evidence expired, consumption refuses; restore the appropriate trusted checkpoint explicitly before retrying.

The phase-6 context commands still consume local draft wiki evidence. Automatic ingestion of canonical artifacts into agent clients belongs to phase 8; this phase exposes an explicit verified consumption result. There is no LLM enrichment service or automatic prose rewriting in these workflows.

## Optional independent review

`mode = "independent-review"` additionally requires configured human reviewers, positive approval count, enforced branch/code-owner/status protections, exact reviewed/merged tree equality and current eligible approvals. Self-review, stale approvals and changes requested are refused. The artifact adapter also requires an actual independent environment approval. This mode is tested with mocked provider evidence and is **not enabled** in this fork. CODEOWNERS is an ownership declaration, not proof that any of those settings are enforced.

## Validation

Tests use disposable repositories and mocked provider replies. They cover deterministic cross-process extraction, checkout/dependency custody, governance/lifecycle refusal, source identity, optional review controls, archive identity/digests, automatic publication without approval, payload round trips, fallback and complete historical consumption. Hosted Docker/workflow execution requires activation and is not claimed by local tests.

The package run before the remote-activation extension passed 207 tests with one Windows-only skip in 174.02 seconds under the 4 GiB job cap. Focused checks cover publisher-ref identity and activation-file parsing, including attempted shell injection. Final local validation passed 211 tests with one Windows-only skip in 152.81 seconds under the 4 GiB cap; formatting and unused-import checks passed. New serialization and policy-custody cases also passed focused runs. The wheel builds, Bash syntax checks pass, and valid Python files pass Black and unused-import checks. The workflow assertions pass. The added runtime recipe and build-workflow checks also pass. Hosted execution is verified separately after activation.
