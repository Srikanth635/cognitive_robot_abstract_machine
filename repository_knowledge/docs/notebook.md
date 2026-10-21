# Local agent connection and model playground

Use [knowledge_playground.ipynb](../notebooks/knowledge_playground.ipynb) to inspect the real MCP server without credentials, then optionally test source-grounded Q&A and code proposals with your own model. These experiments provide usage observations; they do not certify generated changes or close the [release checklist](phase9-release.md).

## Installation and source selection

Use the configured Python 3.12 interpreter. The approved optional extra contains LiteLLM, MCP, ipykernel, nbformat and nbclient. Core indexing and CLI imports do not require this extra.

```powershell
./repository_knowledge/.venv/Scripts/python.exe -m pip install --editable './repository_knowledge[notebook]'
```

The local setup selects the isolated `cram_wiki` checkout, not another checkout's files. Its working-tree cache is `repository_knowledge/.cache/agent-working.sqlite3`. Both the notebook and MCP host select `HEAD`, working-tree scope, and explicit untracked inclusions `repository_knowledge` and `test/repository_knowledge_test`. Ignored files, virtual environments, cache files and generated ORM interfaces remain excluded by the existing source policy.

Create the cache once, from the repository root:

```powershell
./repository_knowledge/scripts/run_command.ps1 -Module repository_knowledge `
  -Arguments @('--repository', '.',
    '--database', 'repository_knowledge/.cache/agent-working.sqlite3',
    '--scope', 'working-tree',
    '--include-untracked', 'repository_knowledge',
    '--include-untracked', 'test/repository_knowledge_test', 'index')
```

After editing indexed source, staging or committing, use the same command with `refresh` instead of `index`. Working-tree freshness binds HEAD, indexed disk bytes, Git staging state and inclusion policy. No refresh hook or background worker is installed by this setup. Keep the selected cache paired with its checkout; never copy a cache from another working tree and label it current.

## Codex connection

The Codex CLI registers a stdio server using an absolute interpreter, checkout and cache. The checked-in [Codex template](../config/codex.example.toml) shows the fields. Replace its placeholders before using it. The official CLI preserves unrelated host configuration:

```powershell
$repository = (Get-Location).Path
codex mcp add cram-knowledge -- "$repository/repository_knowledge/.venv/Scripts/python.exe" `
  -m repository_knowledge.interfaces.mcp `
  --repository "$repository" `
  --database "$repository/repository_knowledge/.cache/agent-working.sqlite3" `
  --scope working-tree `
  --include-untracked repository_knowledge `
  --include-untracked test/repository_knowledge_test
codex mcp get cram-knowledge --json
```

Set `tool_timeout_sec = 300` and `startup_timeout_sec = 30` in this server's table in your private Codex config. Rich context exceeded the host's default 60-second tool deadline in the measured pilot. Reload/restart the host or start a fresh session after configuration changes; registering the server does not add tools to an already running conversation's catalog. See [official Codex MCP documentation](https://learn.chatgpt.com/docs/extend/mcp?surface=cli).

The server offers bounded, read-only `query_repository`. The coding host can use returned evidence when making ordinary changes with its own editing tools. The MCP server itself neither edits files nor runs generated code. Navigation, references, relationships, examples, test discovery, context and freshness share the existing application dispatcher. This local connection supplies source evidence; it does not activate newer canonical publication or relabel local wiki drafts as independently verified guidance.

## Notebook kernel and credentials

Select the **CRAM Knowledge** kernel. The Windows local kernel launches `scripts/run_kernel.ps1`, which delegates to the existing guarded command runner with at least 8 GiB free memory and a 4 GiB process-tree cap. This includes the kernel and repository-server children. Other machines must configure an equivalent resource limit before running experiments. The portable notebook records the kernel name, never an absolute personal path.

The Windows kernel uses Jupyter's standard `signal` interrupt mode. The launcher forwards extra kernel arguments, including nbclient's temporary history selection. The Windows regression checks the authoritative execution reply for `KeyboardInterrupt`; Jupyter can display an interrupt as stderr rather than a notebook error object.

Use an ignored local copy for interactive sessions so your provider selections and saved outputs stay out of the tracked template:

```powershell
New-Item -ItemType Directory -Force repository_knowledge/.cache/notebooks | Out-Null
Copy-Item repository_knowledge/notebooks/knowledge_playground.ipynb repository_knowledge/.cache/notebooks/knowledge_playground.ipynb
```

The local setup creates that copy once. Do not copy over it later if it contains an experiment you want to keep. Run the notebook's offline cells first. The connection checks source freshness, discovers the actual MCP tool and describes the selected declaration. `CRAM_NOTEBOOK_REPOSITORY`, `CRAM_NOTEBOOK_DATABASE`, `CRAM_NOTEBOOK_SCOPE` and `CRAM_NOTEBOOK_SYMBOL` can override the defaults for isolated fixture runs or a different source selection.

Then select a [LiteLLM provider](https://docs.litellm.ai/docs/providers), a model available to your account and optionally a compatible/local endpoint. Set `RUN_LIVE = True` and enter the API key in the hidden prompt. No key belongs in a notebook cell, repository file or chat. Runtime settings exclude the key from their representation and are rejected by the evidence serializer. The trial answer redacts the supplied key. Provider/SDK debug output is outside that protection; leave debug logging disabled. Restarting the kernel clears the in-memory credential. Existing environment credentials may also be used by a provider.

The default `Run All` does not call a provider, ask for a key or save reports. Live execution sends questions and model-selected repository evidence to the selected provider and may incur charges. It does not transmit the whole repository automatically.

## Model execution and observations

The optional interface uses LiteLLM's existing `Harness.TOOL_LOOP`, not a second orchestration engine. Its scratch sandbox is an empty temporary directory: SDK filesystem snapshots cannot traverse the CRAM checkout. The only supplied function is a flat schema adapter to the existing MCP query contract; models cannot select the checkout, database or executable.

Requests retain the existing page/depth/character bounds. Defaults allow six model-selected tool attempts, 2,048 output tokens per completion, 60 seconds per provider request and 300 seconds per model turn. Repository queries have their own 300-second response deadline. These are configurable budgets, not latency promises. Source is checked before any paid request and again after the turn; changed source refuses the result. Intermediate MCP query failures remain visible to the SDK/model, and the stop reason remains visible in the receipt.

Receipts record model identity, answer, stop reason, source commit/snapshot/scope, elapsed time, reported input/output tokens and model calls, model tool attempts and individual repository query sizes/times. Missing SDK usage remains `None`, not measured zero. A positive SDK cost estimate is not a provider billing receipt. Initial operator verification is distinguished from model-selected queries. The final source check is an operator validation and is not counted as a model tool call.

Optional saving writes only typed `ModelTrial` evidence into the ignored `.cache/notebook-trials/` directory with a unique filename. Reports contain answers and query selections, which can still be sensitive. No provider settings are serialized. Clear notebook outputs before sharing or committing an executed copy.

## Validation and the next experiment

Tests exercise the actual SDK tool loop with a mocked provider and real MCP subprocesses. They verify usage aggregation, typed round trips, credential exclusion, default live refusal and source-change refusal. nbformat validates the clean notebook, and nbclient executes every cell against an isolated committed fixture using the configured interpreter. No CI credential or external model call is needed; the package CI job installs the optional extra and uses LiteLLM's bundled cost map without an import-time metadata fetch.

To measure usefulness, choose one representative question and one small coding task with expected evidence and independent acceptance checks. Keep source, model and task fixed, compare against your normal source-navigation workflow, and record actual provider usage plus end-to-end task results. A generated proposal or a smaller response alone does not establish correctness or token savings. Live model trials, hosted CI and a successor canonical trust-root activation remain separate work.


## Robot planning context

For a simulation-plan request, the model-facing `query_repository` function accepts `profile="plan-generation"` with the existing `prepare-context` tool and a qualified `symbol_identifier`. The profile adds action/plan contracts, deferred-grounding evidence and methods from the PR2 place-setting demonstration. Inspect omissions and excerpt limits; use bounded symbol queries for the parts omitted. The demonstration defaults to kinematic simulation, and its source does not prove support for a different robot or backend.

This is PG1 of the [robot plan generation development](plan-generation.md). The notebook can propose text using this evidence, but a dedicated complete-plan workflow and simulation validator are subsequent phases. No robot world or provider request is started by selecting the profile.
