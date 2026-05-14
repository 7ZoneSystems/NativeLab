# CLI - `nativelab --cli`

NativeLab now has a first-class terminal control center. It uses the same engines,
model registries, API profiles, Labs endpoints, skills, pipelines, and integration
profiles as the GUI, but keeps everything scriptable from a shell.

First launch still runs setup when no CLI preferences exist. After setup,
`nativelab --cli` opens the interactive menu:

```text
Chat
Models
API Models
Skills
Labs
Pipelines
Integrations
Status
Setup
Quit
```

For a beginner walkthrough, see
[`nativelab/cli/cli_guide.md`](../nativelab/cli/cli_guide.md).

## Direct Commands

```bash
nativelab --cli                         # setup if needed, then menu
nativelab --cli setup [--reset]         # onboarding wizard
nativelab --cli chat [--model PATH_OR_REF] [--ctx N] [--system TEXT]
nativelab --cli status
nativelab --cli lint <file.py> ...

nativelab --cli models [list|load|default] [target] [--ctx N] [--json]
nativelab --cli api-models [list|load|default] [target] [--json]
nativelab --cli skills [list|create|edit|delete|enable|disable|chat-on|chat-off] [name] [--json]

nativelab --cli labs list [--json]
nativelab --cli labs code-edit [--file path] [--prompt text] [--save] [--save-as path] [--edit-response] [--no-diff]
nativelab --cli labs py-to-doc --mode single --file path.py --out-dir docs/generated
nativelab --cli labs py-to-doc --mode queue --file a.py --file b.py
nativelab --cli labs py-to-doc --mode project --project ./src --out-dir docs/generated

nativelab --cli pipeline list [--json]
nativelab --cli pipeline show <name> [--json]
nativelab --cli pipeline run <name> [--text text | --file input.txt]

nativelab --cli endpoint [/snapshot|/runtime|/models|/api_models|/limits|/pipelines|/labs|/skills] [--json]
nativelab --cli serve --port 8765

nativelab --cli integrations list [--json]
nativelab --cli integrations discord list|show|create|edit|delete|run [name] [--json]
nativelab --cli integrations whatsapp list|show|create|edit|delete|run [name] [--json]
```

`--json` is available on inspect/list/show style commands so shell scripts can
consume NativeLab state without scraping tables.

## Chat Slash Commands

Inside `nativelab --cli chat`, normal messages go to the loaded model. Lines
starting with `/` control the shared runtime:

| Command | Effect |
| --- | --- |
| `/help` | List commands. |
| `/status` | Show backend, loaded model, ctx, and skill injection state. |
| `/models` | List registered local GGUF models. |
| `/api-models` | List saved API model profiles and refs. |
| `/load <path|@api/name>` | Load a local model or API model profile. |
| `/unload` | Unload the active model/API profile. |
| `/ctx <n>` | Change local model context and reload. |
| `/skills on|off|list` | Toggle or inspect active skill injection for chat. |
| `/pipelines` | List saved visual pipelines. |
| `/pipeline <name>` | Show a saved pipeline. |
| `/pipeline run <name> [text]` | Run a saved pipeline with text input. |
| `/labs` | List Labs routes. |
| `/py-to-doc <file.py> [...]` | Generate docs for one or more Python files. |
| `/py-to-doc project <dir>` | Generate project docs with checkpoint resume support. |
| `/code-edit [file] -- <request>` | Run the structured edit Lab and print a diff. |
| `/endpoint <route>` | Inspect an integration endpoint route. |
| `/serve [port]` | Serve the integration endpoint until Ctrl+C. |
| `/system <text>` | Set the current system prompt. |
| `/reset` | Clear conversation history. |
| `/lint <file...>` | Run Python linting. |
| `/save <file>` | Save conversation JSON. |
| `/quit` | Exit chat. |

Inline file references still work in normal chat messages:

```text
Explain @nativelab/labs/endpoints.py in simple terms.
```

The CLI embeds readable text files into the prompt and caps each file at about
60 KB.

## Models And API Models

Local and API models share one selector/runtime. Use API refs anywhere a model
target is accepted:

```bash
nativelab --cli api-models list
nativelab --cli api-models load "@api/grok4.1%28no%20reasoning%29"
nativelab --cli models default /path/to/model.gguf --ctx 8192
```

Defaults are saved in `localllm/cli_prefs.json`. API keys are redacted in all
display commands.

## Skills

Skills live in `localllm/skill/skills.json`. The CLI ensures the built-in
`edit` skill exists, lets you create/edit/delete skills, and stores the chat
skill toggle in `localllm/cli_prefs.json`.

```bash
nativelab --cli skills list
nativelab --cli skills create refactor-helper
nativelab --cli skills enable edit
nativelab --cli skills chat-on
```

When chat skills are on, active skill names, descriptions, and instructions are
injected through the shared Labs endpoint. Integrations that call the same LLM
endpoint can benefit from that context indirectly.

## Labs

`code-edit` uses the same structured edit operations as the GUI Lab. It writes
the current temp code to `localllm/temp_code_edit_file` and edit history to
`localllm/temp_code_edit.json`.

```bash
nativelab --cli labs code-edit --file app.py --prompt "add input validation"
nativelab --cli labs code-edit --prompt "create a tiny Flask app"
nativelab --cli labs code-edit --file app.py --prompt "simplify parser" --edit-response --save-as app_v2.py
```

`--edit-response` writes the raw model JSON to
`localllm/temp_code_edit_response.json` and opens `$EDITOR` before applying the
operations.

`py-to-doc` supports single file, queue, and project modes. Project mode uses
the GUI worker and its checkpoint/resume files under `localllm/temp`, so a
restart can resume after previously completed files/functions.

## Pipelines

The CLI can inspect and run saved GUI-created pipelines:

```bash
nativelab --cli pipeline list
nativelab --cli pipeline show direct
cat prompt.txt | nativelab --cli pipeline run direct
```

Pipeline authoring remains GUI-only in this pass.

## Integrations

The endpoint browser exposes NativeLab state for bots and external scripts:

```bash
nativelab --cli endpoint /snapshot --json
nativelab --cli endpoint /skills --json
nativelab --cli serve --port 8765
```

Discord and WhatsApp profile commands use the same JSON stores as the GUI:

```bash
nativelab --cli integrations discord create bot1
nativelab --cli integrations discord run bot1
nativelab --cli integrations whatsapp create wa1
nativelab --cli integrations whatsapp run wa1
```

Foreground bot runs print visible logs and stop with Ctrl+C.

## Files Written

| Path | Purpose |
| --- | --- |
| `localllm/cli_prefs.json` | CLI default model, ctx, and skill toggle. |
| `localllm/custom_models.json` | Extra local model paths. |
| `localllm/model_configs.json` | Model roles and runtime settings. |
| `localllm/api_models.json` | Saved API model profiles. |
| `localllm/skill/skills.json` | Shared model skill library. |
| `localllm/temp_code_edit.json` | Code Edit Lab history. |
| `localllm/temp_code_edit_response.json` | Last editable Code Edit model JSON response. |
| `localllm/temp_code_edit_file` | Code Edit Lab working file. |
| `localllm/temp/` | py-to-doc project checkpoints. |
| `localllm/integrations/*.json` | Discord and WhatsApp connector profiles. |

## Developer Notes

The CLI runtime lives in [`nativelab/cli/runtime.py`](../nativelab/cli/runtime.py).
It builds one shared `LabEndpoints` plus `IntegrationEndpoints` pair and binds
them to the loaded local/API engine. Argparse commands, the interactive menu,
the chat REPL, Labs, pipelines, and `serve` all reuse that runtime.
