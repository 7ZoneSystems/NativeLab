# Integrations

NativeLab exposes a small integration surface for bots, local scripts, webhook
bridges, and cloud workers. The goal is to let external tools discover what the
app can do without importing the main window.

## Where to find it

Open the **Integrations** tab in the GUI. The tab is split into sub-tabs:

- **Endpoints**: select a route, copy JSON responses, and start the local HTTP bridge.
- **Discord Bot**: create reusable bot connector profiles and save credentials locally.

The Python API lives in `nativelab/integrations/`.

## Endpoint routes

The endpoint returns plain JSON-compatible dictionaries.

| Route | Method | Purpose |
|---|---:|---|
| `/snapshot` | GET | Full catalog: routes, runtime, limits, models, API models, pipelines, labs. |
| `/runtime` | GET | Active backend, model name/path, context size, server port, loaded state. |
| `/models` | GET | Registered local GGUF models with role, family, quant, and vision metadata. |
| `/api_models` | GET | Saved API model configs with API keys redacted. |
| `/limits` | GET | Defaults, roles, app config, and config field metadata. |
| `/pipelines` | GET | Saved visual pipelines with block and connection counts. |
| `/pipelines/{name}` | GET | Raw saved pipeline JSON definition. |
| `/labs` | GET | Registered Labs features and integration metadata. |
| `/labs/py_to_doc` | GET | Metadata for the py-to-doc lab route. |
| `/call_llm` | POST | Send a prompt/messages to the active NativeLab engine. |
| `/integrations/discord_bots` | GET | Saved Discord connector profiles with tokens redacted. |

## Start the local HTTP endpoint

In the **Integrations** tab, set a port and click **Start**. The default URL is:

```text
http://127.0.0.1:8765
```

Example requests:

```bash
curl http://127.0.0.1:8765/runtime
curl http://127.0.0.1:8765/pipelines
curl http://127.0.0.1:8765/labs/py_to_doc
```

Call the active model:

```bash
curl -X POST http://127.0.0.1:8765/call_llm \
  -H "content-type: application/json" \
  -d '{"prompt":"Write a short project status update.","n_predict":300}'
```

`/call_llm` accepts:

- `prompt`: user prompt string.
- `messages`: OpenAI-style message list, optional if `prompt` is provided.
- `system_prompt`: optional system message.
- `n_predict`: max generation tokens.
- `temperature`, `top_p`, `repeat_penalty`: sampling controls.

## Python API

For in-process integrations, use `IntegrationEndpoints` directly:

```python
from nativelab.integrations import IntegrationEndpoints

endpoints = IntegrationEndpoints()
print(endpoints.handle("/snapshot"))
print(endpoints.to_json("/models"))
```

When NativeLab binds the integration endpoint inside the GUI, it can also route
work through the live app:

```python
text = endpoints.call_llm(prompt="Summarize the loaded model state.")
endpoints.request_context(8192)
endpoints.request_load_model("/path/to/model.gguf")
endpoints.request_unload()
```

For HTTP exposure, use `IntegrationHttpEndpoint`:

```python
from nativelab.integrations import IntegrationEndpoints, IntegrationHttpEndpoint

endpoints = IntegrationEndpoints()
server = IntegrationHttpEndpoint(endpoints, port=8765)
server.start()
```

Inside the GUI this is already wired to the live NativeLab endpoint.

## Discord bot connector

The Discord Bot sub-tab saves reusable connector profiles in:

```text
localllm/integrations/discord_bots.json
```

Create `bot1`, save it, then create `bot2` the same way. Each profile stores:

- bot token, application ID, optional guild ID
- NativeLab endpoint URL
- reply behavior, including ephemeral replies and direct `@Bot` mention replies
- editable system prompt, with the NativeLab Discord prompt kept as a preset
- request queue settings
- Discord permissions needed by the bot
- NativeLab access controls for model, runtime, pipeline, lab, and model-list commands

The runtime bot file is stored at:

```text
nativelab/integrations/examples/discord_bot.py
```

It reads the saved profile and exposes slash commands:

- `/help`: show commands enabled for this profile.
- `/ask`: sends the prompt to `POST /call_llm`.
- `/status`: reads `/runtime`.
- `/pipelines` and `/pipeline`: read saved pipeline metadata.
- `/labs` and `/lab`: read Labs route metadata such as `/labs/py_to_doc`.
- `/models`: reads local and API model catalog routes.
- `@Bot <message>`: optional direct mention reply mode, enabled per saved profile.

Setup:

```bash
python -m pip install discord.py aiohttp
```

Packaged installs include these dependencies. When running from a fresh clone,
install them with the command above or `python -m pip install -e .`.

Start NativeLab, open **Integrations > Endpoints**, click **Start**, then open
**Discord Bot** and save a profile. Use **Start Bot** / **Stop Bot** in that
sub-tab to run the selected profile from inside the app. The **Bot Logs** panel
shows startup, Discord connection, endpoint calls, queue activity, and command
errors.

You can also run the same saved profile from a terminal:

```bash
export DISCORD_BOT_PROFILE="bot1"
python nativelab/integrations/examples/discord_bot.py
```

You can override the token or endpoint for a run without changing the saved
profile:

```bash
export DISCORD_BOT_TOKEN="your-token"
export NATIVELAB_INTEGRATION_URL="http://127.0.0.1:8765"
```

## Security notes

- The built-in HTTP endpoint binds to `127.0.0.1` only.
- API keys are redacted in `/api_models`.
- Discord bot tokens are saved locally in `localllm/integrations/discord_bots.json`.
- Direct mention replies require Message Content Intent enabled in Discord's
  Developer Portal for that bot application.
- Do not expose the endpoint directly to the public internet.
- Treat `/call_llm` as a trusted local capability because it can send prompts to
  whichever model or API backend is loaded in NativeLab.
