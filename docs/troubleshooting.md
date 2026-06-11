# Troubleshooting

Common errors and their fixes. If something here doesn't match what you're seeing, open an issue at <https://github.com/7zonesystems/NativeLab/issues>.

---

## Install / launch

### `command not found: nativelab`

The package installed but the script isn't on `$PATH`. Run it as a module instead:

```bash
python -m nativelab          # GUI
python -m nativelab --cli    # CLI
```

If you used `pip install --user`, also make sure your user bin dir is on `$PATH`:

```bash
# Linux / macOS
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# Windows
# Add %APPDATA%\Python\Python3xx\Scripts to your PATH env var
```

### `ModuleNotFoundError: No module named 'PyQt6'`

`pip install nativelab` should pull this in. If it didn't:

```bash
pip install PyQt6
```

If `pip` complains about wheels, your Python is probably too old. PyQt6 wheels start at 3.10.

### `ModuleNotFoundError: No module named 'nativelab'`

You're either missing the install, or running Python from a folder where `nativelab/` exists as a sibling directory and Python imports the *folder* instead of the installed package. Run from a different cwd or use `python -m nativelab`.

---

## llama.cpp binaries

### "No llama.cpp binary found."

The CLI / GUI couldn't locate `llama-server` or `llama-cli`. Three fixes:

1. **Easiest** - open the GUI, go to **⬇️ Download**, install a release.
2. **Manual** - drop binaries into `./llama/bin/` (relative to your working dir).
3. **Custom path** - set explicit paths in the GUI's **🖥️ Server** tab. They land in `./localllm/server_config.json`.

### "Server start failed - falling back to llama-cli mode"

Not fatal. The model still works, but each prompt restarts the process (slow for multi-turn). Common reasons:

- The chosen port is already in use → adjust port range in **Server** tab.
- The server binary is missing - only `llama-cli` is on disk.
- A stale `llama-server` from a previous crash is hogging the port. The app calls `kill_stray_llama_servers()` at startup, but if that misses, `pkill llama-server` (Linux/macOS) or Task Manager (Windows) clears it.

### Permission denied on Linux/macOS

```bash
chmod -R +x ./llama/bin/
```

---

## Models

### "Model not found"

`LlamaEngine.load()` couldn't resolve the path. Check:

- The path in the model registry matches the file on disk (paths can drift if the file moved).
- The file isn't 0 bytes (interrupted download). Re-run the wizard or re-download from the **Download** tab - `.part` files resume automatically.

### Model loads but replies are gibberish / repeats / ignores instructions

Almost always a **prompt template mismatch**. NativeLab auto-detects the family from the filename, but if the file was renamed or uses an unusual name, detection can fall back to a generic template.

Quick fix: in the Models tab, edit the model and change the `family` field to match the actual model. Add a new family entry if needed - see [models.md#adding-a-new-model-family](models.md#adding-a-new-model-family).

### "Model load: Failed" with no detail

Check **Dev > Logs**. Most common causes:

- Out of RAM - pick a smaller quant or lower the context.
- Corrupted GGUF - re-download.
- llama.cpp version too old for the quant format (e.g. IQ-quants need a recent build).

---

## Performance

### Replies are very slow

- Lower the context size (`/ctx 2048` in the CLI, slider in the GUI).
- Use a smaller quant (Q4_K_M is the sweet spot for most hardware).
- Enable GPU offload in the **Server** tab if you have a GPU.
- If you're in CLI fallback mode, get the server binary working - server mode is dramatically faster for multi-turn.

### Replies are cut off mid-sentence

The `n_predict` budget for that call ran out. Either:

- Increase `default_n_predict` in **Config**, or per-model `n_predict` in the Models tab.
- Just ask the model to "continue" as your next message.

### High RAM usage during summarisation

Expected for long documents. The watchdog spills reference caches to disk when free RAM drops below `ram_watchdog_mb` (default 800 MB). You can:

- Lower `max_ram_chunks` in **Config**.
- Raise `ram_watchdog_mb` to spill earlier.
- Set `auto_spill_on_start: true` to start lean.

---

## CLI

### Setup wizard exits with "Aborted: cannot proceed without binaries or API mode"

You answered "no" when asked whether to continue without local binaries. Re-run with `nativelab --cli setup` and either install binaries first or answer "y" to use API-only mode.

### CLI shows no icon at startup

The icon only renders on iTerm2 / WezTerm / VS Code's terminal / Hyper / mintty / Kitty. On other terminals (gnome-terminal, xterm, plain SSH) you'll just see the ASCII banner - that's the silent fallback, not a bug.

### `@file` ref is ignored

Make sure there are no spaces or quotes between `@` and the path:

```text
✗  @ ./foo.py            # space breaks it
✗  @"./foo.py"           # quotes break it
✓  @./foo.py
✓  @/abs/path/to/foo.py
```

The path is resolved relative to the current working directory.

---

## Pipelines

### "Pipeline doesn't progress past the first block"

- Check that **Input** is connected to something downstream.
- A FILTER block may have terminated the pipeline; check the execution log for "drop" lines.
- Each port can only have one outgoing connection on non-logic blocks. If you need to fan out, use a SPLIT block.

### Loop never terminates

The loop count is the maximum number of *visits* to that edge. If you exceed it via another path, it won't loop. Add a clear branch condition or use a fixed iteration count.

### Pipeline or AI Builder says the context limit is too small

The request plus reserved output tokens exceeds the loaded model context window.
Increase the model context limit and reload the model, or shorten the prompt /
canvas history. In AI Builder, the preflight check blocks the request before it
is sent to the model.

If the upstream server returns an error like:

```text
request (...) exceeds the available context size (...)
```

NativeLab should show a normal dialog explaining that the prompt is too large
and suggesting a larger context or shorter input. The same error is still logged
for debugging.

### AI Builder: "The model response did not contain a JSON object"

AI Builder retries once with a stricter JSON-only prompt. If it still fails,
make the request shorter and more direct, for example:

```text
Make an input -> model -> output pipeline.
```

Avoid asking for explanations in the same request. The model must return a JSON
object, not Markdown or prose.

### AI Builder generated a pipeline but it will not save

Generated pipelines still run through the normal validator. Fix the reported
validation issue manually or ask the AI Builder to revise the graph. Common
causes are missing Input/Output blocks, invalid connection endpoints, direct
model-to-model links, missing LLM logic instructions, or unsafe Custom Code.

### Pipeline sidebar disappeared

The pipeline builder sidebars snap into a narrow rail when dragged too small.
Look at the left or right edge of the canvas and click the circular arrow button
to reopen the sidebar.

### Custom Code block: "name X is not defined"

The sandbox restricts builtins. The full list available in custom code is documented in [workflows.md#custom-code](workflows.md#deterministic-logic-no-model-calls). For wider access, your code should run outside the sandbox (e.g. as a regular Python script invoked via subprocess from a Custom Code block).

---

## API models

### "API test failed: 401 Unauthorized"

Wrong API key, or the key doesn't have access to the model you specified. Re-check both fields in the API Models tab.

### "API test failed: connection refused"

The base URL is wrong or the server isn't running. For self-hosted servers (Ollama, LM Studio, vLLM), make sure they're listening on the address + port you specified.

### Anthropic API: "credit balance is too low"

Anthropic returns 400s when out of credit. The error message in **Dev > Logs** includes the upstream response body.

---

## Reset to a clean state

If things are weirdly broken and you want to start over without uninstalling:

```bash
# From your nativelab working dir
rm -rf localllm/cli_prefs.json localllm/server_config.json
rm -rf paused_jobs/ ref_cache/ ref_index/ sessions/
rm -f app_config.json
```

Models in `./localllm/*.gguf` are preserved. Re-launch and the wizard / GUI will recreate everything.

---

## Still stuck?

- **Dev > Logs** in the GUI has the full error chain - it's the most useful single place to look.
- Run with `python -u -m nativelab --cli` to see all stderr in real time.
- Open an issue with the relevant log lines + your OS / Python version: <https://github.com/7zonesystems/NativeLab/issues>.
