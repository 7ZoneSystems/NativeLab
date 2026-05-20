# Installation

NativeLab runs on **Linux, macOS, and Windows**. Same install, same commands.

---

## 1. Install the Python package

```bash
pip install nativelab
```

This pulls in `PyQt6` for the GUI and `psutil` / `PyPDF2` for live RAM monitoring and PDF support. The CLI works out of the box on top of the same install.

> **Python version**: 3.10+ recommended. The minimum supported is 3.9, but PyQt6 wheels ship for 3.10 and up.

Verify:

```bash
nativelab --help
nativelab --cli --help
```

---

## 2. Pick a workspace folder

NativeLab keeps everything in the **current working directory** - no hidden config files in your home directory. Create a folder you'll use as the default working dir:

```bash
mkdir ~/nativelab
cd ~/nativelab
```

Anything NativeLab writes (models, sessions, configs, paused jobs) lands here. Move the folder to back it up; delete it to start fresh.

---

## 3. Install llama.cpp binaries

You need `llama-server` and `llama-cli` somewhere on disk. There are three ways to get them:

### Option A - Use the GUI's built-in installer (easiest)

```bash
nativelab
```

In the desktop app, open the **⬇️ Download** tab. Pick the latest llama.cpp release for your platform and click Install. Binaries land in `./llama/bin/` automatically.

### Option B - Download a release zip manually

Grab the right archive from <https://github.com/ggml-org/llama.cpp/releases> and extract it into `./llama/bin/`:

```text
llama/bin/
├── llama-cli
├── llama-server
└── (helper libraries)
```

On Linux/macOS, make them executable:

```bash
chmod -R +x llama/bin/
```

### Option C - Build from source

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make -j
mkdir -p ../nativelab/llama/bin
cp build/bin/llama-cli build/bin/llama-server ../nativelab/llama/bin/
```

NativeLab's path resolution order:

1. `cli_path` / `server_path` from `server_config.json` (set via the GUI's **Server** tab).
2. Bundled binaries at `llama-bin/` (frozen builds only).
3. Dev fallback `./llama/bin/`.

---

## 4. Install or download a model

You need at least one model. GGUF is the simplest local path, but NativeLab can also register running Ollama models and optional HF Transformers snapshots.

### From the GUI

- Open the **⬇️ Download** tab.
- Type a HuggingFace repo (e.g. `TheBloke/Mistral-7B-Instruct-v0.2-GGUF`) and click **Search**.
- Download a quantization that fits your RAM.

The same Download tab can also:

- Download full HF Transformers snapshots into `localllm/hf_transformers/` and register them as `hf:<local-folder>`.
- Pull Ollama models through an already-running Ollama daemon and register them as `ollama:<model>`.

HF Transformers support is optional:

```bash
pip install -e ".[hf]"
```

### From the CLI

```bash
nativelab --cli setup
```

The wizard offers three pre-vetted starter models or lets you enter any HF repo ID and pick a quantization.

### From your own files

Drop any `.gguf` file into `./localllm/`, or use the **Browse GGUF…** button in the GUI's Models tab.

> **Sizing rough guide** - 7B Q4 ≈ 4.5 GB RAM · 13B Q5 ≈ 9.5 GB · 70B Q4 ≈ 38 GB. GPU offload (Server tab) reduces RAM by moving layers to VRAM.

---

## 5. Run it

```bash
nativelab            # GUI
nativelab --cli      # CLI
```

You're done. From here:

- New to terminals? Read the [CLI beginner guide](../nativelab/cli/cli_guide.md).
- Want a feature tour? See [features.md](features.md).
- Hit a snag? Try [troubleshooting.md](troubleshooting.md).
