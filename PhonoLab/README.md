# PhonoLab Mobile Node

PhonoLab is a standalone Android-first mobile app that lives inside this repo
but is not packaged by the root NativeLab project.

- llama.cpp-only execution
- resumable Hugging Face GGUF downloads
- small-model catalog for low-memory phones
- mobile-safe model registry under the `PhonoLab` data directory
- context, path traversal, download size, timeout, and subprocess guardrails
- full GUI control flow with setup, model download, load, and chat actions
- Android Studio project with manifest permissions, Gradle config, and native
  Android UI under `android/`

## Storage

Runtime files are stored under a directory named `PhonoLab`.

- Android: app private storage, `.../PhonoLab`
- iOS: `~/Documents/PhonoLab`
- Desktop/dev: `./PhonoLab/data`

Override desktop preview storage with `PHONOLAB_HOME=/path/to/PhonoLab`.

## Android Studio

Open `PhonoLab/android` directly in Android Studio. It is a complete Android
project with its own Gradle files and does not depend on the root NativeLab
package metadata.

The Android GUI can:

1. Pull llama.cpp source from GitHub into app-private storage.
2. Download supported small GGUF models from Hugging Face.
3. Register downloaded models under app-private `PhonoLab/models`.
4. Load and run a bundled `llama-cli` runtime from `PhonoLab/runtime/bin`.

Android still needs a platform-built `llama-cli` binary. Put it at:

```text
<app files>/PhonoLab/runtime/bin/llama-cli
```

The Android Studio project includes a `syncLlamaCpp` Gradle task that clones or
updates llama.cpp source under `PhonoLab/android/external/llama.cpp` for native
runtime development.

## Python GUI Preview

The Python side is GUI-only. It exists for desktop preview and shared mobile
logic tests:

```bash
cd PhonoLab
python main.py
```

No PhonoLab CLI entrypoint is exported from `pyproject.toml`.

## Mobile Runtime Flow

1. Pull llama.cpp source.
2. Build or bundle the platform llama.cpp runtime for Android or iOS.
3. Download a small GGUF model from the in-app model catalog.
4. Load the model.
5. Chat through the mobile llama.cpp engine.

The Python layer is orchestration only. The prompt budget hot path can use the
optional C helper in `PhonoLab/native/mobile_core.c`; when the compiled helper
is not present, PhonoLab uses the same conservative Python fallback.
