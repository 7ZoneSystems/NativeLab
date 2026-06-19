# Contributing to NativeLab & PhonoLab

Thanks for your interest in contributing. This document explains how to report bugs, suggest features, and submit code changes for both NativeLab (desktop) and PhonoLab (Android).

## Reporting Bugs

Open a GitHub issue and include:

- **NativeLab**: OS, Python version, full error traceback, steps to reproduce
- **PhonoLab**: Android version, device model, logcat output, steps to reproduce
- What you expected vs what actually happened

## Suggesting Features

Open an issue before writing any code. Describe what you want and why. This avoids wasted effort if the feature does not fit the project direction.

## Development Setup

### NativeLab (Desktop)

```bash
git clone https://github.com/7zonesystems/NativeLab.git
python -m venv env
source env/bin/activate
pip install -r requirements.txt
cd NativeLab
python main.py
```

### PhonoLab (Android)

```bash
git clone https://github.com/7zonesystems/NativeLab.git
cd NativeLab/PhonoLab/android
./setup_binaries.sh        # Download llama.cpp arm64 binaries
./gradlew assembleDebug    # Build APK
adb install app/build/outputs/apk/debug/app-debug.apk
```

Or open `PhonoLab/android/` directly in Android Studio.

## Submitting a Pull Request

1. Fork the repository
2. Create a branch from `main`: `git checkout -b your-feature-name`
3. Make your changes
4. Test that the app runs without errors
5. Open a pull request against `main` with a clear description of what changed and why

Do not open PRs directly against `main` without a linked issue or prior discussion.

## Code Style

### NativeLab (Python)
- Follow PEP8
- Use descriptive variable names
- Keep functions focused on a single responsibility
- No unused imports

### PhonoLab (Kotlin)
- Follow Kotlin coding conventions
- Use `runOnUi {}` for UI updates in fragments (lifecycle-safe)
- Use `optJSONObject()` / `optString()` for JSON parsing (never `getJSONObject()`)
- Use `PhonoLabApp` singletons - never create private instances in fragments
- Use `ThreadLocal<SimpleDateFormat>` for date formatting in multi-threaded code
- API endpoints return structured errors (model_not_loaded, server_busy, etc.) - never blank responses

## Project-Specific Rules

### NativeLab (Python)

**No circular imports.** Do not add top-level imports that create circular dependencies. If a module needs something from a higher-level module, use a lazy import inside the function that needs it.

**Config values come from GlobalConfig only.** Do not hardcode paths, thread counts, context sizes, or other configuration values.

**New inference engines go in `core/engines/`.** Follow the pattern of existing engines.

**New UI components go in `UI/Qt6widgets/`.** Keep UI logic separate from business logic.

**LAN device discovery lives in `api_server/`.** Device scanning, connection testing, and the Devices tab UI are in `api_server/device_discovery.py` and `api_server/devices_tab.py`. Uses existing `ApiRegistry` and `ApiConfig` for registration. Smart auth flow: auto-connects without key first, prompts only when auth required or key changed.

**All network operations go through `core/http_client.py`.** Use `get_http_client()` for HTTP requests. Never use raw `urllib.request` or `http.client` directly in new code.

**All backend operations go through `core/backend.py`.** Use `get_backend()` for model loading, generation, device management. Returns `BackendResult` with structured error handling.

### PhonoLab (Android)

**All fragments use PhonoLabApp singletons.** Never create `PhonoLabStore()`, `LlamaRuntime()`, or `ModelManager()` in fragments.

**Use `ChatFragment.Host` interface.** Never cast `activity as? MainActivity` directly.

**Lifecycle guards.** Always use `runOnUi {}` instead of raw `main.post {}` in fragments.

**No bare `error()` / `check()` / `require()`.** Return error strings or use `safeRunState()`.

**Session logs are capped at 500.** Use `ChatSession.addLog()`, don't append to `logs` directly.

## What Not to Contribute

- Features that require an internet connection at runtime (this is a local-first app)
- Dependencies that are not cross-platform
- Large binary files

## Contact

If you have questions, reach out before opening a PR:

- GitHub: [@7zonesystems](https://github.com/7zonesystems)
- Email: 7zonesystems@zohomail.in
