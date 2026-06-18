# PhonoLab Documentation

Developer reference for the PhonoLab mobile LLM app.
Read these instead of scanning the whole repo.

---

## Documents

| File | What It Covers |
|------|----------------|
| [FILE_INDEX.md](FILE_INDEX.md) | Every file — purpose, classes, functions, constants, dependencies |
| [CONSTANTS.md](CONSTANTS.md) | All limits, defaults, paths, colors, URLs, regex patterns |
| [ANDROID_APP.md](ANDROID_APP.md) | Android architecture, JNI, crash handling, math, build |
| [CONTRIBUTING.md](CONTRIBUTING.md) | "I want to change X" → which file, dependency chains |

---

## Quick Navigation

### I need to change...

- **Model defaults** → `config.py` / `data/ModelManager.kt`
- **Model catalog** → `small_models.py` / `ModelCatalog.kt`
- **Chat format** → `data/ChatSession.kt` / `api/PhonoLabApiServer.kt`
- **Theme colors** → `res/values/colors.xml` / `values-night/colors.xml`
- **Math rendering** → `util/MathHelper.kt` / `adapter/ChatAdapter.kt`
- **API config** → `api/ApiConfig.kt`
- **Storage paths** → `paths.py` / `StorageManager.kt`
- **Safety limits** → `safety.py` / `native/mobile_core.c`
- **Crash handling** → `PhonoLabApp.kt` (crash handler + error reporting)
- **Build config** → `app/build.gradle.kts`
- **Attachments** → `ui/AttachmentBottomSheet.kt` + `ChatFragment.kt`
- **RAG processing** → `RagProcessor.kt`
- **Vision models** → `ModelFamily.kt` + `LlamaRuntime.kt`
- **Error banner** → `ui/ErrorBannerView.kt` + `PhonoLabApp.showError()`
- **API models** → `ui/ModelsFragment.kt` + `PhonoLabStore.getApiModels()`
- **Download helpers** → `util/UiHelpers.kt`

### I need to understand...

- **How models run** → [ANDROID_APP.md](ANDROID_APP.md) "Key Flows"
- **All constants** → [CONSTANTS.md](CONSTANTS.md)
- **File structure** → [FILE_INDEX.md](FILE_INDEX.md)
- **What to edit** → [CONTRIBUTING.md](CONTRIBUTING.md)
- **Error handling layers** → [ANDROID_APP.md](ANDROID_APP.md) "Error Handling"
- **Lifecycle protection** → [ANDROID_APP.md](ANDROID_APP.md) "Lifecycle Protection"
- **Singleton pattern** → [ANDROID_APP.md](ANDROID_APP.md) "Singleton Pattern"
- **Math rendering** → [ANDROID_APP.md](ANDROID_APP.md) "Math Rendering"
- **Attachment flow** → [CONTRIBUTING.md](CONTRIBUTING.md) "Change attachment handling"
