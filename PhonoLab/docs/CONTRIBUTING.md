# PhonoLab Contributor Quick Reference

What to edit when you need to change specific things.

---

## "I want to change..."

### Model defaults (context, threads, tokens, temperature)
- **Python:** `config.py` → `MobileConfig` defaults
- **Android:** `data/ModelManager.kt` → `ModelConfig` defaults + `fromJson()`
- **Bounds:** `config.py` → `from_dict()` clamping, `safety.py` → `sampler_args()`
- **Applied at runtime:** `LlamaRuntime.startServer()` reads from `loadedConfig`

### Add a new model to the catalog
- **Python:** `small_models.py` → `SMALL_MODEL_CATALOG`
- **Android:** `ModelCatalog.kt` → `ModelCatalog.items`
- Keep both in sync! Same keys, repos, min_ram_mb.

### Change the chat prompt format
- **Android:** `data/ChatSession.kt` → `buildMessages()` / `buildRequestBody()`
- **API server:** `api/PhonoLabApiServer.kt` → `messagesToPrompt()`
- **Python:** `engine/llama_cpp.py` → `_command()`

### Change theme colors
- **Android XML:** `res/values/colors.xml` (light) / `values-night/colors.xml` (dark)
- **ThemeManager:** `theme/ThemeManager.kt` → `Dark` / `Light` objects
- **Code access:** `ThemeManager.color(context, R.color.ph_xxx)` or `ThemeManager.hexColor("key")`
- **Python Kivy:** `ui/kivy_app.py` → `C` dict

### Change API server defaults
- **Port/host:** `api/ApiConfig.kt` → constructor defaults
- **Endpoints:** `api/PhonoLabApiServer.kt` → `handleClient()` routing
- **Streaming:** `handleOpenAiChatStreaming()` → SSE chunk format

### Change download limits
- **Python:** `config.py` → `max_download_bytes` (7 GB)
- **Android:** `SafeDownloader.kt` → `maxDownloadBytes` (7 GB)

### Change storage paths
- **Python:** `paths.py` → `default_home()`
- **Android:** `StorageManager.kt` → `getBaseDir()` + subdir methods
- **Android store:** `PhonoLabStore.kt` → directory properties

### Change llama-server startup args
- **Android:** `LlamaRuntime.kt` → `startServer()` - reads from ModelConfig
- **Python:** `engine/llama_cpp.py` → `_command()`

### Change math detection
- `util/MathHelper.kt` → regex patterns (DELIMITED_MATH, LATEX_CMD, EXPONENT, etc.)
- `wrapAutoMath()` - lambda replacements, NOT backreferences (Android ICU)
- NEVER use literal `{` or `}` in regex - use `[{]` and `[}]`

### Change crash handling
- **Global:** `PhonoLabApp.kt` → `installCrashHandler()`
- **Runtime:** `LlamaRuntime.kt` → `load()` returns, `generate()` returns, `_abort` flag
- **JNI:** `LlamaCppManager.kt` → try-catch around nativeExec/kill/waitPid
- **Fragment:** `ChatFragment.kt` → `showError()`, `sendPrompt()` error handling

### Change model picker
- `ui/ChatFragment.kt` → `showModelPicker()` - AlertDialog with custom adapter
- Layout: `res/layout/ph_spinner_dropdown_item.xml`

### Add a new Android fragment
1. Create `ui/NewFragment.kt`
2. Create `res/layout/fragment_new.xml`
3. Add nav item in `activity_main.xml` sidebar
4. Wire in `MainActivity.kt` → `setupSidebar()`
5. Add to `currentFragmentTag` in `navigateTo()` + `initializeApp()`

### Add a new API endpoint
1. Add route in `PhonoLabApiServer.kt` → `handleClient()`
2. Add handler method
3. For streaming: use `handleOpenAiChatStreaming()` pattern (guard socket writes)

### Change attachment handling
- **Picker UI**: `ui/AttachmentBottomSheet.kt` - add options to `bottom_sheet_attachment.xml`
- **Image picker**: `ChatFragment.kt` → `imagePickerLauncher` (PickVisualMedia fallback GetContent)
- **Document picker**: `ChatFragment.kt` → `documentPickerLauncher` (OpenDocument)
- **RAG processing**: `RagProcessor.kt` - extraction, chunking, retrieval
- **Preview chip**: `fragment_chat.xml` → `attachment_chip` layout
- **Message injection**: `ChatFragment.sendPrompt()` - RAG chunks prepended to message

### Change error handling
- **Fatal errors**: `PhonoLabApp.kt` → `installCrashHandler()` - AlertDialog with Restart/Exit
- **Non-fatal errors**: `PhonoLabApp.showError()` → `ErrorBannerView` (red, auto-dismiss 5s)
- **State errors**: `PhonoLabApp.safeRunState()` - catches ISE/IAE, shows banner
- **Lifecycle guards**: `runOnUi {}` in fragments - `isAdded` check before UI ops

---

## File Relationships

### Python dependency chain
```
config.py ←── paths.py (CONFIG_FILE)
registry.py ←── config.py, hardware.py, paths.py, safety.py
engine/llama_cpp.py ←── config.py, llama_cpp_setup.py, registry.py, safety.py
ui/kivy_app.py ←── config.py, downloads.py, engine, hardware.py, llama_cpp_setup.py, registry.py, safety.py, small_models.py
```

### Android dependency chain
```
PhonoLabApp (singletons)
  ├─ PhonoLabStore ← StorageManager
  ├─ SessionManager ← PhonoLabStore
  ├─ ModelManager ← PhonoLabStore
  └─ LlamaRuntime ← PhonoLabStore, ModelManager
       └─ LlamaCppManager ← PhonoLabStore (nativeLoaded guard)
            └─ runner.cpp (JNI)

MainActivity ← PhonoLabApp singletons, implements ChatFragment.Host
  └─ Fragments ← PhonoLabApp singletons (via application cast)
       ├─ ChatFragment → LlamaRuntime, SessionManager, ModelManager, ChatAdapter
       │    ├─ AttachmentBottomSheet → Image/Document picker
       │    ├─ RagProcessor → document extraction + chunking
       │    └─ Host.onSessionChanged() → refreshes sidebar
       ├─ ModelsFragment → ModelManager, LlamaRuntime, ApiModelAdapter
       ├─ DownloadsFragment → LlamaRuntime, SafeDownloader, worker.shutdown()
       └─ ApiFragment → PhonoLabApiServer, LlamaRuntime (captured lambdas)

util/UiHelpers - calcPercent() shared helper
ui/ErrorBannerView - non-fatal error banner (auto-dismiss)
```

---

## Testing

```bash
# Python tests
cd PhonoLab
python -m pytest tests/ -v

# Android project structure test
python -m pytest tests/test_android_project.py -v
```

---

## Common Gotchas

1. **Android ICU regex** - NEVER use literal `{`/`}` in patterns. Use `[{]`/`[}]`. Use lambda replacements, not `$1`.
2. **Singletons** - `PhonoLabApp` holds singletons. Don't create new instances in fragments. Use `(requireActivity().application as PhonoLabApp).store` etc.
3. **load() returns String?** - null=success, non-null=error. Don't use try-catch for control flow.
4. **generate() returns String** - check for `[ERROR]` or `[ABORTED]` prefix.
5. **abort() vs unload()** - abort() stops generation but keeps model loaded. unload() kills server.
6. **Theme switch** - `recreate()` is safe because PhonoLabApp singletons survive.
7. **Default directory** - `StorageManager.markUsingDefault()` must be called for persistence.
8. **Model dedup** - `ModelManager.syncDiscovery()` uses `canonicalPath` to avoid double entries.
9. **JNI argv[0]** - runner.cpp prepends binary path automatically.
10. **LD_LIBRARY_PATH** - MUST be exactly nativeLibraryDir.
11. **Lifecycle guards** - Always use `runOnUi {}` (checks `isAdded`) instead of raw `main.post {}`.
12. **Thread safety** - Use `ThreadLocal<SimpleDateFormat>` for date formatting in multi-threaded code.
13. **JSON parsing** - Use `optJSONObject()` / `optString()` instead of `getJSONObject()` / `getString()` to avoid crashes on corrupt data.
14. **Cross-routing** - Use `ChatFragment.Host` interface, never cast directly to `MainActivity`.
15. **Session logs** - Capped at 500 entries via `ChatSession.addLog()`. Don't append to `logs` directly.
16. **renameTo()** - Can fail on Android cross-filesystem. Use copy+delete fallback.
17. **nativeLoaded** - Check `LlamaCppManager.nativeLoaded` before JNI calls. It's false if librunner.so failed to load.
18. **API body limit** - PhonoLabApiServer rejects bodies > 1MB (413 response).
19. **Executor shutdown** - Always call `worker.shutdown()` in `onDestroyView()` for fragment executors.
20. **Error reporting** - Use `PhonoLabApp.showError(msg, fatal)` for app-wide errors. Fatal shows dialog, non-fatal shows banner.
