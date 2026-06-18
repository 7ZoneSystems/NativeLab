# PhonoLab Android App Architecture

## Overview

PhonoLab Android runs llama.cpp models on-device via a bundled `llama-server` binary. The app uses JNI fork+execve to bypass Android's W^X restrictions, communicates with the server over localhost HTTP, and provides a chat UI with math rendering, session management, attachments, RAG document processing, and a LAN API server.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  PhonoLabApp (Application singleton)                    │
│    store, sessionManager, modelManager, runtime          │
│    showError(fatal) — banner or restart dialog           │
│    safeRunState() — IllegalStateException handler        │
│    └─ survives recreate() (theme switch, config change)  │
├─────────────────────────────────────────────────────────┤
│  MainActivity (implements ChatFragment.Host)             │
│    ├─ Toolbar + DrawerLayout sidebar                    │
│    ├─ Fragments: Chat, Models, Downloads, API           │
│    └─ Session list, search, nav items                   │
├─────────────────────────────────────────────────────────┤
│  ChatFragment                                            │
│    ├─ ChatAdapter (RecyclerView, math WebView)          │
│    ├─ Model picker (AlertDialog with ✓/⬇)              │
│    ├─ Attachment button → Image / Document picker       │
│    ├─ RAG processing with progress bar                  │
│    ├─ Download progress bar                             │
│    └─ Generating banner with stop button                │
├─────────────────────────────────────────────────────────┤
│  LlamaRuntime → LlamaCppManager → JNI runner.cpp        │
│    ├─ load(model) → startServer via fork+execve         │
│    ├─ generate(prompt) → POST /v1/chat/completions      │
│    ├─ nativeLoaded flag guards all JNI calls            │
│    └─ abort() → sets flag, doesn't kill server          │
├─────────────────────────────────────────────────────────┤
│  PhonoLabApiServer (OpenAI + Anthropic compatible)      │
│    ├─ Fixed thread pool (4 threads)                     │
│    ├─ 1MB body size limit                               │
│    └─ Socket write guards for client disconnects        │
├─────────────────────────────────────────────────────────┤
│  nativeLibraryDir/libllama_server.so (bundled binary)   │
│  nativeLibraryDir/librunner.so (JNI wrapper)            │
└─────────────────────────────────────────────────────────┘
```

---

## Singleton Pattern

**All fragments use `PhonoLabApp` singletons.** No fragment creates its own `PhonoLabStore`, `LlamaRuntime`, or `ModelManager`.

```kotlin
val app = requireActivity().application as PhonoLabApp
store = app.store
runtime = app.runtime
modelManager = app.modelManager
```

This ensures model loading in `ModelsFragment` is visible to `ChatFragment`, and the API server in `ApiFragment` uses the same runtime.

---

## Key Flows

### First Launch
1. `MainActivity.onCreate()` → `StorageManager.loadPersistedUri()`
2. No folder → show dialog: "Choose Folder" or "Use Default"
3. "Use Default" → `storageManager.markUsingDefault()` → saves to SharedPreferences
4. `initializeApp()` → uses `PhonoLabApp` singletons
5. ChatFragment → auto-setup if first launch

### Model Loading
1. User taps model picker → `showModelPicker()` (AlertDialog with ✓/⬇)
2. ✓ model → `loadModelFromPath(path)` → `runtime.load(model)`
3. ⬇ model → `downloadAndLoad(candidate)` → green progress bar → auto-load
4. `LlamaRuntime.load()` → looks up `ModelConfig` from `ModelManager` → `startServer(cfg)`
5. Server args use per-model config: ctx, threads, maxTokens
6. Vision models: auto-detects mmproj, passes `--mmproj` flag

### Chat Generation
1. User types prompt, taps send
2. `sendPrompt()` → adds messages to session + adapter
3. If document attached → RAG chunks prepended as context
4. If no model loaded → `findAutoLoadModel()` → auto-load last/first available
5. `runtime.generate(prompt, onToken)` → POST `/v1/chat/completions` (SSE streaming)
6. Tokens appended to adapter + session via `appendToLast()`
7. On completion → `sessionManager.save()` → refresh sidebar via `Host.onSessionChanged()`

### Document Attachment (RAG)
1. User taps attach button → `AttachmentBottomSheet` (Image | Document)
2. Document → `OpenDocument` picker (.pdf, .txt, .doc, .docx)
3. `RagProcessor.processDocument()` — extracts text, chunks (1500 chars, 200 overlap)
4. Progress bar shown during processing
5. On send → `retrieveChunks(query, chunks, topK=3)` → prepended to message

### Stop Generation
1. User taps stop → `runtime.abort()` (sets `_abort` flag)
2. SSE loop checks flag each line → breaks cleanly
3. Returns `[ABORTED]` → session saved, status shows "Stopped"
4. Server stays running (model stays loaded)

### Theme Switch
1. User taps "Change Theme" → `ThemeManager.toggleTheme()` → `recreate()`
2. `onSaveInstanceState` saves `currentFragmentTag`
3. Activity recreated → `PhonoLabApp` singletons survive
4. `initializeApp(savedInstanceState)` restores correct fragment
5. `isFinishing` check prevents runtime kill on recreate

---

## Error Handling

### Fatal Errors (app can't continue)
- `PhonoLabApp` crash handler → `AlertDialog` with "Restart" / "Exit" buttons
- Distinguishes: `IllegalStateException`, `IllegalArgumentException`, `NullPointerException`, `OutOfMemoryError`
- Each gets a specific title and user-friendly message

### Non-Fatal Errors (operation failed, app continues)
- `PhonoLabApp.showError(message, fatal=false)` → red `ErrorBannerView` top-right, auto-dismisses 5s
- `safeRunState(tag, operation, block)` — catches `IllegalStateException`/`IllegalArgumentException`, shows banner
- `ChatFragment.showError()` → error as assistant message with ⚠️

### Error Reporting Layers

| Layer | Component | What it does |
|-------|-----------|--------------|
| Global | `PhonoLabApp.installCrashHandler()` | UncaughtExceptionHandler: log, kill processes, restart dialog |
| Runtime | `LlamaRuntime.load()` | Returns `String?` instead of throwing |
| Runtime | `LlamaRuntime.generate()` | Returns `[ERROR]`/`[ABORTED]` prefix instead of throwing |
| Runtime | `LlamaRuntime._abort` | Volatile flag checked each SSE line |
| JNI | `LlamaCppManager.nativeLoaded` | Flag guards all JNI calls |
| JNI | `LlamaCppManager.startServer()` | Returns -1 on failure (no throw) |
| Download | `LlamaCppManager.downloadAndInstall()` | Top-level try-catch, returns null on failure |
| Download | `SafeDownloader` | `renameTo()` fallback to copy+delete |
| API Server | `PhonoLabApiServer` | 1MB body limit, socket write guards, fixed thread pool |
| Fragment | `ChatFragment.runOnUi{}` | `isAdded` lifecycle guard on all UI updates |
| Fragment | `ChatFragment.Host` | Interface decouples from MainActivity |
| Data | `ChatSession.fromJson()` | `optJSONObject()` instead of `getJSONObject()` |

---

## Lifecycle Protection

### UI Update Safety
All `main.post {}` blocks in fragments use `runOnUi {}` helper:
```kotlin
private fun runOnUi(block: () -> Unit) {
    main.post { if (isAdded) block() }
}
```
Prevents crashes when fragment view is destroyed before posted runnable executes.

### Worker Executor Shutdown
- `DownloadsFragment.onDestroyView()` → `worker.shutdown()`
- `PhonoLabApiServer.stop()` → `executor.shutdownNow()`
- `ChatFragment.onDestroyView()` → conditional shutdown (not during generation)

### Session Log Capping
- `ChatSession.addLog()` — caps at 500 entries, trims oldest
- `ChatFragment.sessionLogs` — capped at 500

---

## Math Rendering

### Detection (MathHelper)
- Explicit: `$...$`, `$$...$$`, `\(...\)`, `\[...\]`
- Auto-detect: `\frac`, `\sqrt`, `\sin`, `x^2`, `a_n`, etc.

### Rendering (ChatAdapter)
- `hasMath()` → true → show WebView with KaTeX
- `wrapAutoMath()` wraps detected math in `$...$`
- WebView loads KaTeX from CDN (jsdelivr.net)
- Theme colors passed to HTML (txt color, code bg)

### Android ICU Note
- NEVER use literal `{` or `}` in regex — use `[{]` and `[}]`
- Use lambda replacements, not backreferences (`$1`)

---

## Build

```bash
cd PhonoLab/android
./setup_binaries.sh        # Download llama.cpp arm64 binaries
./gradlew assembleDebug    # Build APK
adb install app/build/outputs/apk/debug/app-debug.apk
```

### setup_binaries.sh
1. Fetches latest llama.cpp GitHub release
2. Downloads android-arm64 tarball
3. Copies .so to `jniLibs/arm64-v8a/`
4. Renames `llama-server` → `libllama_server.so`

---

## File Map

| File | Role |
|------|------|
| PhonoLabApp.kt | Application singleton, crash handler, error reporting |
| MainActivity.kt | Entry point, drawer, fragments, implements ChatFragment.Host |
| LlamaRuntime.kt | Server lifecycle, generate, abort, vision model support |
| LlamaCppManager.kt | Binary management, JNI bridge, nativeLoaded guard |
| PhonoLabStore.kt | Storage paths, safe file access, API model config |
| StorageManager.kt | SAF folder picker, URI persistence |
| SafeDownloader.kt | HuggingFace downloads with dedup, renameTo fallback |
| ModelCatalog.kt | Built-in model catalog |
| ModelFamily.kt | Chat template detection, VLM detection, mmproj pairing |
| RagProcessor.kt | Document RAG: PDF/text/docx extraction, chunking, retrieval |
| ChatSession.kt | ChatMessage (with imageBase64), ChatSession, buildMessages, addLog cap |
| SessionManager.kt | Session CRUD (JSON files) |
| ModelManager.kt | ModelConfig + registry CRUD |
| ApiConfig.kt | Server config, key generation |
| PhonoLabApiServer.kt | HTTP server (OpenAI + Anthropic + SSE), body size limit |
| ThemeManager.kt | Dark/light palette management |
| MathHelper.kt | Math detection + KaTeX wrapping |
| ChatFragment.kt | Chat UI, model picker, attachments, RAG, Host interface |
| ModelsFragment.kt | Model list + parameter editor + API models |
| DownloadsFragment.kt | Runtime install + catalog, worker shutdown |
| ApiFragment.kt | API server control panel, captured lambdas |
| ErrorBannerView.kt | Red exclamation banner (non-fatal errors) |
| AttachmentBottomSheet.kt | Image/Document picker bottom sheet |
| ChatAdapter.kt | Chat messages with math WebView |
| SessionAdapter.kt | Session list with date grouping |
| ModelAdapter.kt | Model list items |
| CatalogAdapter.kt | Catalog download items |
| ApiModelAdapter.kt | API model list items |
| UiHelpers.kt | calcPercent() download progress helper |
| runner.cpp | JNI fork+execve wrapper |

---

## Permissions

| Permission | Why |
|------------|-----|
| INTERNET | HF downloads, API server |
| ACCESS_NETWORK_STATE | Network detection |
| READ_MEDIA_* | File access (Android 13+) |
| POST_NOTIFICATIONS | Download progress |
| FOREGROUND_SERVICE | Background model loading |
| WAKE_LOCK | Keep alive during generation |
