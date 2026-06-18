# PhonoLab Android Studio App

Open this directory directly in Android Studio.

## Features

### ChatGPT-style UI
- **Bottom input bar** with send button and attachment button
- **Collapsible sidebar** with chat history grouped by date
- **New Chat** button in sidebar header
- **Search** conversations
- **Long-press** sessions for rename/delete/export

### Document & Image Attachments
- **Attachment button** (paperclip) left of model picker
- **Image picker**: gallery (PhotoPicker on Android 13+, fallback to file picker)
- **Document picker**: .pdf, .txt, .doc, .docx files
- **RAG processing**: extracts text, chunks (1500 chars), keyword-based retrieval
- **Progress bar** shown during document processing
- **Preview chip** with filename and clear button

### Light/Dark Theme (NativeLab-inspired)
- **Dark**: Studio dark palette (`#09090d` bg, `#55C2A4` teal accent)
- **Light**: Cream & sage palette (`#fdf6f0` bg, `#55C2A4` teal accent)
- Toggle via sidebar button (sun/moon icon)
- System-aware (follows device theme by default)

### Model Management (Models tab)
- **Model library**: list all downloaded GGUF models
- **Per-model parameters**: Context, Threads, Temperature, Top-P, Top-K, Repeat Penalty, Max Tokens
- **Load/Remove** models
- **Parameter validation** with warnings
- **Model registry**: persists config in `model_registry.json`
- **API Models section**: configure external API providers (OpenAI, etc.)

### Downloads (Downloads tab)
- **llama.cpp runtime**: auto-install binary or pull source
- **Model catalog**: download small GGUF models (SmolLM2, Qwen, Llama 3.2, TinyLlama)
- **Progress tracking** with percentage

### Chat History
- **Session persistence**: saved as JSON files in `sessions/` directory
- **Auto-title** from first user message
- **Date-grouped** in sidebar (Today, Yesterday, dates)
- **Export** to Markdown via share intent
- **Search** across all sessions
- **Log capping**: max 500 log entries per session

### Error Handling
- **Fatal errors**: restart dialog with "Restart" or "Exit" buttons
- **Non-fatal errors**: red exclamation banner (auto-dismisses 5s)
- **State errors**: caught by `safeRunState()`, shows banner
- **Lifecycle guards**: all UI updates check `isAdded` before touching views

### Responsive Layout
- **Phone**: 280dp sidebar, full-width content
- **Tablet** (sw600dp): 320dp sidebar, 600dp max content width centered

## Architecture

```
org.nativelab.phonolab/
├── PhonoLabApp.kt           (Singletons, crash handler, error reporting)
├── MainActivity.kt          (DrawerLayout + sidebar, implements ChatFragment.Host)
├── LlamaRuntime.kt          (Server lifecycle, generate, abort, vision support)
├── LlamaCppManager.kt       (JNI bridge, nativeLoaded guard)
├── PhonoLabStore.kt         (Storage paths, API model config)
├── StorageManager.kt        (SAF folder picker)
├── SafeDownloader.kt        (HuggingFace downloads with resume)
├── ModelCatalog.kt          (Built-in model catalog)
├── ModelFamily.kt           (Chat template + VLM detection)
├── RagProcessor.kt          (Document RAG: extract, chunk, retrieve)
├── data/
│   ├── ChatSession.kt       (Session + Message + imageBase64 + log cap)
│   ├── SessionManager.kt    (JSON persistence)
│   └── ModelManager.kt      (GGUF registry + per-model config)
├── api/
│   ├── ApiConfig.kt         (Server config, key generation)
│   └── PhonoLabApiServer.kt (HTTP server, body limit, socket guards)
├── ui/
│   ├── ChatFragment.kt      (Chat + attachments + RAG + Host interface)
│   ├── ModelsFragment.kt    (Model library + params + API models)
│   ├── DownloadsFragment.kt (Runtime + model downloads)
│   ├── ApiFragment.kt       (API server control panel)
│   ├── ErrorBannerView.kt   (Red exclamation banner)
│   └── AttachmentBottomSheet.kt (Image/Document picker)
├── adapter/
│   ├── ChatAdapter.kt       (Message bubbles + math WebView)
│   ├── SessionAdapter.kt    (Date-grouped session list)
│   ├── ModelAdapter.kt      (Model list)
│   ├── CatalogAdapter.kt    (Download catalog)
│   └── ApiModelAdapter.kt   (API model list)
├── theme/ThemeManager.kt    (Light/dark palette system)
├── util/
│   ├── MathHelper.kt        (Math detection + KaTeX wrapping)
│   └── UiHelpers.kt         (calcPercent helper)
└── runner.cpp               (JNI fork+execve wrapper)
```

## Build & Run

### Android Studio (Recommended)
```
File → Open → PhonoLab/android/ → OK
Run → Run 'app'  (Shift+F10)
```

### Command Line
```bash
cd PhonoLab/android

# Generate wrapper (first time only, or use Android Studio)
gradle wrapper --gradle-version 8.9

# Build debug APK
./gradlew assembleDebug

# Install on connected device
./gradlew installDebug

# Run unit tests
./gradlew test

# Lint check
./gradlew lint
```

### APK Location
```
PhonoLab/android/app/build/outputs/apk/debug/app-debug.apk
```

## Dependencies
- `com.google.android.material:material:1.12.0` (Material3)
- `androidx.drawerlayout:drawerlayout:1.2.0`
- `androidx.recyclerview:recyclerview:1.3.2`
- `androidx.cardview:cardview:1.0.0`
- `androidx.fragment:fragment-ktx:1.8.5`
- `org.json:json:20231013`

## Permissions
- `INTERNET` - model/runtime downloads, API server
- `ACCESS_NETWORK_STATE` - connectivity check
- `READ_MEDIA_*` - file access (Android 13+)

All data stays in app-private storage.
