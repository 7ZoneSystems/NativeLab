# PhonoLab Android Studio App

Open this directory directly in Android Studio.

## Features

### ChatGPT-style UI
- **Bottom input bar** with send button
- **Collapsible sidebar** with chat history grouped by date
- **New Chat** button in sidebar header
- **Search** conversations
- **Long-press** sessions for rename/delete/export

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

### Responsive Layout
- **Phone**: 280dp sidebar, full-width content
- **Tablet** (sw600dp): 320dp sidebar, 600dp max content width centered

## Architecture

```
org.nativelab.phonolab/
├── MainActivity.kt          (DrawerLayout + sidebar navigation)
├── theme/ThemeManager.kt    (Light/dark palette system)
├── data/
│   ├── ChatSession.kt       (Session + Message data classes)
│   ├── SessionManager.kt    (JSON persistence)
│   └── ModelManager.kt      (GGUF registry + per-model config)
├── ui/
│   ├── ChatFragment.kt      (Chat + bottom input)
│   ├── ModelsFragment.kt    (Model library + params editor)
│   └── DownloadsFragment.kt (Runtime + model downloads)
├── adapter/
│   ├── ChatAdapter.kt       (Message bubbles)
│   ├── SessionAdapter.kt    (Date-grouped session list)
│   ├── ModelAdapter.kt      (Model list)
│   └── CatalogAdapter.kt    (Download catalog)
├── LlamaRuntime.kt          (llama.cpp binary/source management)
├── PhonoLabStore.kt         (App storage paths)
├── SafeDownloader.kt        (Resumable HTTP downloads)
└── ModelCatalog.kt          (Small model catalog)
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
- `INTERNET` - model/runtime downloads
- `ACCESS_NETWORK_STATE` - connectivity check

All data stays in app-private storage.
