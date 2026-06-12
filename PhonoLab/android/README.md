# PhonoLab Android Studio App

Open this directory directly in Android Studio.

This Android app is standalone from the root NativeLab package. It provides a
full GUI for:

- **Auto-setup**: downloads and installs llama-cli binary on first launch
- pulling llama.cpp source from GitHub (fallback)
- downloading mobile-sized GGUF models from Hugging Face
- registering models in app-private storage
- loading a bundled or installed `llama-cli`
- running local chat through llama.cpp

## UI Theme

PhonoLab uses a dark theme inspired by NativeLab's studio palette:

- **Background**: `#09090d` (deep dark)
- **Surface**: `#1e1e2e` (card backgrounds)
- **Accent**: `#55C2A4` (PhonoLab teal)
- **Text**: `#ededf5` / `#7a7a9a` / `#48485e` (three tiers)

The UI is fully responsive across phone screen sizes with card-based layout
sections for Runtime, Model, and Chat.

## Runtime Binary

The app **auto-installs** the llama-cli binary for Android arm64 on first
launch. If the download fails, it falls back to pulling llama.cpp source.

Supported binary locations (checked in order):

1. packaged native executable: `app/src/main/jniLibs/arm64-v8a/libllama-cli.so`
2. bundled asset: `app/src/main/assets/runtimes/android-arm64/llama-cli`
3. auto-downloaded: `<filesDir>/PhonoLab/runtime/bin/llama-cli`

## Source Sync

Android Studio Gradle task:

```bash
./gradlew syncLlamaCpp
```

This clones or updates llama.cpp under `android/external/llama.cpp`.

## Permissions

Only network permissions are declared:

- `INTERNET`
- `ACCESS_NETWORK_STATE`

Models and runtime files stay in app-private storage, so broad storage
permissions are intentionally not requested.

## Dependencies

- `com.google.android.material:material:1.12.0` (Material3 theming)
- `androidx.core:core-ktx:1.15.0`
- `androidx.appcompat:appcompat:1.7.0`
- `org.json:json:20231013`
