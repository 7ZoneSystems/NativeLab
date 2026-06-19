# jniLibs/arm64-v8a - llama.cpp binaries

## Setup

Download the official llama.cpp Android release and place files here:

```bash
# 1. Download latest release
cd /tmp
wget -q https://api.github.com/repos/ggml-org/llama.cpp/releases/latest -O release.json
TAG=$(grep tag_name release.json | cut -d'"' -f4)
echo "Latest release: $TAG"

# 2. Download arm64 binary
wget "https://github.com/ggml-org/llama.cpp/releases/download/${TAG}/llama-${TAG}-bin-android-arm64.tar.gz"

# 3. Extract
tar xzf ... -C llama-arm64/

# 4. Copy .so files
cp llama-arm64/*.so app/src/main/jniLibs/arm64-v8a/

# 5. Rename stub binary to lib*.so (required by Android)
cp llama-arm64/llama-server app/src/main/jniLibs/arm64-v8a/libllama_server.so

# 6. Verify
ls -lh app/src/main/jniLibs/arm64-v8a/
```

## Expected files

```
libllama_server.so          ~7KB    (stub launcher - dlopen()s impl)
libllama-server-impl.so     ~62MB   (real server)
libllama-common.so          ~82MB   (common utilities)
libllama.so                 ~32MB   (core library)
libggml.so                  ~5MB    (GGML base)
libggml-base.so             ~7MB    (GGML base layer)
libggml-rpc.so              ~5MB    (GGML RPC)
libmtmd.so                  ~11MB   (multimodal)
libggml-cpu-android_*.so    ~4-5MB  (CPU variants)
```

## How it works

1. Gradle packages all `lib*.so` into the APK
2. Android installer extracts them to `/data/app/<pkg>/lib/arm64/` (nativeLibraryDir)
3. App calls `exec(nativeLibraryDir/libllama_server.so)` via JNI
4. Stub dlopen()s impl .so from the same directory - no permissions needed

## Troubleshooting

- **Empty directory**: Run the download script above
- **"Binary not found" error**: Ensure `libllama_server.so` exists (not `llama-server`)
- **"dlopen failed" error**: Ensure ALL .so files are present, not just the stub
- **APK too large**: Remove `x86_64` from `abiFilters` in build.gradle.kts
