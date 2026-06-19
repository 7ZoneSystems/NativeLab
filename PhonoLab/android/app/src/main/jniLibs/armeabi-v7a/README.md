# jniLibs/armeabi-v7a - 32-bit llama.cpp binaries

## Optional - for older/budget devices

Only needed if you want to support 32-bit ARM devices.

```bash
TAG=$(grep tag_name /tmp/release.json | cut -d'"' -f4)
wget "https://github.com/ggml-org/llama.cpp/releases/download/${TAG}/llama-${TAG}-bin-android-armeabi-v7a.tar.gz"
unzip "llama-${TAG}-bin-android-armeabi-v7a.tar.gz" -d llama-arm32/
cp llama-arm32/*.so app/src/main/jniLibs/armeabi-v7a/
cp llama-arm32/llama-server app/src/main/jniLibs/armeabi-v7a/libllama_server.so
```

## Note

If no 32-bit release is available, remove `armeabi-v7a` from `abiFilters` in build.gradle.kts.
The app will still work on all 64-bit devices (arm64-v8a covers ~95% of active devices).
