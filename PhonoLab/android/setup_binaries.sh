#!/bin/bash
set -e

JNILIBS="app/src/main/jniLibs"

echo "Fetching latest llama.cpp release info..."

# Use the releases page to find the latest Android asset
# The Android build sometimes lags 1-2 builds behind the tag
# So we search release assets for the actual android tarball name
RELEASE_JSON=$(curl -s \
  -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10")

# Find the first release that actually has an android-arm64 asset
ARM64_URL=$(echo "$RELEASE_JSON" | python3 -c "
import sys, json
releases = json.load(sys.stdin)
for r in releases:
    for a in r.get('assets', []):
        name = a['name']
        if 'android-arm64' in name and name.endswith('.tar.gz'):
            print(a['browser_download_url'])
            sys.exit(0)
print('')
")

if [ -z "$ARM64_URL" ]; then
    echo ""
    echo "ERROR: Could not find android-arm64 asset in recent releases."
    echo "Open this page and find the tarball manually:"
    echo "  https://github.com/ggml-org/llama.cpp/releases"
    echo ""
    echo "Look for a file named:  llama-bXXXX-bin-android-arm64.tar.gz"
    echo "Then run:"
    echo "  wget <url> -O /tmp/llama-arm64.tar.gz"
    echo "  mkdir -p /tmp/llama-arm64 && tar xzf /tmp/llama-arm64.tar.gz -C /tmp/llama-arm64/"
    echo "  cp /tmp/llama-arm64/*.so $JNILIBS/arm64-v8a/"
    echo "  cp /tmp/llama-arm64/llama-server $JNILIBS/arm64-v8a/libllama_server.so"
    exit 1
fi

echo "Found: $ARM64_URL"
echo ""

# Download
mkdir -p /tmp/llama-android-setup
cd /tmp/llama-android-setup
rm -f llama-arm64.tar.gz
rm -rf llama-arm64

echo "Downloading (~70-80MB)..."
wget -q --show-progress "$ARM64_URL" -O llama-arm64.tar.gz

echo "Extracting..."
mkdir -p llama-arm64
tar xzf llama-arm64.tar.gz -C llama-arm64/

echo ""
echo "Contents:"
ls -lh llama-arm64/

# Copy to jniLibs
mkdir -p "$OLDPWD/$JNILIBS/arm64-v8a"

# Find the actual content directory (archive may have a subdirectory)
CONTENT_DIR=$(find llama-arm64 -maxdepth 2 -name "*.so" | head -1 | xargs dirname 2>/dev/null)
if [ -z "$CONTENT_DIR" ]; then
    # Try one level deeper
    CONTENT_DIR=$(find llama-arm64 -mindepth 1 -maxdepth 2 -type d | head -1)
fi

echo "Content dir: $CONTENT_DIR"
echo "Files:"
ls -lh "$CONTENT_DIR/"

# Copy all .so files
SO_COUNT=$(ls "$CONTENT_DIR"/*.so 2>/dev/null | wc -l)
if [ "$SO_COUNT" -eq 0 ]; then
    echo "ERROR: No .so files found. See above."
    exit 1
fi

cp "$CONTENT_DIR"/*.so "$OLDPWD/$JNILIBS/arm64-v8a/"
echo "✅ Copied $SO_COUNT .so files"

# Copy and rename stub binary
if [ -f "$CONTENT_DIR/llama-server" ]; then
    cp "$CONTENT_DIR/llama-server" "$OLDPWD/$JNILIBS/arm64-v8a/libllama_server.so"
    echo "✅ libllama_server.so (stub binary)"
else
    echo "⚠️  llama-server not found. Files in $CONTENT_DIR:"
    ls "$CONTENT_DIR/"
fi

# Cleanup
cd "$OLDPWD"
rm -rf /tmp/llama-android-setup

echo ""
echo "=== jniLibs/arm64-v8a/ ==="
ls -lh "$JNILIBS/arm64-v8a/"
echo ""
echo "✅ Done! Now run: ./gradlew assembleDebug"