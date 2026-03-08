#!/bin/bash
# build.sh — One-click build for Native Lab Pro (Linux)
# Usage: chmod +x build.sh && ./build.sh

set -e

echo ""
echo "╔══════════════════════════════════════╗"
echo "║   Native Lab Pro — PyInstaller Build ║"
echo "║              Linux                   ║"
echo "╚══════════════════════════════════════╝"
echo ""

# ── Check dependencies ────────────────────────────────────────────────────────
echo "[INFO] Checking dependencies..."

python3 -m PyInstaller --version > /dev/null 2>&1 || {
    echo "[INFO] PyInstaller not found. Installing into venv..."
    python3 -m pip install pyinstaller
}

# Check PyQt6
python3 -c "import PyQt6" > /dev/null 2>&1 || {
    echo "[ERROR] PyQt6 not found. Install with:"
    echo "        pip install PyQt6"
    exit 1
}

# Check llama-bin
if [ ! -f "llama/bin/llama-cli" ]; then
    echo "[ERROR] llama-bin/llama-cli not found."
    echo "        Download Linux build from:"
    echo "        https://github.com/ggerganov/llama.cpp/releases"
    echo "        Look for: llama-*-bin-ubuntu-x64.zip"
    echo "        Extract into llama-bin/ folder"
    exit 1
fi

# Make llama binaries executable
echo "[INFO] Setting execute permissions on llama binaries..."
chmod +x llama/bin/llama-cli
chmod +x llama/bin/llama-server 2>/dev/null || true

# ── Clean previous build ──────────────────────────────────────────────────────
echo "[INFO] Cleaning previous build..."
rm -rf dist/NativeLabPro build/ 2>/dev/null || true

# ── Run PyInstaller ───────────────────────────────────────────────────────────
echo "[INFO] Building with PyInstaller..."
python3 -m PyInstaller NativeLab.spec --noconfirm --clean

# ── Post-build setup ──────────────────────────────────────────────────────────
echo "[INFO] Creating required data folders..."
mkdir -p dist/NativeLabPro/localllm
mkdir -p dist/NativeLabPro/sessions
mkdir -p dist/NativeLabPro/chat_refs
mkdir -p dist/NativeLabPro/ref_cache
mkdir -p dist/NativeLabPro/ref_index
mkdir -p dist/NativeLabPro/paused_jobs

# Make the output executable
chmod +x dist/NativeLabPro/NativeLabPro

# ── Create launcher script ────────────────────────────────────────────────────
cat > dist/NativeLabPro/run.sh << 'EOF'
#!/bin/bash
# Launcher — sets correct working directory before running
cd "$(dirname "$0")"
./NativeLabPro "$@"
EOF
chmod +x dist/NativeLabPro/run.sh

# ── Create .desktop file ─────────────────────────────────────────────────────
cat > dist/NativeLabPro/NativeLabPro.desktop << 'EOF'
[Desktop Entry]
Version=1.0
Type=Application
Name=Native Lab Pro
Comment=Local LLM Desktop
Exec=./run.sh
Icon=icon
Terminal=false
Categories=Science;Education;
EOF

echo ""
echo "✅  Build complete!"
echo "    Output: dist/NativeLabPro/"
echo ""
echo "Next steps:"
echo "  1. Copy your .gguf model into:  dist/NativeLabPro/localllm/"
echo "  2. Run the app:                 cd dist/NativeLabPro && ./run.sh"
echo "  3. Or double-click:             dist/NativeLabPro/NativeLabPro.desktop"
echo ""