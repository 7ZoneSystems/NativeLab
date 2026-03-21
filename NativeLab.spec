# -*- mode: python ; coding: utf-8 -*-
# NativeLab.spec — PyInstaller build for Native Lab Pro v2
# Build command:  pyinstaller NativeLab.spec

import sys
from pathlib import Path

block_cipher = None

a = Analysis(
    ['NativeLab/main.py'],           # ← rename to your actual .py filename
    pathex=['.'],
    binaries=[
        # ── llama.cpp binaries (Linux) ─────────────────────────────────────
        ('llama/bin/llama-cli',    'llama-bin'),
        ('llama/bin/llama-server', 'llama-bin'),
    ],
    datas=[
        # ── Icons ──────────────────────────────────────────────────────────
        ('icon.ico', '.'),
        ('icon.png', '.'),

        # ── App config (if exists already) ─────────────────────────────────
        # ('app_config.json', '.'),   # uncomment if you want to ship defaults
    ],
    hiddenimports=[
        # PyQt6 modules that PyInstaller misses
        'PyQt6.QtWidgets',
        'PyQt6.QtGui',
        'PyQt6.QtCore',
        'PyQt6.sip',
        # Optional deps
        'psutil',
        'PyPDF2',
        # stdlib that sometimes gets missed in frozen builds
        'multiprocessing',
        'multiprocessing.pool',
        'concurrent.futures',
        'hashlib',
        'pickle',
        'ast',
        'socket',
        'http.client',
        'json',
        'pathlib',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,       # onedir mode — binaries stay separate
    name='NativeLabPro',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,                    # compress — set False if UPX not installed
    console=False,               # no black console window on launch
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico',             # app icon (Windows .ico)
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[
        'vcruntime140.dll',      # never UPX these — breaks them
        'python3*.dll',
        'Qt6*.dll',
    ],
    name='NativeLabPro',         # output folder name inside dist/
)
