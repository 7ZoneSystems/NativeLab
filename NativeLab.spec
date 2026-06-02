# -*- mode: python ; coding: utf-8 -*-
# NativeLab.spec - PyInstaller build for NativeLab
# Build command:  pyinstaller NativeLab.spec

import sys
import re
from pathlib import Path

block_cipher = None
exe_icon = 'nativelab/icon.ico' if sys.platform == 'win32' else None
bundle_icon = 'nativelab/icon.icns'


def project_version():
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    try:
        import tomllib
    except ModuleNotFoundError:
        tomllib = None
    try:
        if tomllib is not None:
            data = tomllib.loads(text)
            return str(data.get("project", {}).get("version") or "0.0.0")
    except Exception:
        pass
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', text)
    return match.group(1) if match else "0.0.0"


def package_datas():
    """Mirror [tool.setuptools.package-data] from pyproject.toml."""
    patterns = [
        ("nativelab/*.png", "nativelab"),
        ("nativelab/*.ico", "nativelab"),
        ("nativelab/assets/icons/*.svg", "nativelab/assets/icons"),
        ("nativelab/assets/katex/LICENSE", "nativelab/assets/katex"),
        ("nativelab/assets/katex/*.css", "nativelab/assets/katex"),
        ("nativelab/assets/katex/*.js", "nativelab/assets/katex"),
        ("nativelab/assets/katex/fonts/*", "nativelab/assets/katex/fonts"),
        ("nativelab/integrations/examples/*", "nativelab/integrations/examples"),
    ]
    datas = []
    for pattern, dest in patterns:
        for path in sorted(Path(".").glob(pattern)):
            if path.is_file() and "__pycache__" not in path.parts:
                datas.append((str(path), dest))
    return datas


a = Analysis(
    ['nativelab/main.py'],
    pathex=['.'],
    # Do not bundle prebuilt llama.cpp runtime binaries. Users can install
    # them into ./llama/bin/ or configure custom paths.
    binaries=[],
    datas=package_datas(),
    hiddenimports=[
        # PyQt6 modules that PyInstaller misses
        'PyQt6.QtWidgets',
        'PyQt6.QtGui',
        'PyQt6.QtCore',
        'PyQt6.sip',
        # Optional deps
        'psutil',
        'PyPDF2',
        'aiohttp',
        'aiohttp.web',
        'discord',
        'discord.ext.commands',
        # Modules launched dynamically by integrations
        'nativelab.integrations.examples.discord_bot',
        'nativelab.integrations.examples.whatsapp_bot',
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
    exclude_binaries=True,       # onedir mode - binaries stay separate
    name='NativeLab',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,                    # compress - set False if UPX not installed
    console=False,               # no black console window on launch
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=exe_icon,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[
        'vcruntime140.dll',      # never UPX these - breaks them
        'python3*.dll',
        'Qt6*.dll',
    ],
    name='NativeLab',
)

if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='NativeLab.app',
        icon=bundle_icon if Path(bundle_icon).exists() else None,
        bundle_identifier='ai.nativelab.app',
        info_plist={
            'CFBundleDisplayName': 'NativeLab',
            'CFBundleName': 'NativeLab',
            'CFBundleShortVersionString': project_version(),
            'CFBundleVersion': project_version(),
            'LSMinimumSystemVersion': '12.0',
            'NSAppTransportSecurity': {
                'NSAllowsLocalNetworking': True,
            },
            'NSHighResolutionCapable': True,
        },
    )
