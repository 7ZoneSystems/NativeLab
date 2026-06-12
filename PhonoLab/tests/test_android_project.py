from __future__ import annotations

from pathlib import Path


ANDROID_DIR = Path(__file__).resolve().parents[1] / "android"


def test_android_studio_project_files_exist():
    required = [
        "settings.gradle.kts",
        "build.gradle.kts",
        "app/build.gradle.kts",
        "app/src/main/AndroidManifest.xml",
        "app/src/main/java/org/nativelab/phonolab/MainActivity.kt",
        "app/src/main/java/org/nativelab/phonolab/SafeDownloader.kt",
        "app/src/main/java/org/nativelab/phonolab/LlamaRuntime.kt",
        "app/src/main/res/xml/network_security_config.xml",
    ]

    for rel in required:
        assert (ANDROID_DIR / rel).exists(), rel


def test_android_manifest_uses_private_storage_permissions_only():
    text = (ANDROID_DIR / "app/src/main/AndroidManifest.xml").read_text(encoding="utf-8")

    assert "android.permission.INTERNET" in text
    assert "android.permission.ACCESS_NETWORK_STATE" in text
    assert "READ_EXTERNAL_STORAGE" not in text
    assert "WRITE_EXTERNAL_STORAGE" not in text
    assert "MANAGE_EXTERNAL_STORAGE" not in text


def test_android_gradle_has_llama_sync_task():
    text = (ANDROID_DIR / "build.gradle.kts").read_text(encoding="utf-8")

    assert "syncLlamaCpp" in text
    assert "https://github.com/ggml-org/llama.cpp.git" in text
