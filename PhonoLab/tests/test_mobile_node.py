from pathlib import Path

import pytest

from PhonoLab.config import MobileConfig
from PhonoLab.downloads import HfFile, choose_gguf_file
from PhonoLab.paths import safe_child_path
from PhonoLab.registry import MobileModelRegistry
from PhonoLab.safety import ContextLimitError, SafetyError, guard_prompt, validate_repo_id
from PhonoLab.small_models import SMALL_MODEL_CATALOG


def test_phonolab_safe_child_path_rejects_traversal(tmp_path):
    assert safe_child_path(tmp_path, "models/tiny.gguf") == tmp_path / "models" / "tiny.gguf"

    with pytest.raises(ValueError):
        safe_child_path(tmp_path, "../escape.gguf")

    with pytest.raises(ValueError):
        safe_child_path(tmp_path, "/tmp/escape.gguf")


def test_phonolab_repo_validation():
    assert validate_repo_id("owner/model-name") == "owner/model-name"

    with pytest.raises(SafetyError):
        validate_repo_id("../bad")

    with pytest.raises(SafetyError):
        validate_repo_id("missing-owner")


def test_phonolab_context_guard_raises_before_runtime():
    cfg = MobileConfig(ctx=32, n_predict=24, max_prompt_chars=10000)
    prompt = " ".join(["token"] * 80)

    with pytest.raises(ContextLimitError):
        guard_prompt(prompt, cfg)


def test_phonolab_choose_gguf_prefers_safe_quant():
    candidate = SMALL_MODEL_CATALOG[0]
    selected = choose_gguf_file(
        [
            HfFile("model-mmproj-Q4_K_M.gguf", 10),
            HfFile("model-Q2_K.gguf", 5),
            HfFile("model-Q4_K_M.gguf", 6),
        ],
        candidate,
    )

    assert selected.name == "model-Q4_K_M.gguf"


def test_phonolab_registry_round_trip(tmp_path, monkeypatch):
    import PhonoLab.registry as registry_mod
    import PhonoLab.safety as safety_mod

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    model_path = models_dir / "tiny.Q4_K_M.gguf"
    model_path.write_bytes(b"GGUF")

    monkeypatch.setattr(safety_mod, "MODELS_DIR", models_dir)
    monkeypatch.setattr(registry_mod, "MODELS_DIR", models_dir)

    registry_path = tmp_path / "registry.json"
    registry = MobileModelRegistry(registry_path)
    cfg = registry.add(model_path, repo="owner/model", quant="Q4_K_M", set_active=False)

    assert cfg.name == "tiny.Q4_K_M.gguf"
    loaded = MobileModelRegistry(registry_path).all()
    assert len(loaded) == 1
    assert Path(loaded[0].path) == model_path
