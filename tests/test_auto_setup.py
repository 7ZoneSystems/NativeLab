import unittest

from nativelab.core.auto_setup import (
    BACKEND_HF_TRANSFORMERS,
    HardwareProfile,
    choose_gguf_file,
    choose_model_for_hardware,
    choose_runtime_asset,
    normalize_setup_backend,
    normalize_auto_setup_state,
    setup_backend_label,
)
from nativelab.Server.hfdwld import _safe_child_path


class AutoSetupPlanningTests(unittest.TestCase):
    def test_low_memory_uses_minimal_model(self):
        hw = HardwareProfile(
            os_name="Linux",
            arch="x86_64",
            cpu_threads=4,
            ram_total_mb=4096,
            ram_available_mb=1800,
            ram_used_mb=2296,
            gpus=[],
        )
        self.assertEqual(choose_model_for_hardware(hw).key, "tinyllama-1b")

    def test_cuda_machine_gets_high_tier_model(self):
        hw = HardwareProfile(
            os_name="Linux",
            arch="x86_64",
            cpu_threads=16,
            ram_total_mb=32768,
            ram_available_mb=24000,
            ram_used_mb=8768,
            gpus=[{"idx": 0, "name": "RTX", "type": "cuda", "vram_mb": 12288, "vram_free_mb": 11000}],
        )
        self.assertEqual(choose_model_for_hardware(hw).key, "llama31-8b")

    def test_backend_choice_changes_model_catalog(self):
        hw = HardwareProfile(
            os_name="Linux",
            arch="x86_64",
            cpu_threads=16,
            ram_total_mb=32768,
            ram_available_mb=24000,
            ram_used_mb=8768,
            gpus=[{"idx": 0, "name": "RTX", "type": "cuda", "vram_mb": 12288, "vram_free_mb": 11000}],
        )
        self.assertEqual(choose_model_for_hardware(hw, BACKEND_HF_TRANSFORMERS).key, "qwen25-7b-hf")

    def test_backend_aliases_normalize(self):
        self.assertEqual(normalize_setup_backend("hugging face"), BACKEND_HF_TRANSFORMERS)
        self.assertIn("hard", setup_backend_label("hf"))

    def test_state_normalization_preserves_resumable_stage(self):
        state = normalize_auto_setup_state({"stage": "model_download", "backend": "hugging-face"})
        self.assertEqual(state["status"], "running")
        self.assertEqual(state["backend"], BACKEND_HF_TRANSFORMERS)
        self.assertGreaterEqual(state["version"], 2)

    def test_runtime_asset_prefers_matching_accelerator(self):
        releases = [{
            "tag": "b9999",
            "assets": [
                {"name": "llama-b9999-bin-ubuntu-x64.zip", "url": "cpu", "size": 1},
                {"name": "llama-b9999-bin-ubuntu-vulkan-x64.zip", "url": "vk", "size": 1},
                {"name": "llama-b9999-bin-ubuntu-cuda-cu12-x64.zip", "url": "cu", "size": 1},
            ],
        }]
        self.assertEqual(choose_runtime_asset(releases, "cuda")["url"], "cu")
        self.assertEqual(choose_runtime_asset(releases, "vulkan")["url"], "vk")
        self.assertEqual(choose_runtime_asset(releases, "cpu")["url"], "cpu")

    def test_cpu_runtime_does_not_pick_gpu_only_asset(self):
        releases = [{
            "tag": "b9999",
            "assets": [
                {"name": "llama-b9999-bin-ubuntu-vulkan-x64.zip", "url": "vk", "size": 1},
                {"name": "llama-b9999-bin-ubuntu-cuda-cu12-x64.zip", "url": "cu", "size": 1},
            ],
        }]
        self.assertIsNone(choose_runtime_asset(releases, "cpu"))

    def test_gguf_file_prefers_quant_and_skips_projectors(self):
        choice = choose_model_for_hardware(HardwareProfile(
            os_name="Linux",
            arch="x86_64",
            cpu_threads=8,
            ram_total_mb=8192,
            ram_available_mb=5000,
            ram_used_mb=3192,
            gpus=[],
        ))
        files = [
            {"rfilename": "model-mmproj.gguf", "size": 100},
            {"rfilename": "model.Q2_K.gguf", "size": 500},
            {"rfilename": "model.Q4_K_M.gguf", "size": 900},
        ]
        self.assertEqual(choose_gguf_file(files, choice)["rfilename"], "model.Q4_K_M.gguf")

    def test_download_paths_stay_inside_destination(self):
        self.assertEqual(str(_safe_child_path("/tmp/models", "nested/model.gguf")), "/tmp/models/nested/model.gguf")
        with self.assertRaises(ValueError):
            _safe_child_path("/tmp/models", "../model.gguf")


if __name__ == "__main__":
    unittest.main()
