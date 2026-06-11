import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("NATIVELAB_NO_GUI", "1")


class PipelineStoragePathTests(unittest.TestCase):
    def test_pipeline_dir_is_local_llm(self):
        from nativelab.GlobalConfig.config_global import PIPELINES_DIR

        self.assertEqual(PIPELINES_DIR, Path("localllm") / "pipelines")

    def test_legacy_portable_pipeline_import_redirects_to_local_llm(self):
        from nativelab.GlobalConfig.config_global import PIPELINES_DIR
        from nativelab.core.data_portability import _target_path

        self.assertEqual(
            _target_path("~/.native_lab/pipelines/example.json"),
            PIPELINES_DIR / "example.json",
        )

    def test_pipeline_names_cannot_escape_pipeline_dir(self):
        import nativelab.pipelinebuilder.pipefunctions as pipefunctions
        from nativelab.pipelinebuilder.pipefunctions import pipeline_path, save_pipeline

        old_dir = pipefunctions.PIPELINES_DIR
        with tempfile.TemporaryDirectory() as tmp:
            try:
                root = Path(tmp)
                pipefunctions.PIPELINES_DIR = root / "pipelines"
                save_pipeline("../escape.json", [], [])

                self.assertTrue((root / "pipelines" / "escape.json").exists())
                self.assertFalse((root / "escape.json").exists())
                self.assertEqual(pipeline_path("../escape.json"), root / "pipelines" / "escape.json")
            finally:
                pipefunctions.PIPELINES_DIR = old_dir

    def test_legacy_migration_skips_symlinks(self):
        import nativelab.GlobalConfig.const as const

        old_legacy = const.LEGACY_PIPELINES_DIR
        old_new = const.PIPELINES_DIR
        with tempfile.TemporaryDirectory() as tmp:
            try:
                root = Path(tmp)
                legacy = root / "legacy"
                new = root / "new"
                legacy.mkdir()
                new.mkdir()
                (legacy / "regular.json").write_text("{}", encoding="utf-8")
                target = root / "secret.json"
                target.write_text('{"secret": true}', encoding="utf-8")
                symlink = legacy / "linked.json"
                try:
                    symlink.symlink_to(target)
                except OSError:
                    self.skipTest("symlink creation is not available")

                const.LEGACY_PIPELINES_DIR = legacy
                const.PIPELINES_DIR = new
                copied = const.migrate_legacy_pipelines()

                self.assertEqual(copied, 1)
                self.assertTrue((new / "regular.json").exists())
                self.assertFalse((new / "linked.json").exists())
            finally:
                const.LEGACY_PIPELINES_DIR = old_legacy
                const.PIPELINES_DIR = old_new


if __name__ == "__main__":
    unittest.main()
