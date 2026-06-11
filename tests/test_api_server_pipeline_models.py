import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("NATIVELAB_NO_GUI", "1")

from nativelab.api_server.catalog import model_catalog, resolve_model_ref
from nativelab.api_server.config import ACTIVE_MODEL_REF, ApiServerConfig
from nativelab.api_server.server import NativeLabApiServer
from nativelab.pipelinebuilder.blck_typ import PipelineBlockType, PipelineConnection
from nativelab.pipelinebuilder.pipblck import PipelineBlock
from nativelab.pipelinebuilder.pipefunctions import save_pipeline


class _FakeLabEndpoints:
    model_path = ""
    is_loaded = False

    def active_engine(self):
        return None

    def _log(self, *_args):
        return None


class ApiServerPipelineModelTests(unittest.TestCase):
    def setUp(self):
        import nativelab.GlobalConfig.config_global as config_global
        import nativelab.pipelinebuilder.pipefunctions as pipefunctions

        self._old_config_dir = config_global.PIPELINES_DIR
        self._old_pipe_dir = pipefunctions.PIPELINES_DIR
        self._tmp = tempfile.TemporaryDirectory()
        self._pipeline_dir = Path(self._tmp.name)
        config_global.PIPELINES_DIR = self._pipeline_dir
        pipefunctions.PIPELINES_DIR = self._pipeline_dir

    def tearDown(self):
        import nativelab.GlobalConfig.config_global as config_global
        import nativelab.pipelinebuilder.pipefunctions as pipefunctions

        config_global.PIPELINES_DIR = self._old_config_dir
        pipefunctions.PIPELINES_DIR = self._old_pipe_dir
        self._tmp.cleanup()

    def _save_passthrough_pipeline(self, name="api-server-pipe"):
        inp = PipelineBlock(PipelineBlockType.INPUT, 0, 0, label="Input")
        out = PipelineBlock(PipelineBlockType.OUTPUT, 220, 0, label="Output")
        conn = PipelineConnection(inp.bid, "E", out.bid, "W")
        save_pipeline(name, [inp, out], [conn])

    def test_api_server_model_catalog_includes_saved_pipelines(self):
        self._save_passthrough_pipeline("hosted-pipe")

        rows = model_catalog()
        refs = {row["native_ref"] for row in rows}

        self.assertIn("pipeline:hosted-pipe", refs)
        self.assertEqual(resolve_model_ref("pipeline:hosted-pipe"), "pipeline:hosted-pipe")

    def test_selected_pipeline_ref_runs_without_model_load(self):
        self._save_passthrough_pipeline("echo-hosted")
        cfg = ApiServerConfig(model_ref="pipeline:echo-hosted", require_api_key=False)
        server = NativeLabApiServer(_FakeLabEndpoints(), cfg)

        model_id = server._ensure_model(None)
        text = server._generate_pipeline(
            model_id,
            [{"role": "user", "content": "hello from hosted pipeline"}],
        )

        self.assertEqual(model_id, "pipeline:echo-hosted")
        self.assertEqual(text, "hello from hosted pipeline")

    def test_missing_requested_pipeline_ref_is_rejected(self):
        cfg = ApiServerConfig(model_ref=ACTIVE_MODEL_REF, require_api_key=False)
        server = NativeLabApiServer(_FakeLabEndpoints(), cfg)

        with self.assertRaisesRegex(RuntimeError, "Pipeline model not found"):
            server._ensure_model("pipeline:missing")

    def test_missing_selected_pipeline_ref_is_rejected(self):
        cfg = ApiServerConfig(model_ref="pipeline:missing", require_api_key=False)
        server = NativeLabApiServer(_FakeLabEndpoints(), cfg)

        with self.assertRaisesRegex(RuntimeError, "Pipeline model not found"):
            server._ensure_model(None)

    def test_public_health_payload_does_not_expose_runtime(self):
        cfg = ApiServerConfig(model_ref=ACTIVE_MODEL_REF, require_api_key=True)
        server = NativeLabApiServer(_FakeLabEndpoints(), cfg)

        payload = server._public_health_payload()

        self.assertEqual(payload["auth"]["required"], True)
        self.assertNotIn("runtime", payload)
        self.assertNotIn("endpoints", payload)


if __name__ == "__main__":
    unittest.main()
