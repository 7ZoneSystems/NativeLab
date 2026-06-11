import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("NATIVELAB_NO_GUI", "1")

from nativelab.integrations.endpoints import IntegrationEndpoints
from nativelab.pipelinebuilder.blck_typ import PipelineBlockType, PipelineConnection
from nativelab.pipelinebuilder.pipblck import PipelineBlock
from nativelab.pipelinebuilder.pipefunctions import save_pipeline


class PipelineApiModelTests(unittest.TestCase):
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

    def _save_passthrough_pipeline(self, name="api-pass"):
        inp = PipelineBlock(PipelineBlockType.INPUT, 0, 0, label="Input")
        out = PipelineBlock(PipelineBlockType.OUTPUT, 220, 0, label="Output")
        conn = PipelineConnection(inp.bid, "E", out.bid, "W")
        save_pipeline(name, [inp, out], [conn])

    def test_saved_pipelines_are_exposed_as_openai_models(self):
        self._save_passthrough_pipeline("public-pipe")

        data = IntegrationEndpoints().handle("/v1/models")

        ids = {item["id"] for item in data["data"]}
        self.assertIn("pipeline:public-pipe", ids)
        self.assertEqual(data["object"], "list")

    def test_openai_chat_completion_runs_saved_pipeline(self):
        self._save_passthrough_pipeline("echo-pipe")
        endpoints = IntegrationEndpoints()

        status, data = endpoints.openai_chat_completion({
            "model": "pipeline:echo-pipe",
            "messages": [{"role": "user", "content": "hello from api"}],
        })

        self.assertEqual(status, 200)
        self.assertEqual(data["model"], "pipeline:echo-pipe")
        self.assertEqual(
            data["choices"][0]["message"]["content"],
            "hello from api",
        )
        self.assertEqual(data["nativelab"]["kind"], "pipeline")
        self.assertEqual(data["nativelab"]["pipeline"], "echo-pipe")

    def test_unknown_pipeline_model_returns_openai_error(self):
        status, data = IntegrationEndpoints().openai_chat_completion({
            "model": "pipeline:missing",
            "messages": [{"role": "user", "content": "hello"}],
        })

        self.assertEqual(status, 404)
        self.assertIn("error", data)


if __name__ == "__main__":
    unittest.main()
