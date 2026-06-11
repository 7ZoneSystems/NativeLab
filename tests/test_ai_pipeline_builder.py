import json
import os
import tempfile
import unittest

os.environ.setdefault("NATIVELAB_NO_GUI", "1")

from nativelab.Model.model_global import api_model_ref
from nativelab.pipelinebuilder.aibuilder import planner
from nativelab.pipelinebuilder.aibuilder import dialog as ai_dialog
from nativelab.pipelinebuilder.aibuilder.context import (
    AiBuilderHistoryStore,
    build_smart_context_request,
    format_canvas_state,
    is_canvas_empty,
)
from nativelab.pipelinebuilder.aibuilder.dialog import AiPipelineBuildWorker
from nativelab.pipelinebuilder.aibuilder.planner import (
    GeneratedPipeline,
    PipelineJsonError,
    estimate_ai_builder_budget,
    extract_json_object,
    normalize_pipeline_data,
    save_generated_pipeline,
    sanitize_pipeline_name,
)
from nativelab.pipelinebuilder.blck_typ import PipelineBlockType
from nativelab.pipelinebuilder.pipblck import PipelineBlock


class AiPipelineBuilderPlannerTests(unittest.TestCase):
    def setUp(self):
        self._old_counter = PipelineBlock._id_counter
        PipelineBlock._id_counter = 0

    def tearDown(self):
        PipelineBlock._id_counter = self._old_counter

    def test_sanitizes_pipeline_name_for_safe_json_file(self):
        self.assertEqual(
            sanitize_pipeline_name("../bad/name\\pipeline?.json"),
            "bad-name-pipeline-.json",
        )
        self.assertEqual(sanitize_pipeline_name("   "), "ai-pipeline")

    def test_budget_blocks_before_context_overflow(self):
        class Engine:
            ctx_value = 128

        budget = estimate_ai_builder_budget(
            Engine(),
            "tiny",
            "Build a short pipeline.",
            n_predict=64,
        )

        self.assertTrue(budget.overflow)
        self.assertGreater(budget.projected_tokens, budget.limit_tokens)
        self.assertTrue(budget.messages)

    def test_extracts_first_balanced_json_object_from_fenced_response(self):
        raw = """Sure.
```json
{"version": 2, "blocks": [{"metadata": {"text": "brace } inside string"}}], "connections": []}
```
"""

        data = extract_json_object(raw)

        self.assertEqual(data["version"], 2)
        self.assertEqual(data["blocks"][0]["metadata"]["text"], "brace } inside string")

    def test_extracts_json_from_encoded_string_response(self):
        inner = json.dumps({"version": 2, "blocks": [], "connections": []})
        raw = json.dumps(inner)

        data = extract_json_object(raw)

        self.assertEqual(data["version"], 2)

    def test_normalize_repairs_ids_connections_and_required_defaults(self):
        data = normalize_pipeline_data({
            "blocks": [
                {"bid": 7, "btype": "input", "x": 0, "y": 0, "metadata": {}},
                {"bid": 7, "btype": "transform", "x": 200, "y": 0, "metadata": {}},
                {"bid": 8, "btype": "output", "x": 400, "y": 0, "metadata": {}},
            ],
            "connections": [
                {"from_block_id": 7, "from_port": "bad", "to_block_id": 8, "to_port": "bad"},
                {"from_block_id": 999, "to_block_id": 8},
            ],
        })

        bids = [block["bid"] for block in data["blocks"]]
        self.assertEqual(len(bids), len(set(bids)))
        self.assertEqual(len(data["connections"]), 1)
        self.assertEqual(data["connections"][0]["from_port"], "E")
        self.assertEqual(data["connections"][0]["to_port"], "W")
        transform = next(b for b in data["blocks"] if b["btype"] == PipelineBlockType.TRANSFORM)
        self.assertEqual(transform["metadata"]["transform_type"], "strip")

    def test_save_generated_pipeline_uses_existing_save_and_active_model(self):
        captured = {}
        original_save = planner.save_pipeline

        def fake_save(name, blocks, connections):
            captured["name"] = name
            captured["blocks"] = blocks
            captured["connections"] = connections

        planner.save_pipeline = fake_save
        try:
            raw = json.dumps({
                "version": 2,
                "blocks": [
                    {"bid": 1, "btype": "input", "x": 0, "y": 0, "label": "Input", "metadata": {}},
                    {"bid": 2, "btype": "model", "x": 200, "y": 0, "label": "Model", "metadata": {}},
                    {"bid": 3, "btype": "output", "x": 400, "y": 0, "label": "Output", "metadata": {}},
                ],
                "connections": [
                    {"from_block_id": 1, "from_port": "E", "to_block_id": 2, "to_port": "W"},
                    {"from_block_id": 2, "from_port": "E", "to_block_id": 3, "to_port": "W"},
                ],
            })

            result = save_generated_pipeline(
                "ai/generated",
                raw,
                active_model_ref=api_model_ref("builder-api"),
                active_model_role="general",
            )
        finally:
            planner.save_pipeline = original_save

        self.assertEqual(result.name, "ai-generated")
        self.assertEqual(captured["name"], "ai-generated")
        model_block = next(b for b in captured["blocks"] if b.btype == PipelineBlockType.MODEL)
        self.assertEqual(model_block.model_path, api_model_ref("builder-api"))
        self.assertEqual(len(captured["connections"]), 2)

    def test_worker_retries_once_when_first_response_has_no_json(self):
        class Engine:
            is_loaded = True
            ctx_value = 12000

        calls = []
        original_generate = ai_dialog.generate_pipeline_response
        original_save = ai_dialog.save_generated_pipeline

        def fake_generate(engine, messages, **kwargs):
            calls.append((messages, kwargs))
            return json.dumps({
                "version": 2,
                "blocks": [
                    {"bid": 1, "btype": "input", "x": 0, "y": 0, "metadata": {}},
                    {"bid": 2, "btype": "output", "x": 200, "y": 0, "metadata": {}},
                ],
                "connections": [
                    {"from_block_id": 1, "from_port": "E", "to_block_id": 2, "to_port": "W"},
                ],
            })

        def fake_save(name, raw, **kwargs):
            if raw == "not json":
                raise PipelineJsonError("The model response did not contain a JSON object.", raw)
            return GeneratedPipeline(name=name, raw_response=raw, data={}, blocks=[], connections=[])

        ai_dialog.generate_pipeline_response = fake_generate
        ai_dialog.save_generated_pipeline = fake_save
        try:
            worker = AiPipelineBuildWorker(
                Engine(),
                pipeline_name="retry-test",
                user_request="make a small pipeline",
            )
            result = worker._save_or_retry("not json")
        finally:
            ai_dialog.generate_pipeline_response = original_generate
            ai_dialog.save_generated_pipeline = original_save

        self.assertEqual(result.name, "retry-test")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1]["temperature"], 0.0)


class AiPipelineBuilderContextTests(unittest.TestCase):
    def test_empty_history_and_empty_canvas_passes_prompt_through(self):
        with tempfile.TemporaryDirectory() as td:
            history = AiBuilderHistoryStore(root=td)

            smart = build_smart_context_request(
                "make a simple pipeline",
                history=history,
                canvas_state={"blocks": [], "connections": []},
            )

        self.assertEqual(smart.command, "")
        self.assertEqual(smart.model_request, "make a simple pipeline")
        self.assertTrue(is_canvas_empty({"blocks": []}))

    def test_first_prompt_with_existing_canvas_includes_get_data_result(self):
        canvas = {
            "blocks": [{"bid": 1, "btype": "input", "label": "Input"}],
            "connections": [],
        }
        with tempfile.TemporaryDirectory() as td:
            history = AiBuilderHistoryStore(root=td)

            smart = build_smart_context_request(
                "add a model and output",
                history=history,
                canvas_state=canvas,
            )

        self.assertIn("first AI Builder prompt for an existing canvas", smart.model_request)
        self.assertIn("/get_data", smart.model_request)
        self.assertIn('"btype": "input"', smart.model_request)

    def test_get_data_command_returns_canvas_without_model_request(self):
        canvas = {"blocks": [{"bid": 3, "btype": "output"}], "connections": []}
        with tempfile.TemporaryDirectory() as td:
            history = AiBuilderHistoryStore(root=td)

            smart = build_smart_context_request(
                "/get_data",
                history=history,
                canvas_state=canvas,
            )

        self.assertEqual(smart.command, "get_data")
        self.assertEqual(smart.model_request, "")
        self.assertEqual(smart.notice, format_canvas_state(canvas))

    def test_context_command_is_intercepted_for_compaction(self):
        with tempfile.TemporaryDirectory() as td:
            history = AiBuilderHistoryStore(root=td)
            history.append_user("make pipeline")

            smart = build_smart_context_request(
                "/context",
                history=history,
                canvas_state={},
            )

        self.assertEqual(smart.command, "context")
        self.assertEqual(smart.model_request, "")

    def test_history_persists_under_configured_directory(self):
        with tempfile.TemporaryDirectory() as td:
            first = AiBuilderHistoryStore(root=td)
            first.append_user("first request")
            first.append_assistant("saved result")

            second = AiBuilderHistoryStore(root=td)

        self.assertEqual(len(second.messages), 2)
        self.assertEqual(second.messages[0]["content"], "first request")


if __name__ == "__main__":
    unittest.main()
