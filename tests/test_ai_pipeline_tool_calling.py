import json
import os
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("NATIVELAB_NO_GUI", "1")

from nativelab.pipelinebuilder.aibuilder import dialog as ai_dialog
from nativelab.pipelinebuilder.aibuilder import planner
from nativelab.pipelinebuilder.aibuilder.mcp_verify import (
    McpVerificationReport,
    McpVerifyResult,
    _find_tool_for_block,
    _infer_arg_name,
    extract_tool_blocks,
    fix_mcp_blocks_after_verification,
    heal_connections_after_removal,
    verify_all_tool_blocks,
    verify_web_search_block,
)
from nativelab.pipelinebuilder.aibuilder.planner import (
    GeneratedPipeline,
    PipelineJsonError,
    auto_repair_pipeline_graph,
    build_ai_builder_diagnostic_retry_messages,
    estimate_ai_builder_diagnostic_retry_budget,
    save_generated_pipeline,
)
from nativelab.pipelinebuilder.blck_typ import PipelineBlockType
from nativelab.pipelinebuilder.executionWorker import PipelineExecutionWorker
from nativelab.pipelinebuilder.pipblck import PipelineBlock
from nativelab.pipelinebuilder.validation import validate_pipeline


class AiPipelineToolCallingTests(unittest.TestCase):
    def setUp(self):
        self._old_counter = PipelineBlock._id_counter
        PipelineBlock._id_counter = 0

    def tearDown(self):
        PipelineBlock._id_counter = self._old_counter

    def test_intelligent_tool_name_matching(self):
        available_tools = [
            {"name": "brave_web_search", "description": "Search the web"},
            {"name": "read_text_file", "description": "Read file contents"},
            {"name": "sqlite_query_db", "description": "Run SQL query"},
        ]

        # 1. Exact match
        block1 = {"metadata": {"mcp_tool_name": "brave_web_search"}}
        self.assertEqual(_find_tool_for_block(block1, available_tools), "brave_web_search")

        # 2. Case-insensitive match
        block2 = {"metadata": {"mcp_tool_name": "BRAVE_WEB_SEARCH"}}
        self.assertEqual(_find_tool_for_block(block2, available_tools), "brave_web_search")

        # 3. Normalized punctuation / kebab-case
        block3 = {"metadata": {"mcp_tool_name": "read-text-file"}}
        self.assertEqual(_find_tool_for_block(block3, available_tools), "read_text_file")

        # 4. Substring containment
        block4 = {"metadata": {"mcp_tool_name": "web_search"}}
        self.assertEqual(_find_tool_for_block(block4, available_tools), "brave_web_search")

        # 5. Fuzzy similarity
        block5 = {"metadata": {"mcp_tool_name": "query_sqlite"}}
        self.assertEqual(_find_tool_for_block(block5, available_tools), "sqlite_query_db")

    def test_tool_argument_inference_from_schema(self):
        block = {"metadata": {"mcp_tool_name": "search", "mcp_arg_name": ""}}
        tool_with_query = {
            "name": "search",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        }
        _infer_arg_name(block, tool_with_query)
        self.assertEqual(block["metadata"]["mcp_arg_name"], "query")

        # Tool with required file_path
        block2 = {"metadata": {"mcp_tool_name": "read", "mcp_arg_name": "input"}}
        tool_with_path = {
            "name": "read",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                },
                "required": ["file_path"],
            },
        }
        _infer_arg_name(block2, tool_with_path)
        self.assertEqual(block2["metadata"]["mcp_arg_name"], "file_path")

    def test_web_search_verification_and_synonym_normalization(self):
        block = {
            "bid": 2,
            "label": "Web Search Block",
            "btype": "web_search",
            "metadata": {
                "ws_categories": ["tech", "academic", "finance", "unknown_category"],
                "ws_max_results": 999,  # exceeds upper bound
                "ws_timeout": 1,        # below lower bound
                "ws_output_format": "xml",  # invalid format
            },
        }
        result = verify_web_search_block(block)

        self.assertEqual(result.block_type, "web_search")
        meta = block["metadata"]
        # tech -> it, academic -> science, finance -> news
        self.assertIn("it", meta["ws_categories"])
        self.assertIn("science", meta["ws_categories"])
        self.assertIn("news", meta["ws_categories"])
        self.assertNotIn("unknown_category", meta["ws_categories"])
        # Clamped values
        self.assertEqual(meta["ws_max_results"], 50)
        self.assertEqual(meta["ws_timeout"], 3)
        self.assertEqual(meta["ws_output_format"], "text")

    def test_heal_connections_after_tool_block_removal(self):
        # 1 (Input) -> 2 (Unreachable MCP) -> 3 (Model) -> 4 (Output)
        connections = [
            {"from_block_id": 1, "from_port": "E", "to_block_id": 2, "to_port": "W"},
            {"from_block_id": 2, "from_port": "E", "to_block_id": 3, "to_port": "W"},
            {"from_block_id": 3, "from_port": "E", "to_block_id": 4, "to_port": "W"},
        ]
        healed = heal_connections_after_removal(connections, [2])

        self.assertEqual(len(healed), 2)
        # Check bridged connection 1 -> 3
        bridged = next(c for c in healed if c["from_block_id"] == 1)
        self.assertEqual(bridged["to_block_id"], 3)
        self.assertEqual(bridged["from_port"], "E")
        self.assertEqual(bridged["to_port"], "W")
        # Check remaining connection 3 -> 4
        tail = next(c for c in healed if c["from_block_id"] == 3)
        self.assertEqual(tail["to_block_id"], 4)

    def test_auto_repair_pipeline_graph(self):
        # Defective graph:
        # 1. No input block
        # 2. Direct model-to-model connection (bid 10 -> bid 20)
        # 3. No connection to output block
        blocks = [
            {"bid": 10, "btype": "model", "x": 100, "y": 100, "model_path": "@api/test-model", "label": "M1", "metadata": {}},
            {"bid": 20, "btype": "model", "x": 400, "y": 100, "model_path": "@api/test-model", "label": "M2", "metadata": {}},
            {"bid": 30, "btype": "output", "x": 600, "y": 100, "label": "Out", "metadata": {}},
        ]
        connections = [
            {"from_block_id": 10, "from_port": "E", "to_block_id": 20, "to_port": "W"},
        ]

        repaired_blocks, repaired_conns, notes = auto_repair_pipeline_graph(blocks, connections)

        # 1. Input block inserted
        self.assertTrue(any(b["btype"] == PipelineBlockType.INPUT for b in repaired_blocks))
        # 2. Intermediate block inserted between M1 (10) and M2 (20)
        self.assertTrue(any(b["btype"] == PipelineBlockType.INTERMEDIATE for b in repaired_blocks))
        # 3. Output block connected
        out_bid = next(b["bid"] for b in repaired_blocks if b["btype"] == PipelineBlockType.OUTPUT)
        self.assertTrue(any(c["to_block_id"] == out_bid for c in repaired_conns))
        # 4. Verify validate_pipeline passes
        pipeline_blocks, pipeline_conns = planner.pipeline_data_to_blocks({
            "blocks": repaired_blocks,
            "connections": repaired_conns,
        })
        err = validate_pipeline(pipeline_blocks, pipeline_conns)
        self.assertIsNone(err)

    def test_concurrent_tool_verification(self):
        blocks = [
            {"bid": 1, "btype": "web_search", "label": "Web 1", "metadata": {"ws_categories": ["it"]}},
            {"bid": 2, "btype": "web_search", "label": "Web 2", "metadata": {"ws_categories": ["science"]}},
        ]
        report = verify_all_tool_blocks(blocks, max_workers=2)

        self.assertEqual(len(report.results), 2)
        self.assertTrue(report.has_tools)
        self.assertEqual(len(report.web_search_results), 2)

    def test_worker_autonomous_retry_on_validation_error(self):
        class Engine:
            is_loaded = True
            ctx_value = 12000

        calls = []
        original_generate = ai_dialog.generate_pipeline_response
        original_save = ai_dialog.save_generated_pipeline

        valid_json = json.dumps({
            "version": 2,
            "blocks": [
                {"bid": 1, "btype": "input", "x": 0, "y": 0, "metadata": {}},
                {"bid": 2, "btype": "model", "x": 200, "y": 0, "model_path": "@api/test-model", "metadata": {}},
                {"bid": 3, "btype": "intermediate", "x": 400, "y": 0, "metadata": {}},
                {"bid": 4, "btype": "output", "x": 600, "y": 0, "metadata": {}},
            ],
            "connections": [
                {"from_block_id": 1, "from_port": "E", "to_block_id": 2, "to_port": "W"},
                {"from_block_id": 2, "from_port": "E", "to_block_id": 3, "to_port": "W"},
                {"from_block_id": 3, "from_port": "E", "to_block_id": 4, "to_port": "W"},
            ],
        })

        def fake_generate(engine, messages, **kwargs):
            calls.append((messages, kwargs))
            return valid_json

        attempt = [0]
        def fake_save(name, raw, **kwargs):
            if attempt[0] == 0:
                attempt[0] += 1
                raise ValueError("Cycle limit exceeded in connection between 2 and 3.")
            data = json.loads(raw)
            blocks, conns = planner.pipeline_data_to_blocks(data)
            return GeneratedPipeline(name=name, raw_response=raw, data=data, blocks=blocks, connections=conns)

        ai_dialog.generate_pipeline_response = fake_generate
        ai_dialog.save_generated_pipeline = fake_save
        try:
            worker = ai_dialog.AiPipelineBuildWorker(
                Engine(),
                pipeline_name="val-retry-test",
                user_request="build a pipeline with 2 models",
            )
            result = worker._save_or_retry(valid_json)
        finally:
            ai_dialog.generate_pipeline_response = original_generate
            ai_dialog.save_generated_pipeline = original_save

        self.assertEqual(result.name, "val-retry-test")
        self.assertEqual(len(calls), 1)
        prompt_content = calls[0][0][1]["content"]
        self.assertIn("Validation/Error Feedback to Fix", prompt_content)
        self.assertIn("Cycle limit exceeded", prompt_content)

    def test_execution_worker_mcp_autonomous_retry_and_passthrough(self):
        worker = PipelineExecutionWorker([], [], "", None)
        block = PipelineBlock("mcp_server", 100, 100, "", "general", "MCP Tool")
        block.metadata = {
            "mcp_transport": "stdio",
            "mcp_url": "echo test",
            "mcp_tool_name": "test_tool",
            "mcp_max_retries": 1,
            "mcp_passthrough_on_err": True,
        }

        # Mock McpClient to fail once then succeed
        with patch("nativelab.integrations.mcp_client.McpClient") as mock_client_cls:
            mock_inst = MagicMock()
            mock_inst.execute.side_effect = [(False, "transient pipe error"), (True, "tool output success")]
            mock_client_cls.return_value = mock_inst

            result = worker._run_mcp_block(block, "incoming context")
            self.assertEqual(result, "tool output success")
            self.assertEqual(mock_inst.execute.call_count, 2)

        # Mock McpClient to fail all attempts with passthrough enabled
        with patch("nativelab.integrations.mcp_client.McpClient") as mock_client_cls:
            mock_inst = MagicMock()
            mock_inst.execute.return_value = (False, "fatal connection error")
            mock_client_cls.return_value = mock_inst

            result2 = worker._run_mcp_block(block, "original context")
            self.assertIn("Tool Notice:", result2)
            self.assertIn("original context", result2)

    def test_execution_worker_web_search_retry_and_passthrough(self):
        worker = PipelineExecutionWorker([], [], "", None)
        block = PipelineBlock("web_search", 100, 100, "", "general", "Search")
        block.metadata = {
            "ws_categories": ["science"],
            "ws_language": "en",
            "ws_max_results": 5,
            "ws_timeout": 5,
            "ws_output_format": "text",
            "ws_passthrough_on_err": True,
        }

        with patch("nativelab.web_search.web_search_text") as mock_search:
            # First attempt fails with exception, second succeeds
            mock_search.side_effect = [Exception("engine timeout"), "1. Quantum Computing Article\nURL: https://..."]
            result = worker._run_web_search_block(block, "quantum computing")
            self.assertIn("Quantum Computing Article", result)
            self.assertEqual(mock_search.call_count, 2)


if __name__ == "__main__":
    unittest.main()
