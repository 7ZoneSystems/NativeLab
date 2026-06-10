import unittest

from nativelab.pipelinebuilder.blck_typ import PipelineBlockType
from nativelab.pipelinebuilder.pipefunctions import (
    list_example_pipelines,
    load_example_pipeline,
)


class PipelineExampleTests(unittest.TestCase):
    def test_shipped_examples_are_listed(self):
        names = {item["name"] for item in list_example_pipelines()}
        self.assertIn("quick-answer", names)
        self.assertIn("draft-review", names)
        self.assertIn("clean-summarize", names)
        self.assertIn("triage-router", names)
        self.assertIn("research-synthesis-fanout", names)
        self.assertIn("llm-quality-gate", names)
        self.assertIn("llm-classify-and-respond", names)
        self.assertIn("briefing-pack-builder", names)

    def test_examples_deserialize_to_valid_blocks(self):
        for item in list_example_pipelines():
            with self.subTest(example=item["name"]):
                blocks, connections = load_example_pipeline(item["name"])
                block_types = {block.btype for block in blocks}
                self.assertIn(PipelineBlockType.INPUT, block_types)
                self.assertIn(PipelineBlockType.OUTPUT, block_types)
                self.assertGreaterEqual(len(connections), 1)

    def test_examples_keep_model_placeholders_unbound(self):
        model_backed_types = {
            PipelineBlockType.MODEL,
            PipelineBlockType.LLM_IF,
            PipelineBlockType.LLM_SWITCH,
            PipelineBlockType.LLM_FILTER,
            PipelineBlockType.LLM_TRANSFORM,
            PipelineBlockType.LLM_SCORE,
        }
        for item in list_example_pipelines():
            blocks, _ = load_example_pipeline(item["name"])
            model_blocks = [
                block for block in blocks
                if block.btype in model_backed_types
            ]
            if model_blocks:
                self.assertTrue(any(not block.model_path for block in model_blocks))

    def test_complex_examples_exercise_advanced_blocks(self):
        expected = {
            "triage-router": {PipelineBlockType.IF_ELSE, PipelineBlockType.TRANSFORM},
            "research-synthesis-fanout": {
                PipelineBlockType.SPLIT,
                PipelineBlockType.MERGE,
                PipelineBlockType.KNOWLEDGE,
            },
            "llm-quality-gate": {PipelineBlockType.LLM_SCORE},
            "llm-classify-and-respond": {PipelineBlockType.LLM_SWITCH},
            "briefing-pack-builder": {
                PipelineBlockType.CUSTOM_CODE,
                PipelineBlockType.SPLIT,
                PipelineBlockType.MERGE,
                PipelineBlockType.LLM_TRANSFORM,
            },
        }
        for name, block_types in expected.items():
            with self.subTest(example=name):
                blocks, connections = load_example_pipeline(name)
                actual = {block.btype for block in blocks}
                self.assertTrue(block_types.issubset(actual))
                self.assertGreaterEqual(len(blocks), 8)
                self.assertGreaterEqual(len(connections), len(blocks) - 1)


if __name__ == "__main__":
    unittest.main()
