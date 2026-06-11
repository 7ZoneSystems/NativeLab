import os
import unittest

os.environ.setdefault("NATIVELAB_NO_GUI", "1")

from nativelab.native.pipeline_core import (
    apply_transform,
    merge_texts,
    normalize_ids,
    route_edges,
    validate_records,
    would_form_loop,
)
from nativelab.pipelinebuilder.blck_typ import PipelineBlockType, PipelineConnection
from nativelab.pipelinebuilder.graph_ops import route_connections
from nativelab.pipelinebuilder.pipblck import PipelineBlock
from nativelab.pipelinebuilder.validation import validate_pipeline


class PipelineNativeCoreTests(unittest.TestCase):
    def test_normalize_ids_repairs_duplicates_and_remaps_connections(self):
        result = normalize_ids([7, 7, 8, "bad"], [(7, 8), (8, 7)], 0)

        ids = result["ids"]
        self.assertEqual(ids[0], 7)
        self.assertEqual(ids[2], 8)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertGreater(ids[1], 8)
        self.assertGreater(ids[3], 8)
        self.assertEqual(result["connections"], [(7, 8), (8, 7)])
        self.assertGreaterEqual(result["counter"], max(ids))

    def test_would_form_loop_detects_existing_reachability(self):
        self.assertTrue(would_form_loop([(1, 2), (2, 3)], 3, 1))
        self.assertFalse(would_form_loop([(1, 2), (2, 3)], 1, 3))

    def test_transform_and_merge_match_python_semantics(self):
        self.assertEqual(
            apply_transform("body", {"transform_type": "prefix", "transform_val": "head"}),
            "head\nbody",
        )
        self.assertEqual(
            apply_transform("a-b-a", {
                "transform_type": "replace",
                "transform_find": "a",
                "transform_repl": "x",
            }),
            "x-b-x",
        )
        self.assertEqual(
            apply_transform("abcdef", {"transform_type": "truncate", "transform_val": "3"}),
            "abc",
        )
        self.assertEqual(
            merge_texts(["first", "second"], {"merge_mode": "prepend", "merge_sep": "|"}),
            "second|first",
        )
        self.assertEqual(
            merge_texts(["a", "b"], {"merge_mode": "json"}),
            '[\n  "a",\n  "b"\n]',
        )

    def test_route_edges_preserves_visit_limits_and_branch_rules(self):
        records = [
            (0, 1, "E", 2, False, 1),
            (1, 1, "W", 3, False, 1),
            (2, 1, "S", 4, True, 2),
        ]
        visits = {}

        first = route_edges(records, 1, visits, mode="if", branch_key="TRUE")
        second = route_edges(records, 1, visits, mode="if", branch_key="TRUE")
        third = route_edges(records, 1, visits, mode="if", branch_key="TRUE")

        self.assertEqual(first, [(0, 2, "E"), (2, 4, "S")])
        self.assertEqual(second, [(2, 4, "S")])
        self.assertEqual(third, [])
        self.assertEqual(visits, {"1->2": 1, "1->4": 2})

    def test_route_connections_returns_original_connection_objects(self):
        conns = [
            PipelineConnection(1, "E", 2, "W"),
            PipelineConnection(1, "W", 3, "E"),
        ]
        visits = {}

        selected = route_connections(conns, 1, visits, mode="if", branch_key="FALSE")

        self.assertEqual(selected, [conns[1]])
        self.assertEqual(visits, {"1->3": 1})

    def test_validate_records_preserves_readiness_messages(self):
        self.assertEqual(validate_records([], 0), "Canvas is empty - add blocks first.")
        self.assertEqual(
            validate_records([
                {"btype": "input", "label": "Input", "metadata": {}},
                {"btype": "output", "label": "Output", "metadata": {}},
                {"btype": "reference", "label": "Ref", "metadata": {}},
            ], 1),
            "Reference block 'Ref' has no text.\nRight-click it \u2192 Configure block\u2026",
        )
        self.assertIsNone(
            validate_records([
                {"btype": "input", "label": "Input", "metadata": {}},
                {"btype": "output", "label": "Output", "metadata": {}},
            ], 1)
        )


class PipelineValidationWrapperTests(unittest.TestCase):
    def setUp(self):
        self._old_counter = PipelineBlock._id_counter
        PipelineBlock._id_counter = 0

    def tearDown(self):
        PipelineBlock._id_counter = self._old_counter

    def test_validate_pipeline_uses_block_records_without_ui_tab(self):
        inp = PipelineBlock(PipelineBlockType.INPUT, 0, 0)
        out = PipelineBlock(PipelineBlockType.OUTPUT, 100, 0)
        conn = PipelineConnection(inp.bid, "E", out.bid, "W")

        self.assertIsNone(validate_pipeline([inp, out], [conn]))

    def test_validate_pipeline_keeps_logic_safeguards(self):
        inp = PipelineBlock(PipelineBlockType.INPUT, 0, 0)
        transform = PipelineBlock(PipelineBlockType.TRANSFORM, 100, 0)
        out = PipelineBlock(PipelineBlockType.OUTPUT, 200, 0)
        conns = [
            PipelineConnection(inp.bid, "E", transform.bid, "W"),
            PipelineConnection(transform.bid, "E", out.bid, "W"),
        ]

        self.assertEqual(
            validate_pipeline([inp, transform, out], conns),
            "TRANSFORM block 'Model' has no transform type set.\nRight-click it \u2192 Configure block\u2026",
        )


if __name__ == "__main__":
    unittest.main()
