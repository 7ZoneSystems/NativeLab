import os
import unittest

os.environ.setdefault("NATIVELAB_NO_GUI", "1")

from nativelab.pipelinebuilder.blck_typ import PipelineBlockType, PipelineConnection
from nativelab.pipelinebuilder.canvas import PipelineCanvas
from nativelab.pipelinebuilder.pipblck import PipelineBlock


class PipelineCanvasIdTests(unittest.TestCase):
    def setUp(self):
        self._old_counter = PipelineBlock._id_counter
        PipelineBlock._id_counter = 0

    def tearDown(self):
        PipelineBlock._id_counter = self._old_counter

    def test_remove_block_uses_object_identity_when_ids_collide(self):
        canvas = PipelineCanvas()
        first = PipelineBlock(PipelineBlockType.INPUT, 0, 0)
        second = PipelineBlock(PipelineBlockType.OUTPUT, 20, 0)
        second.bid = first.bid
        canvas.blocks = [first, second]
        canvas.connections = [PipelineConnection(first.bid, "E", second.bid, "W")]

        canvas.remove_block(first)

        self.assertEqual(canvas.blocks, [second])
        self.assertEqual(canvas.connections, [])

    def test_add_block_syncs_counter_above_loaded_blocks(self):
        canvas = PipelineCanvas()
        loaded = PipelineBlock(PipelineBlockType.INPUT, 0, 0)
        loaded.bid = 42
        PipelineBlock._id_counter = 0
        canvas.blocks = [loaded]

        created = canvas.add_block(PipelineBlockType.OUTPUT, 180, 0)

        self.assertEqual(created.bid, 43)
        self.assertNotEqual(created.bid, loaded.bid)

    def test_add_block_repairs_duplicate_inserted_pipeline_ids(self):
        canvas = PipelineCanvas()
        first = PipelineBlock(PipelineBlockType.INPUT, 0, 0)
        second = PipelineBlock(PipelineBlockType.INTERMEDIATE, 20, 0)
        third = PipelineBlock(PipelineBlockType.OUTPUT, 40, 0)
        first.bid = 7
        second.bid = 7
        third.bid = 8
        PipelineBlock._id_counter = 0
        canvas.blocks = [first, second, third]

        created = canvas.add_block(PipelineBlockType.MODEL, 80, 0)

        bids = [block.bid for block in canvas.blocks]
        self.assertEqual(len(bids), len(set(bids)))
        self.assertNotIn(created.bid, {7, 8})
        self.assertGreater(created.bid, 8)

    def test_normalize_block_ids_repairs_invalid_loaded_ids(self):
        canvas = PipelineCanvas()
        first = PipelineBlock(PipelineBlockType.INPUT, 0, 0)
        second = PipelineBlock(PipelineBlockType.OUTPUT, 20, 0)
        first.bid = 0
        second.bid = "bad"
        canvas.blocks = [first, second]

        canvas.normalize_block_ids()

        bids = [block.bid for block in canvas.blocks]
        self.assertEqual(len(bids), len(set(bids)))
        self.assertTrue(all(isinstance(bid, int) and bid > 0 for bid in bids))

    def test_clear_all_does_not_rewind_global_counter(self):
        canvas = PipelineCanvas()
        created = canvas.add_block(PipelineBlockType.INPUT, 0, 0)

        canvas.clear_all()

        self.assertGreaterEqual(PipelineBlock._id_counter, created.bid)


if __name__ == "__main__":
    unittest.main()
