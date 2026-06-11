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

    def test_canvas_grows_to_fit_far_loaded_blocks(self):
        canvas = PipelineCanvas()
        far = PipelineBlock(PipelineBlockType.OUTPUT, 2600, 1800)
        canvas.blocks = [far]

        canvas.ensure_canvas_fits_blocks()

        self.assertGreaterEqual(canvas._canvas_w, far.x + far.w)
        self.assertGreaterEqual(canvas._canvas_h, far.y + far.h)

    def test_clear_all_resets_canvas_to_minimum_size(self):
        canvas = PipelineCanvas()
        far = PipelineBlock(PipelineBlockType.OUTPUT, 2600, 1800)
        canvas.blocks = [far]
        canvas.ensure_canvas_fits_blocks()

        canvas.clear_all()

        self.assertEqual(canvas._canvas_w, canvas.MIN_CANVAS_W)
        self.assertEqual(canvas._canvas_h, canvas.MIN_CANVAS_H)

    def test_pan_drag_updates_scrollbars(self):
        class _Point:
            def __init__(self, x, y):
                self._x = x
                self._y = y

            def x(self):
                return self._x

            def y(self):
                return self._y

        class _Event:
            def __init__(self, x, y):
                self._point = _Point(x, y)

            def globalPosition(self):
                return self._point

        class _Bar:
            def __init__(self, value):
                self._value = value

            def value(self):
                return self._value

            def setValue(self, value):
                self._value = value

        class _Scroll:
            def __init__(self):
                self.h = _Bar(100)
                self.v = _Bar(200)

            def horizontalScrollBar(self):
                return self.h

            def verticalScrollBar(self):
                return self.v

        canvas = PipelineCanvas()
        scroll = _Scroll()
        canvas._scroll_area = lambda: scroll

        canvas._start_pan(_Event(300, 400))
        canvas._pan_to(_Event(250, 360))

        self.assertEqual(scroll.h.value(), 150)
        self.assertEqual(scroll.v.value(), 240)


if __name__ == "__main__":
    unittest.main()
