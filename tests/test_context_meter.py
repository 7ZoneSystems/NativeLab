import unittest

from nativelab.core.context_meter import context_meter, estimate_tokens


class ContextMeterTest(unittest.TestCase):
    def test_prompt_snapshot_and_output_growth(self):
        snap = context_meter.report_prompt(
            source="Labs",
            prompt="abcd" * 20,
            n_predict=16,
            limit_tokens=64,
        )
        self.assertEqual(snap["source"], "Labs")
        self.assertEqual(snap["input_tokens"], estimate_tokens("abcd" * 20))
        self.assertEqual(snap["used_tokens"], snap["input_tokens"])
        self.assertEqual(snap["projected_tokens"], snap["input_tokens"] + 16)

        grown = context_meter.append_output("abcd" * 4)
        self.assertGreater(grown["used_tokens"], snap["used_tokens"])
        self.assertEqual(grown["limit_tokens"], 64)


if __name__ == "__main__":
    unittest.main()
