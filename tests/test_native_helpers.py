import unittest

from nativelab.Model.model_family import detect_model_family, detect_quant_type
from nativelab.Model.templates import FAMILY_TEMPLATES
from nativelab.native.engine_helpers import (
    append_cli_sampler_args,
    build_reference_chunks,
    build_text_prompt,
    sampler_payload,
)


class NativeHelpersTest(unittest.TestCase):
    def test_prompt_builder_matches_expected_mistral_format(self):
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": [{"text": "next"}, {"text": "turn"}]},
        ]
        prompt = build_text_prompt(messages, FAMILY_TEMPLATES["mistral"])
        self.assertEqual(
            prompt,
            "<s>[INST] system\nhello [/INST]hi</s>[INST] next\nturn [/INST]",
        )

    def test_sampler_payload_and_cli_args(self):
        self.assertEqual(
            sampler_payload(top_k=50, min_p=0.2, typical_p=0.8, seed=7),
            {"top_k": 50, "min_p": 0.2, "typical_p": 0.8, "seed": 7},
        )
        cmd = []
        append_cli_sampler_args(cmd, top_k=50, min_p=0.2, typical_p=0.8, seed=7)
        self.assertEqual(
            cmd,
            ["--top-k", "50", "--min-p", "0.2", "--typical", "0.8", "--seed", "7"],
        )

    def test_model_detection_routes_through_native_wrappers_or_fallbacks(self):
        filename = "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
        self.assertEqual(detect_model_family(filename).family, "llama3")
        self.assertEqual(detect_quant_type(filename), "Q4_K_M")

    def test_reference_chunking(self):
        self.assertEqual(
            build_reference_chunks("abcdefghij", 4, 2),
            ["abcdef", "efghij", "ij"],
        )


if __name__ == "__main__":
    unittest.main()
