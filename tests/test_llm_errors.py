import unittest

from nativelab.core.llm_errors import explain_llm_error


class LlmErrorExplanationTests(unittest.TestCase):
    def test_llama_context_error_gets_plain_language_actions(self):
        raw = (
            'Model block error: llama-server HTTP 400: '
            '{"error":{"code":400,"message":"request (3749 tokens) exceeds '
            'the available context size (2048 tokens), try increasing it",'
            '"type":"exceed_context_size_error","n_prompt_tokens":3749,"n_ctx":2048}}'
        )

        notice = explain_llm_error(raw, source="Pipeline builder")

        self.assertEqual(notice.category, "context_size")
        self.assertEqual(notice.prompt_tokens, 3749)
        self.assertEqual(notice.context_tokens, 2048)
        self.assertIn("3,749", notice.summary)
        self.assertIn("2,048", notice.summary)
        self.assertIn("Increase the model context size", notice.action)
        self.assertIn(raw, notice.technical_detail)

    def test_http_400_without_context_gets_request_guidance(self):
        notice = explain_llm_error("llama-server HTTP 400: invalid temperature", source="Chat")

        self.assertEqual(notice.category, "bad_request")
        self.assertIn("backend rejected", notice.summary)
        self.assertIn("model settings", notice.action)


if __name__ == "__main__":
    unittest.main()
