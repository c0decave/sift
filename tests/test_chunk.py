import unittest

from sift.chunk import chunk_text, estimate_tokens, input_char_budget


class ChunkTest(unittest.TestCase):
    def test_estimate_tokens(self):
        self.assertEqual(estimate_tokens(""), 0)
        self.assertEqual(estimate_tokens("abcd"), 1)
        self.assertEqual(estimate_tokens("a" * 8), 2)

    def test_budget_positive(self):
        self.assertGreater(input_char_budget(8192, 800), 0)
        # never negative even with an absurd output reservation
        self.assertGreaterEqual(input_char_budget(1000, 100000), 512 * 4)

    def test_short_text_single_chunk(self):
        self.assertEqual(chunk_text("hello world", 100), ["hello world"])

    def test_empty_text(self):
        self.assertEqual(chunk_text("", 100), [])

    def test_large_text_multiple_chunks_within_limit(self):
        text = ("para " * 50 + "\n\n") * 40  # ~ many paragraphs
        chunks = chunk_text(text, max_chars=200, overlap=20)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c), 200)

    def test_hard_split_no_whitespace(self):
        text = "x" * 1000
        chunks = chunk_text(text, max_chars=100)
        self.assertEqual(len(chunks), 10)
        for c in chunks:
            self.assertLessEqual(len(c), 100)
        self.assertEqual("".join(chunks), text)

    def test_coverage_starts_at_beginning(self):
        text = "The quick brown fox. " * 100
        chunks = chunk_text(text, max_chars=120, overlap=10)
        self.assertTrue(text.startswith(chunks[0][:20]))


if __name__ == "__main__":
    unittest.main()
