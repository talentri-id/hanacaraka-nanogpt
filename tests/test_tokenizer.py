from __future__ import annotations

import unittest

from tokenizer_jawa import (
    ConstrainedSyllableBPETokenizer,
    JavaneseSyllableTokenizer,
    segment_javanese,
    validate_token,
)


class TokenizerTests(unittest.TestCase):
    def test_syllable_segmentation_basic(self) -> None:
        text = "ꦲꦤꦕꦫꦏ"
        tokens = segment_javanese(text, keep_whitespace=True, keep_unknown=False)
        self.assertEqual(tokens, ["ꦲ", "ꦤ", "ꦕ", "ꦫ", "ꦏ"])
        for tok in tokens:
            result = validate_token(tok)
            self.assertTrue(result.valid)
            self.assertEqual(result.kind, "syllable")

    def test_bpe_reduces_token_count_without_breaking_roundtrip(self) -> None:
        text = "ꦲꦤꦕꦫꦏ ꦲꦤꦕꦫꦏ ꦲꦤꦕꦫꦏ"
        atomic = JavaneseSyllableTokenizer().segment(text)
        tok = ConstrainedSyllableBPETokenizer(target_vocab_size=64, min_pair_freq=2)
        tok.fit_text(text)
        bpe_tokens = tok.segment(text)

        self.assertEqual(tok.decode_ids(tok.encode_tokens(bpe_tokens)), text)
        self.assertLessEqual(len(bpe_tokens), len(atomic))
        self.assertTrue(any(len(tok.token_pieces_map[piece]) > 1 for piece in bpe_tokens if piece in tok.token_pieces_map))

    def test_bpe_never_merges_across_space(self) -> None:
        text = "ꦲꦤ ꦕꦫ"
        tok = ConstrainedSyllableBPETokenizer(target_vocab_size=64, min_pair_freq=2)
        tok.fit_text(text)
        bpe_tokens = tok.segment(text)
        self.assertIn(" ", bpe_tokens)
        self.assertNotIn("ꦲꦤ ꦕꦫ", bpe_tokens)


if __name__ == "__main__":
    unittest.main()
