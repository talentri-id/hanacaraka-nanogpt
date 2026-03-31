from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

JAVANESE_BLOCK_START = 0xA980
JAVANESE_BLOCK_END = 0xA9DF
DOTTED_CIRCLE = 0x25CC
ZWNJ = 0x200C
ZWJ = 0x200D
JOINERS = {ZWNJ, ZWJ}

BINDU = {0xA980, 0xA981}
FINAL_CONSONANT = {0xA982}
VISARGA = {0xA983}
BASE = set(range(0xA984, 0xA9B3))
NUKTA = 0xA9B3
VOWEL_RIGHT = {0xA9B4, 0xA9B5}
VOWEL_TOP = {0xA9B6, 0xA9B7, 0xA9BC}
VOWEL_BOTTOM = {0xA9B8, 0xA9B9}
VOWEL_LEFT = {0xA9BA, 0xA9BB}
MEDIAL_LEFT = {0xA9BF, 0xA9BD}
MEDIAL_RIGHT = {0xA9BE}
PANGKON = 0xA9C0
JAVANESE_PUNCT = set(range(0xA9C1, 0xA9D0)) | {0xA9DE, 0xA9DF}
DIGITS = set(range(0xA9D0, 0xA9DA))

SYLLABLE_TRAILING_CLASSES = (
    MEDIAL_LEFT,
    MEDIAL_RIGHT,
    VOWEL_LEFT,
    VOWEL_TOP,
    VOWEL_BOTTOM,
    VOWEL_RIGHT,
    BINDU,
    VISARGA,
    FINAL_CONSONANT,
)


@dataclass
class TokenValidation:
    token: str
    kind: str
    valid: bool
    reason: str = ""


@dataclass
class TextMetrics:
    total_tokens: int
    evaluated_tokens: int
    syllable_like_tokens: int
    invalid_tokens: int
    orphan_mark_tokens: int
    explicit_dotted_circles: int

    @property
    def invalid_token_rate(self) -> float:
        return self.invalid_tokens / self.evaluated_tokens if self.evaluated_tokens else 0.0

    @property
    def orphan_mark_rate(self) -> float:
        return self.orphan_mark_tokens / self.evaluated_tokens if self.evaluated_tokens else 0.0

    @property
    def dotted_circle_rate(self) -> float:
        return self.explicit_dotted_circles / self.total_tokens if self.total_tokens else 0.0

    def to_dict(self) -> dict[str, float | int]:
        data = asdict(self)
        data["invalid_token_rate"] = self.invalid_token_rate
        data["orphan_mark_rate"] = self.orphan_mark_rate
        data["dotted_circle_rate"] = self.dotted_circle_rate
        return data


Symbol = tuple[str, ...]
SymbolPair = tuple[Symbol, Symbol]


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", text)


def is_javanese_codepoint(cp: int) -> bool:
    return JAVANESE_BLOCK_START <= cp <= JAVANESE_BLOCK_END or cp == DOTTED_CIRCLE


def is_mark(cp: int) -> bool:
    return (
        cp in BINDU
        or cp in FINAL_CONSONANT
        or cp in VISARGA
        or cp == NUKTA
        or cp in VOWEL_RIGHT
        or cp in VOWEL_TOP
        or cp in VOWEL_BOTTOM
        or cp in VOWEL_LEFT
        or cp in MEDIAL_LEFT
        or cp in MEDIAL_RIGHT
        or cp == PANGKON
    )


def is_base(cp: int) -> bool:
    return cp in BASE or cp == DOTTED_CIRCLE


def is_javanese_punct(cp: int) -> bool:
    return cp in JAVANESE_PUNCT


def filter_to_javanese_text(
    text: str,
    *,
    keep_whitespace: bool = True,
    keep_ascii_punct: bool = False,
    keep_non_javanese: bool = False,
) -> str:
    """Keep Hanacaraka block + whitespace by default.

    This is intentionally conservative for a Hanacaraka-only training corpus.
    """
    text = normalize_text(text)
    out: list[str] = []
    for ch in text:
        cp = ord(ch)
        if keep_non_javanese:
            out.append(ch)
            continue
        if ch.isspace():
            if keep_whitespace:
                out.append(ch)
            continue
        if is_javanese_codepoint(cp) or cp in JOINERS:
            out.append(ch)
            continue
        if keep_ascii_punct and unicodedata.category(ch).startswith("P"):
            out.append(ch)
            continue
    return "".join(out)


def segment_javanese(
    text: str,
    *,
    keep_whitespace: bool = True,
    keep_unknown: bool = True,
) -> list[str]:
    """Greedy orthographic-syllable segmentation for Javanese.

    The implementation follows the class order in Unicode TN47 section 5.
    It is intentionally practical rather than exhaustive for every historical edge case.
    """
    text = normalize_text(text)
    cps = [ord(ch) for ch in text]
    out: list[str] = []
    i = 0

    while i < len(cps):
        cp = cps[i]
        ch = chr(cp)

        if ch.isspace():
            if keep_whitespace:
                out.append(ch)
            i += 1
            continue

        if cp in JOINERS or cp in JAVANESE_PUNCT or cp in DIGITS:
            out.append(ch)
            i += 1
            continue

        if is_base(cp):
            buf = [ch]
            i += 1

            if i < len(cps) and cps[i] == NUKTA:
                buf.append(chr(cps[i]))
                i += 1

            conjuncts = 0
            while (
                conjuncts < 2
                and i + 1 < len(cps)
                and cps[i] == PANGKON
                and cps[i + 1] in BASE
            ):
                buf.append(chr(cps[i]))
                buf.append(chr(cps[i + 1]))
                i += 2
                conjuncts += 1

                if i < len(cps) and cps[i] == NUKTA:
                    buf.append(chr(cps[i]))
                    i += 1

            if i < len(cps) and cps[i] == PANGKON:
                buf.append(chr(cps[i]))
                i += 1
                out.append("".join(buf))
                continue

            for cls in SYLLABLE_TRAILING_CLASSES:
                if i < len(cps) and cps[i] in cls:
                    buf.append(chr(cps[i]))
                    i += 1

            out.append("".join(buf))
            continue

        if keep_unknown:
            out.append(ch)
        i += 1

    return out


def validate_token(token: str) -> TokenValidation:
    if token == "":
        return TokenValidation(token, kind="empty", valid=False, reason="empty-token")

    if all(ch.isspace() for ch in token):
        return TokenValidation(token, kind="whitespace", valid=True)

    cps = [ord(ch) for ch in token]
    if len(cps) == 1 and cps[0] in DIGITS:
        return TokenValidation(token, kind="digit", valid=True)
    if len(cps) == 1 and cps[0] in JAVANESE_PUNCT:
        return TokenValidation(token, kind="punct", valid=True)
    if len(cps) == 1 and cps[0] in JOINERS:
        return TokenValidation(token, kind="joiner", valid=True)
    if not any(is_javanese_codepoint(cp) for cp in cps):
        return TokenValidation(token, kind="other", valid=True)

    i = 0
    n = len(cps)
    if not is_base(cps[i]):
        if is_mark(cps[i]):
            return TokenValidation(
                token,
                kind="orphan_mark",
                valid=False,
                reason="orphan-mark-or-nonbase-start",
            )
        return TokenValidation(token, kind="other", valid=False, reason="non-javanese-start")

    i += 1
    if i < n and cps[i] == NUKTA:
        i += 1

    conjuncts = 0
    while conjuncts < 2 and i + 1 < n and cps[i] == PANGKON and cps[i + 1] in BASE:
        i += 2
        conjuncts += 1
        if i < n and cps[i] == NUKTA:
            i += 1

    if i < n and cps[i] == PANGKON:
        i += 1
        if i == n:
            return TokenValidation(token, kind="syllable", valid=True)
        return TokenValidation(
            token,
            kind="syllable",
            valid=False,
            reason="characters-after-visible-pangkon",
        )

    for cls in SYLLABLE_TRAILING_CLASSES:
        if i < n and cps[i] in cls:
            i += 1

    if i == n:
        return TokenValidation(token, kind="syllable", valid=True)

    if is_mark(cps[i]):
        return TokenValidation(
            token,
            kind="syllable",
            valid=False,
            reason="duplicate-or-out-of-order-mark",
        )
    return TokenValidation(token, kind="syllable", valid=False, reason="unexpected-trailing-character")


def compute_text_metrics(text: str) -> TextMetrics:
    tokens = segment_javanese(text, keep_whitespace=True, keep_unknown=True)
    evaluated = 0
    syllable_like = 0
    invalid = 0
    orphan = 0
    explicit_dotted_circles = text.count(chr(DOTTED_CIRCLE))

    for tok in tokens:
        result = validate_token(tok)
        if result.kind in {"syllable", "orphan_mark"}:
            syllable_like += 1
            evaluated += 1
            if not result.valid:
                invalid += 1
            if result.kind == "orphan_mark":
                orphan += 1

    return TextMetrics(
        total_tokens=len(tokens),
        evaluated_tokens=evaluated,
        syllable_like_tokens=syllable_like,
        invalid_tokens=invalid,
        orphan_mark_tokens=orphan,
        explicit_dotted_circles=explicit_dotted_circles,
    )


def _surface(symbol: Sequence[str]) -> str:
    return "".join(symbol)


def _contains_pair(seq: Sequence[Symbol], pair: SymbolPair) -> bool:
    left, right = pair
    return any(a == left and b == right for a, b in zip(seq, seq[1:]))


def _merge_pair_in_sequence(seq: Sequence[Symbol], pair: SymbolPair) -> tuple[Symbol, ...]:
    left, right = pair
    out: list[Symbol] = []
    i = 0
    while i < len(seq):
        if i + 1 < len(seq) and seq[i] == left and seq[i + 1] == right:
            out.append(seq[i] + seq[i + 1])
            i += 2
            continue
        out.append(seq[i])
        i += 1
    return tuple(out)


class BaseTokenizer:
    tokenizer_type: str = "base"
    unk_token: str | None = None
    replacement_char: str = "�"

    def __init__(self, vocab: Sequence[str] | None = None):
        self.vocab: list[str] = list(vocab) if vocab is not None else []
        self._refresh_maps()

    def _refresh_maps(self) -> None:
        if len(set(self.vocab)) != len(self.vocab):
            raise ValueError("Vocabulary contains duplicate surface forms.")
        self.stoi: dict[str, int] = {tok: i for i, tok in enumerate(self.vocab)}
        self.itos: dict[int, str] = {i: tok for i, tok in enumerate(self.vocab)}

    def __len__(self) -> int:
        return len(self.vocab)

    def segment(self, text: str) -> list[str]:
        raise NotImplementedError

    def fit(self, text_or_tokens: str | Iterable[str]) -> Counter[str]:
        if isinstance(text_or_tokens, str):
            tokens = self.segment(text_or_tokens)
        else:
            tokens = list(text_or_tokens)
        freq = Counter(tokens)
        self.vocab = [tok for tok, _ in sorted(freq.items(), key=lambda item: (-item[1], item[0]))]
        self._refresh_maps()
        return freq

    def fit_text(self, text: str) -> Counter[str]:
        return self.fit(text)

    def encode_tokens(self, tokens: Iterable[str], *, allow_unk: bool = False) -> list[int]:
        ids: list[int] = []
        for tok in tokens:
            if tok in self.stoi:
                ids.append(self.stoi[tok])
                continue
            if allow_unk and self.unk_token is not None and self.unk_token in self.stoi:
                ids.append(self.stoi[self.unk_token])
                continue
            raise KeyError(f"Token not in vocabulary: {tok!r}")
        return ids

    def decode_ids(self, ids: Iterable[int]) -> str:
        pieces: list[str] = []
        for idx in ids:
            tok = self.itos.get(int(idx))
            if tok is None:
                if self.unk_token is not None:
                    pieces.append(self.replacement_char)
                    continue
                raise KeyError(f"Token id not in vocabulary: {idx}")
            if self.unk_token is not None and tok == self.unk_token:
                pieces.append(self.replacement_char)
            else:
                pieces.append(tok)
        return "".join(pieces)

    def encode(self, text: str, *, allow_unk: bool = False) -> list[int]:
        return self.encode_tokens(self.segment(text), allow_unk=allow_unk)

    def _to_payload(self) -> dict[str, object]:
        return {
            "tokenizer_type": self.tokenizer_type,
            "unk_token": self.unk_token,
            "replacement_char": self.replacement_char,
            "vocab": self.vocab,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(json.dumps(self._to_payload(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BaseTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        tok_type = payload["tokenizer_type"]
        if tok_type == JavaneseSyllableTokenizer.tokenizer_type:
            return JavaneseSyllableTokenizer.from_payload(payload)
        if tok_type == CodepointTokenizer.tokenizer_type:
            return CodepointTokenizer.from_payload(payload)
        if tok_type == ConstrainedSyllableBPETokenizer.tokenizer_type:
            return ConstrainedSyllableBPETokenizer.from_payload(payload)
        raise ValueError(f"Unsupported tokenizer type: {tok_type}")


class JavaneseSyllableTokenizer(BaseTokenizer):
    tokenizer_type = "javanese_syllable"

    def segment(self, text: str) -> list[str]:
        return segment_javanese(normalize_text(text), keep_whitespace=True, keep_unknown=True)

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "JavaneseSyllableTokenizer":
        tokenizer = cls(vocab=payload["vocab"])
        tokenizer.unk_token = payload.get("unk_token")
        tokenizer.replacement_char = payload.get("replacement_char", tokenizer.replacement_char)
        tokenizer._refresh_maps()
        return tokenizer


class CodepointTokenizer(BaseTokenizer):
    tokenizer_type = "codepoint"

    def segment(self, text: str) -> list[str]:
        return list(normalize_text(text))

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "CodepointTokenizer":
        tokenizer = cls(vocab=payload["vocab"])
        tokenizer.unk_token = payload.get("unk_token")
        tokenizer.replacement_char = payload.get("replacement_char", tokenizer.replacement_char)
        tokenizer._refresh_maps()
        return tokenizer


class ConstrainedSyllableBPETokenizer(BaseTokenizer):
    """CBPE-style tokenizer with orthographic syllables as atomic units.

    It never splits inside a Javanese orthographic syllable. Merges happen only across
    adjacent syllable tokens inside a word-like span, and never across whitespace,
    punctuation, digits, or joiners.
    """

    tokenizer_type = "syllable_bpe"

    def __init__(
        self,
        vocab: Sequence[str] | None = None,
        *,
        target_vocab_size: int = 4096,
        min_pair_freq: int = 2,
        max_merges: int | None = None,
        token_pieces: Sequence[Sequence[str]] | None = None,
        merges: Sequence[Sequence[Sequence[str]]] | None = None,
    ):
        self.target_vocab_size = int(target_vocab_size)
        self.min_pair_freq = int(min_pair_freq)
        self.max_merges = None if max_merges is None else int(max_merges)
        self.merges: list[SymbolPair] = []
        if merges is not None:
            self.merges = [(tuple(left), tuple(right)) for left, right in merges]
        self.merge_ranks: dict[SymbolPair, int] = {pair: i for i, pair in enumerate(self.merges)}
        self.token_pieces_map: dict[str, Symbol] = {}
        super().__init__(vocab=vocab)
        if token_pieces is not None and vocab is not None:
            self.token_pieces_map = {tok: tuple(pieces) for tok, pieces in zip(vocab, token_pieces)}
        elif vocab is not None:
            self.token_pieces_map = {tok: (tok,) for tok in vocab}
        self._validate_token_pieces_map()

    def _validate_token_pieces_map(self) -> None:
        for tok in self.vocab:
            pieces = self.token_pieces_map.get(tok)
            if pieces is None:
                self.token_pieces_map[tok] = (tok,)
                continue
            if _surface(pieces) != tok:
                raise ValueError(f"Token pieces do not reconstruct token surface: {tok!r} <- {pieces!r}")

    @staticmethod
    def _is_mergeable_atom(tok: str) -> bool:
        result = validate_token(tok)
        return result.kind == "syllable" and result.valid

    def atomic_segment(self, text: str) -> list[str]:
        return segment_javanese(normalize_text(text), keep_whitespace=True, keep_unknown=True)

    def _iter_words(self, atomic_tokens: Sequence[str]) -> Iterable[tuple[str, ...]]:
        current: list[str] = []
        for tok in atomic_tokens:
            if self._is_mergeable_atom(tok):
                current.append(tok)
                continue
            if current:
                yield tuple(current)
                current = []
        if current:
            yield tuple(current)

    def _apply_bpe_to_word(self, atoms: Sequence[str]) -> list[str]:
        symbols: list[Symbol] = [(atom,) for atom in atoms]
        if not self.merge_ranks:
            return [_surface(sym) for sym in symbols]

        while len(symbols) > 1:
            best_rank: int | None = None
            best_idx = -1
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                rank = self.merge_ranks.get(pair)
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_idx = i
            if best_idx < 0:
                break
            merged = symbols[best_idx] + symbols[best_idx + 1]
            symbols = symbols[:best_idx] + [merged] + symbols[best_idx + 2 :]
        return [_surface(sym) for sym in symbols]

    def _apply_bpe_to_sequence(self, atomic_tokens: Sequence[str]) -> list[str]:
        out: list[str] = []
        current: list[str] = []
        for tok in atomic_tokens:
            if self._is_mergeable_atom(tok):
                current.append(tok)
                continue
            if current:
                out.extend(self._apply_bpe_to_word(current))
                current = []
            out.append(tok)
        if current:
            out.extend(self._apply_bpe_to_word(current))
        return out

    def segment(self, text: str) -> list[str]:
        atomic_tokens = self.atomic_segment(text)
        return self._apply_bpe_to_sequence(atomic_tokens)

    def fit(self, text_or_tokens: str | Iterable[str]) -> Counter[str]:
        if isinstance(text_or_tokens, str):
            atomic_tokens = self.atomic_segment(text_or_tokens)
        else:
            atomic_tokens = list(text_or_tokens)

        atomic_vocab = sorted(set(atomic_tokens))
        token_pieces_map: dict[str, Symbol] = {tok: (tok,) for tok in atomic_vocab}

        word_freq = Counter(self._iter_words(atomic_tokens))
        word_symbols: dict[tuple[str, ...], tuple[Symbol, ...]] = {
            word: tuple((atom,) for atom in word) for word in word_freq if word
        }

        target_vocab_size = max(self.target_vocab_size, len(atomic_vocab))
        max_by_target = max(target_vocab_size - len(atomic_vocab), 0)
        merge_budget = max_by_target if self.max_merges is None else min(self.max_merges, max_by_target)

        merges: list[SymbolPair] = []
        for _ in range(merge_budget):
            pair_freq: Counter[SymbolPair] = Counter()
            for word, symbols in word_symbols.items():
                freq = word_freq[word]
                for pair in zip(symbols, symbols[1:]):
                    pair_freq[pair] += freq

            if not pair_freq:
                break

            best_pair, best_count = max(
                pair_freq.items(),
                key=lambda item: (item[1], _surface(item[0][0]) + _surface(item[0][1])),
            )
            if best_count < self.min_pair_freq:
                break

            merged_symbol = best_pair[0] + best_pair[1]
            merged_surface = _surface(merged_symbol)
            if merged_surface not in token_pieces_map:
                token_pieces_map[merged_surface] = merged_symbol
            merges.append(best_pair)

            for word, symbols in list(word_symbols.items()):
                if _contains_pair(symbols, best_pair):
                    word_symbols[word] = _merge_pair_in_sequence(symbols, best_pair)

        self.merges = merges
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}

        final_tokens = self._apply_bpe_to_sequence(atomic_tokens)
        final_freq = Counter(final_tokens)
        all_surfaces = list(token_pieces_map.keys())
        self.vocab = sorted(
            all_surfaces,
            key=lambda tok: (-final_freq.get(tok, 0), len(token_pieces_map[tok]), tok),
        )
        self.token_pieces_map = {tok: token_pieces_map[tok] for tok in self.vocab}
        self._refresh_maps()
        return final_freq

    def fit_text(self, text: str) -> Counter[str]:
        return self.fit(text)

    def _to_payload(self) -> dict[str, object]:
        payload = super()._to_payload()
        payload.update(
            {
                "target_vocab_size": self.target_vocab_size,
                "min_pair_freq": self.min_pair_freq,
                "max_merges": self.max_merges,
                "token_pieces": [list(self.token_pieces_map.get(tok, (tok,))) for tok in self.vocab],
                "merges": [
                    {"left": list(left), "right": list(right)}
                    for left, right in self.merges
                ],
                "merge_count": len(self.merges),
            }
        )
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "ConstrainedSyllableBPETokenizer":
        merges_payload = payload.get("merges", [])
        merges: list[tuple[list[str], list[str]]] = []
        for item in merges_payload:
            if isinstance(item, dict):
                merges.append((list(item["left"]), list(item["right"])))
            else:
                left, right = item
                merges.append((list(left), list(right)))
        tokenizer = cls(
            vocab=payload.get("vocab", []),
            target_vocab_size=payload.get("target_vocab_size", 4096),
            min_pair_freq=payload.get("min_pair_freq", 2),
            max_merges=payload.get("max_merges"),
            token_pieces=payload.get("token_pieces"),
            merges=merges,
        )
        tokenizer.unk_token = payload.get("unk_token")
        tokenizer.replacement_char = payload.get("replacement_char", tokenizer.replacement_char)
        tokenizer._refresh_maps()
        tokenizer._validate_token_pieces_map()
        return tokenizer


def tokenizer_from_name(name: str, **kwargs: object) -> BaseTokenizer:
    if name == "syllable":
        return JavaneseSyllableTokenizer()
    if name == "char":
        return CodepointTokenizer()
    if name == "syllable_bpe":
        return ConstrainedSyllableBPETokenizer(**kwargs)
    raise ValueError(f"Unsupported tokenizer kind: {name}")


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Inspect Javanese syllable and syllable-BPE tokenization.")
    parser.add_argument("--text", type=str, required=True, help="Input text to segment.")
    parser.add_argument("--tokenizer", choices=["syllable", "char", "syllable_bpe"], default="syllable")
    parser.add_argument("--target_vocab_size", type=int, default=128, help="Used only for syllable_bpe.")
    parser.add_argument("--min_pair_freq", type=int, default=2, help="Used only for syllable_bpe.")
    args = parser.parse_args()

    print("SYLLABLES:")
    for idx, tok in enumerate(segment_javanese(args.text, keep_whitespace=True, keep_unknown=True)):
        result = validate_token(tok)
        print(f"{idx:03d}  {tok!r}  kind={result.kind}  valid={result.valid}  reason={result.reason}")

    tokenizer = tokenizer_from_name(
        args.tokenizer,
        target_vocab_size=args.target_vocab_size,
        min_pair_freq=args.min_pair_freq,
    )
    freq = tokenizer.fit_text(args.text)
    tokens = tokenizer.segment(args.text)
    print("\nTOKENIZER:")
    print(f"type={tokenizer.tokenizer_type} vocab_size={len(tokenizer)} token_count={len(tokens)}")
    if isinstance(tokenizer, ConstrainedSyllableBPETokenizer):
        print(f"merge_count={len(tokenizer.merges)}")
    for idx, tok in enumerate(tokens[:80]):
        print(f"{idx:03d}  {tok!r}")
    print("\nTOP TOKENS:")
    for tok, count in sorted(freq.items(), key=lambda item: (-item[1], item[0]))[:20]:
        print(f"{tok!r}: {count}")
    print("\nMETRICS:")
    print(json.dumps(compute_text_metrics(args.text).to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
