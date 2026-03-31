from __future__ import annotations

import html
import json
import re
import subprocess
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from tokenizer_jawa import normalize_text

# Core Javanese code points used by the builtin fallback transliterator.
PANGKON = "\uA9C0"
CECAK_TELU = "\uA9B3"
WULU = "\uA9B6"
WULU_MELIK = "\uA9B7"
SUKU = "\uA9B8"
SUKU_MENDUT = "\uA9B9"
TALING = "\uA9BA"
DIRGA_MURE = "\uA9BB"
PEPET = "\uA9BC"
TARUNG = "\uA9B4"
CECAK = "\uA981"
LAYAR = "\uA982"
WIGNYAN = "\uA983"
HA = "\uA9B2"
LETTER_A = "\uA984"
LETTER_I = "\uA986"
LETTER_U = "\uA988"
LETTER_E = "\uA98C"
LETTER_O = "\uA98E"
LETTER_AI = "\uA98D"
LETTER_QA = "\uA990"
LETTER_KA = "\uA98F"
LETTER_GA = "\uA992"
LETTER_NGA = "\uA994"
LETTER_CA = "\uA995"
LETTER_JA = "\uA997"
LETTER_NYA = "\uA99A"
LETTER_TTA = "\uA99B"
LETTER_DDA = "\uA99D"
LETTER_TA = "\uA9A0"
LETTER_DA = "\uA9A2"
LETTER_NA = "\uA9A4"
LETTER_PA = "\uA9A5"
LETTER_BA = "\uA9A7"
LETTER_MA = "\uA9A9"
LETTER_YA = "\uA9AA"
LETTER_RA = "\uA9AB"
LETTER_LA = "\uA9AD"
LETTER_WA = "\uA9AE"
LETTER_SHA = "\uA9AF"
LETTER_SA = "\uA9B1"

ASCII_HYPHENS = "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"
ASCII_QUOTES = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201B": "'",
    "\u2032": "'",
    "\u02BC": "'",
    "\u201C": '"',
    "\u201D": '"',
    "\u201E": '"',
    "\u2033": '"',
}

WORD_RE = re.compile(r"[0-9A-Za-zÀ-ÖØ-öø-ÿĔĕĚěƏə`'\-]+|\s+|[^0-9A-Za-zÀ-ÖØ-öø-ÿĔĕĚěƏə`'\-\s]", re.UNICODE)
MULTI_CHAR_CONSONANTS = (
    "ng",
    "ny",
    "dh",
    "th",
    "kh",
    "gh",
    "sy",
    "sh",
    "dz",
    "ph",
    "bh",
    "ch",
    "jh",
)
VOWEL_MULTI = ("ai", "aa", "ii", "uu")
REKAN_BASES = {
    "f": LETTER_PA + CECAK_TELU,
    "v": LETTER_PA + CECAK_TELU,
    "z": LETTER_JA + CECAK_TELU,
    "kh": LETTER_KA + CECAK_TELU,
    "gh": LETTER_GA + CECAK_TELU,
    "dz": LETTER_DA + CECAK_TELU,
}
SINGLE_CONSONANT_BASES = {
    "h": HA,
    "n": LETTER_NA,
    "c": LETTER_CA,
    "r": LETTER_RA,
    "k": LETTER_KA,
    "d": LETTER_DA,
    "t": LETTER_TA,
    "s": LETTER_SA,
    "w": LETTER_WA,
    "l": LETTER_LA,
    "p": LETTER_PA,
    "j": LETTER_JA,
    "y": LETTER_YA,
    "m": LETTER_MA,
    "g": LETTER_GA,
    "b": LETTER_BA,
    "q": LETTER_QA,
    "x": LETTER_KA + PANGKON + LETTER_SA,  # fallback when x is truly consonantal
}
DIGRAPH_BASES = {
    "ng": LETTER_NGA,
    "ny": LETTER_NYA,
    "dh": LETTER_DDA,
    "th": LETTER_TTA,
    "sy": LETTER_SHA,
    "sh": LETTER_SHA,
    "ph": LETTER_PA,
    "bh": LETTER_BA,
    "ch": LETTER_CA,
    "jh": LETTER_JA,
}
SPECIAL_FINAL_SIGNS = {
    "ng": CECAK,
    "r": LAYAR,
    "h": WIGNYAN,
}
JAVANESE_DIGITS = {str(i): chr(0xA9D0 + i) for i in range(10)}


@dataclass(slots=True)
class TransliterationResult:
    requested_backend: str
    used_backend: str
    used_fallback: bool


class LatinToJavaneseBuiltin:
    """A practical fallback transliterator.

    This is intentionally conservative and lighter-weight than dedicated
    libraries such as Carakan.js. The goal is to keep corpus building usable
    even when Node/npm are unavailable.
    """

    def __init__(self, *, plain_e_mode: str = "pepet", keep_digits: bool = True):
        if plain_e_mode not in {"pepet", "taling"}:
            raise ValueError("plain_e_mode must be 'pepet' or 'taling'.")
        self.plain_e_mode = plain_e_mode
        self.keep_digits = keep_digits

    def normalize(self, text: str) -> str:
        return normalize_latin_javanese(text)

    def transliterate_text(self, text: str) -> str:
        text = self.normalize(text)
        parts: list[str] = []
        for token in WORD_RE.findall(text):
            if not token:
                continue
            if token.isspace():
                parts.append(token)
                continue
            if _looks_like_word(token):
                parts.append(self.transliterate_word(token))
                continue
            parts.append(token)
        return "".join(parts)

    def transliterate_lines(self, lines: Sequence[str]) -> list[str]:
        return [self.transliterate_text(line) for line in lines]

    def transliterate_word(self, word: str) -> str:
        word = word.strip()
        if not word:
            return ""
        if all(ch.isdigit() for ch in word):
            if not self.keep_digits:
                return ""
            return "".join(JAVANESE_DIGITS.get(ch, "") for ch in word)

        # Hyphenated reduplication or apostrophe-like joins are treated as hard breaks.
        subwords = [w for w in re.split(r"[-']+", word) if w]
        converted = [self._transliterate_atomic_word(w) for w in subwords]
        return " ".join(piece for piece in converted if piece)

    def _transliterate_atomic_word(self, word: str) -> str:
        units = list(_scan_roman_units(word, plain_e_mode=self.plain_e_mode))
        if not units:
            return ""

        out: list[str] = []
        i = 0
        n = len(units)
        while i < n:
            kind, value = units[i]
            if kind == "digit":
                if self.keep_digits:
                    out.append(JAVANESE_DIGITS.get(value, ""))
                i += 1
                continue
            if kind == "punct":
                out.append(value)
                i += 1
                continue
            if kind == "vowel":
                piece = independent_vowel(value)
                i += 1
                while i < n and units[i][0] == "consonant" and units[i][1] in SPECIAL_FINAL_SIGNS:
                    if i + 1 < n and units[i + 1][0] == "vowel":
                        break
                    piece += SPECIAL_FINAL_SIGNS[units[i][1]]
                    i += 1
                out.append(piece)
                continue
            if kind != "consonant":
                out.append(value)
                i += 1
                continue

            piece = consonant_base(value)
            i += 1

            # Build conjunct clusters greedily when another consonant is followed by a vowel.
            while i < n and units[i][0] == "consonant" and i + 1 < n and units[i + 1][0] == "vowel":
                piece += PANGKON + consonant_base(units[i][1])
                i += 1

            if i < n and units[i][0] == "vowel":
                piece += dependent_vowel(units[i][1])
                i += 1
                while i < n and units[i][0] == "consonant" and units[i][1] in SPECIAL_FINAL_SIGNS:
                    if i + 1 < n and units[i + 1][0] == "vowel":
                        break
                    piece += SPECIAL_FINAL_SIGNS[units[i][1]]
                    i += 1
            else:
                piece += PANGKON

            out.append(piece)

        return "".join(out)


def normalize_latin_javanese(text: str) -> str:
    text = html.unescape(text)
    text = normalize_text(text)
    text = text.replace("e`", "è").replace("E`", "È")
    text = text.translate(str.maketrans({ch: "-" for ch in ASCII_HYPHENS}))
    text = text.translate(str.maketrans(ASCII_QUOTES))
    replacements = {
        "ə": "ê",
        "Ə": "Ê",
        "ĕ": "ê",
        "Ĕ": "Ê",
        "ě": "ê",
        "Ě": "Ê",
        "ẽ": "ê",
        "Ẽ": "Ê",
        "ë": "ê",
        "Ë": "Ê",
        "ê": "ê",
        "Ê": "Ê",
        "è": "è",
        "È": "È",
        "é": "é",
        "É": "É",
        "ô": "o",
        "Ô": "O",
    }
    text = "".join(replacements.get(ch, ch) for ch in text)
    # Strip obvious HTML leftovers and zero-width characters.
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()


def choose_backend(
    requested_backend: str,
    *,
    repo_root: Path,
    node_bin: str = "node",
) -> TransliterationResult:
    requested_backend = requested_backend.lower()
    if requested_backend not in {"auto", "carakanjs", "builtin"}:
        raise ValueError("backend must be one of: auto, carakanjs, builtin")

    if requested_backend == "builtin":
        return TransliterationResult(requested_backend, used_backend="builtin", used_fallback=False)

    if carakanjs_available(repo_root=repo_root, node_bin=node_bin):
        return TransliterationResult(requested_backend, used_backend="carakanjs", used_fallback=False)

    if requested_backend == "carakanjs":
        raise RuntimeError(
            "Requested backend 'carakanjs', but it is not available. "
            "Run `npm install` in the repo root first, or switch to --backend builtin."
        )
    return TransliterationResult(requested_backend, used_backend="builtin", used_fallback=True)


def transliterate_lines(
    lines: Sequence[str],
    *,
    backend: str,
    repo_root: Path,
    plain_e_mode: str = "pepet",
    node_bin: str = "node",
    keep_digits: bool = True,
) -> list[str]:
    if backend == "carakanjs":
        return transliterate_lines_carakanjs(lines, repo_root=repo_root, node_bin=node_bin)
    if backend == "builtin":
        builtin = LatinToJavaneseBuiltin(plain_e_mode=plain_e_mode, keep_digits=keep_digits)
        return builtin.transliterate_lines(lines)
    raise ValueError(f"Unsupported backend: {backend}")


def carakanjs_available(*, repo_root: Path, node_bin: str = "node") -> bool:
    bridge = repo_root / "tools" / "carakan_bridge.mjs"
    if not bridge.exists():
        return False
    try:
        proc = subprocess.run(
            [node_bin, str(bridge), "--self-test"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def transliterate_lines_carakanjs(
    lines: Sequence[str],
    *,
    repo_root: Path,
    node_bin: str = "node",
) -> list[str]:
    bridge = repo_root / "tools" / "carakan_bridge.mjs"
    payload = {
        "config": {"useAccents": True, "useSwara": False, "useMurda": False},
        "lines": list(lines),
    }
    proc = subprocess.run(
        [node_bin, str(bridge)],
        cwd=str(repo_root),
        input=json.dumps(payload, ensure_ascii=False),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=max(30, 5 + len(lines) // 50),
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Carakan.js backend failed. stderr was:\n" + proc.stderr.strip()
        )
    data = json.loads(proc.stdout)
    out = data.get("lines")
    if not isinstance(out, list):
        raise RuntimeError("Carakan.js bridge returned invalid JSON payload.")
    return [str(item) for item in out]


def consonant_base(token: str) -> str:
    if token in REKAN_BASES:
        return REKAN_BASES[token]
    if token in DIGRAPH_BASES:
        return DIGRAPH_BASES[token]
    if token in SINGLE_CONSONANT_BASES:
        return SINGLE_CONSONANT_BASES[token]
    # Fallback for rare Latin letters.
    return LETTER_KA + PANGKON + LETTER_SA


def dependent_vowel(vowel: str) -> str:
    if vowel == "a":
        return ""
    if vowel == "aa":
        return TARUNG
    if vowel == "i":
        return WULU
    if vowel == "ii":
        return WULU_MELIK
    if vowel == "u":
        return SUKU
    if vowel == "uu":
        return SUKU_MENDUT
    if vowel == "ê":
        return PEPET
    if vowel == "é":
        return TALING
    if vowel == "o":
        return TALING + TARUNG
    if vowel == "ai":
        return DIRGA_MURE
    raise ValueError(f"Unsupported vowel token: {vowel!r}")


def independent_vowel(vowel: str) -> str:
    if vowel == "a":
        return HA
    if vowel == "aa":
        return LETTER_A + TARUNG
    if vowel == "i":
        return LETTER_I
    if vowel == "ii":
        return LETTER_I + WULU_MELIK
    if vowel == "u":
        return LETTER_U
    if vowel == "uu":
        return LETTER_U + SUKU_MENDUT
    if vowel == "ê":
        return HA + PEPET
    if vowel == "é":
        return LETTER_E
    if vowel == "o":
        return LETTER_O
    if vowel == "ai":
        return LETTER_AI
    raise ValueError(f"Unsupported vowel token: {vowel!r}")


def _looks_like_word(token: str) -> bool:
    return any(ch.isalpha() or ch.isdigit() for ch in token)


def _scan_roman_units(word: str, *, plain_e_mode: str) -> Iterator[tuple[str, str]]:
    word = normalize_latin_javanese(word).lower()
    i = 0
    n = len(word)
    while i < n:
        ch = word[i]
        if ch.isspace():
            i += 1
            continue
        if ch.isdigit():
            yield ("digit", ch)
            i += 1
            continue
        if ch in "-'`":
            yield ("punct", ch)
            i += 1
            continue
        matched = False
        for vowel in VOWEL_MULTI:
            if word.startswith(vowel, i):
                yield ("vowel", _normalize_vowel_token(vowel, plain_e_mode=plain_e_mode))
                i += len(vowel)
                matched = True
                break
        if matched:
            continue
        for cons in MULTI_CHAR_CONSONANTS:
            if word.startswith(cons, i):
                yield ("consonant", cons)
                i += len(cons)
                matched = True
                break
        if matched:
            continue

        if ch in {"a", "i", "u", "o", "e", "é", "è", "ê", "x"}:
            yield ("vowel", _normalize_vowel_token(ch, plain_e_mode=plain_e_mode))
            i += 1
            continue
        if ch.isalpha():
            yield ("consonant", ch)
            i += 1
            continue
        yield ("punct", ch)
        i += 1


def _normalize_vowel_token(token: str, *, plain_e_mode: str) -> str:
    token = token.lower()
    if token in {"è", "é"}:
        return "é"
    if token in {"ê", "x"}:
        return "ê"
    if token == "e":
        return "ê" if plain_e_mode == "pepet" else "é"
    return token


__all__ = [
    "LatinToJavaneseBuiltin",
    "TransliterationResult",
    "carakanjs_available",
    "choose_backend",
    "normalize_latin_javanese",
    "transliterate_lines",
    "transliterate_lines_carakanjs",
]
