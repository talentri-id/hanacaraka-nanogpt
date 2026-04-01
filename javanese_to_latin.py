"""Transliterate Javanese script (Hanacaraka) to Latin alphabet."""
from __future__ import annotations

import re
import unicodedata

# === Javanese Unicode ranges ===
# Independent vowels: A9B4-A9B5 not independent, A984-A98E are
# Consonants: A98F-A9B1 (base letters)
# Dependent vowel signs: A9B4-A9BC
# Virama (pangkon): A9C0
# Various signs: A981-A983 (cecak, layar, wignyan)
# Digits: A9D0-A9D9

# Base consonant to Latin
CONSONANT_MAP = {
    '\uA98F': 'ka',   # ꦏ
    '\uA990': 'qa',   # ꦐ
    '\uA991': 'ki',   # ꦑ (ka mahaprana)
    '\uA992': 'ga',   # ꦒ
    '\uA993': 'gha',  # ꦓ
    '\uA994': 'nga',  # ꦔ
    '\uA995': 'ca',   # ꦕ
    '\uA996': 'cha',  # ꦖ
    '\uA997': 'ja',   # ꦗ
    '\uA998': 'jha',  # ꦘ (ja mahaprana)
    '\uA999': 'jnya', # ꦙ
    '\uA99A': 'nya',  # ꦚ
    '\uA99B': 'tta',  # ꦛ (tha)
    '\uA99C': 'ttha', # ꦜ
    '\uA99D': 'dda',  # ꦝ (dha)
    '\uA99E': 'ddha', # ꦞ
    '\uA99F': 'nna',  # ꦟ
    '\uA9A0': 'ta',   # ꦠ
    '\uA9A1': 'tha',  # ꦡ
    '\uA9A2': 'da',   # ꦢ
    '\uA9A3': 'dha',  # ꦣ
    '\uA9A4': 'na',   # ꦤ
    '\uA9A5': 'pa',   # ꦥ
    '\uA9A6': 'pha',  # ꦦ
    '\uA9A7': 'ba',   # ꦧ
    '\uA9A8': 'bha',  # ꦨ
    '\uA9A9': 'ma',   # ꦩ
    '\uA9AA': 'ya',   # ꦪ
    '\uA9AB': 'ra',   # ꦫ
    '\uA9AC': 'rra',  # ꦬ
    '\uA9AD': 'la',   # ꦭ
    '\uA9AE': 'wa',   # ꦮ
    '\uA9AF': 'sha',  # ꦯ
    '\uA9B0': 'ssa',  # ꦰ
    '\uA9B1': 'sa',   # ꦱ
    '\uA9B2': 'ha',   # ꦲ
}

# Independent vowels
INDEPENDENT_VOWEL_MAP = {
    '\uA984': 'a',    # ꦄ
    '\uA985': 'aa',   # ꦅ
    '\uA986': 'i',    # ꦆ
    '\uA987': 'ii',   # ꦇ
    '\uA988': 'u',    # ꦈ
    '\uA989': 'uu',   # ꦉ
    '\uA98A': 'rre',  # ꦊ
    '\uA98B': 'rro',  # ꦋ
    '\uA98C': 'e',    # ꦌ
    '\uA98D': 'ai',   # ꦍ
    '\uA98E': 'o',    # ꦎ
}

# Dependent vowel signs — modify the inherent 'a'
VOWEL_SIGN_MAP = {
    '\uA9B4': 'aa',   # ꦴ tarung
    '\uA9B5': 'aa',   # ꦵ
    '\uA9B6': 'i',    # ꦶ wulu
    '\uA9B7': 'ii',   # ꦷ wulu melik
    '\uA9B8': 'u',    # ꦸ suku
    '\uA9B9': 'uu',   # ꦹ suku mendut
    '\uA9BA': 'e',    # ꦺ taling (é)
    '\uA9BB': 'ai',   # ꦻ dirga mure
    '\uA9BC': 'e',    # ꦼ pepet (ě)
}

# Final signs (cecak, layar, wignyan)
FINAL_SIGN_MAP = {
    '\uA981': 'ng',   # cecak ꦁ
    '\uA982': 'r',    # layar ꦂ
    '\uA983': 'h',    # wignyan ꦃ
}

PANGKON = '\uA9C0'       # ꧀ virama/killer
CECAK_TELU = '\uA9B3'    # ꦳ cecak telu (foreign sounds)

# Javanese digits
DIGIT_MAP = {chr(0xA9D0 + i): str(i) for i in range(10)}

# Punctuation
PUNCTUATION_MAP = {
    '\uA9C1': '.',    # ꧁ (pada adeg-adeg, but we simplify)
    '\uA9C2': '.',    # ꧂
    '\uA9C3': ',',    # ꧃
    '\uA9C4': ',',    # ꧄
    '\uA9C5': '"',    # ꧅
    '\uA9C6': '"',    # ꧆
    '\uA9C7': '.',    # ꧇
    '\uA9C8': '.',    # ꧈
    '\uA9C9': '.',    # ꧉
    '\uA9CA': '.',    # ꧊
    '\uA9CB': '.',    # ꧋
    '\uA9CC': '.',    # ꧌
    '\uA9CD': '.',    # ꧍
    '\uA9CF': '.',    # ꧏ
}


def _is_javanese_consonant(c: str) -> bool:
    return '\uA98F' <= c <= '\uA9B2'

def _is_javanese_vowel_sign(c: str) -> bool:
    return '\uA9B4' <= c <= '\uA9BC'

def _is_javanese_independent_vowel(c: str) -> bool:
    return '\uA984' <= c <= '\uA98E'

def _is_javanese_final_sign(c: str) -> bool:
    return c in FINAL_SIGN_MAP

def _is_javanese_digit(c: str) -> bool:
    return '\uA9D0' <= c <= '\uA9D9'


def transliterate_to_latin(text: str) -> str:
    """Convert Javanese script text to Latin transliteration."""
    result = []
    i = 0
    chars = list(text)
    n = len(chars)

    while i < n:
        c = chars[i]

        # Whitespace and ASCII pass through
        if c in (' ', '\n', '\t', '\r'):
            result.append(c)
            i += 1
            continue

        # Javanese digits
        if _is_javanese_digit(c):
            result.append(DIGIT_MAP[c])
            i += 1
            continue

        # Javanese punctuation
        if c in PUNCTUATION_MAP:
            result.append(PUNCTUATION_MAP[c])
            i += 1
            continue

        # Independent vowels
        if _is_javanese_independent_vowel(c):
            latin = INDEPENDENT_VOWEL_MAP.get(c, c)
            i += 1
            # Check for vowel signs after independent vowel
            while i < n and _is_javanese_vowel_sign(chars[i]):
                latin += VOWEL_SIGN_MAP.get(chars[i], '')
                i += 1
            # Check for final signs
            while i < n and _is_javanese_final_sign(chars[i]):
                latin += FINAL_SIGN_MAP[chars[i]]
                i += 1
            result.append(latin)
            continue

        # Base consonant
        if _is_javanese_consonant(c):
            base = CONSONANT_MAP.get(c, c)
            consonant_part = base[:-1] if base.endswith('a') and len(base) > 1 else base
            i += 1

            # Skip cecak telu (foreign marker)
            if i < n and chars[i] == CECAK_TELU:
                # Add 'f'/'v'/'z' mapping hint
                if c == '\uA9A5':  # pa + cecak telu = fa
                    consonant_part = 'f'
                elif c == '\uA9A2':  # da + cecak telu = dza
                    consonant_part = 'dz'
                elif c == '\uA9A7':  # ba + cecak telu = va
                    consonant_part = 'v'
                elif c == '\uA98F':  # ka + cecak telu = kha
                    consonant_part = 'kh'
                elif c == '\uA992':  # ga + cecak telu = gha
                    consonant_part = 'gh'
                elif c == '\uA997':  # ja + cecak telu = z
                    consonant_part = 'z'
                i += 1

            # Check what follows
            if i < n and chars[i] == PANGKON:
                # Pangkon: kill the inherent vowel
                i += 1
                # Check if followed by another consonant (conjunct)
                if i < n and _is_javanese_consonant(chars[i]):
                    result.append(consonant_part)
                    # Don't consume the next consonant, let the loop handle it
                else:
                    # Final pangkon — dead consonant at end
                    result.append(consonant_part)
                continue

            # Check for dependent vowel signs
            vowel = 'a'  # inherent vowel
            if i < n and _is_javanese_vowel_sign(chars[i]):
                # taling (ꦺ) + tarung (ꦴ) = o
                if chars[i] == '\uA9BA' and i + 1 < n and chars[i + 1] == '\uA9B4':
                    vowel = 'o'
                    i += 2
                else:
                    vowel = VOWEL_SIGN_MAP.get(chars[i], 'a')
                    i += 1

            result.append(consonant_part + vowel)

            # Check for final signs (cecak, layar, wignyan)
            while i < n and _is_javanese_final_sign(chars[i]):
                result.append(FINAL_SIGN_MAP[chars[i]])
                i += 1

            continue

        # Final signs standalone
        if _is_javanese_final_sign(c):
            result.append(FINAL_SIGN_MAP[c])
            i += 1
            continue

        # Pangkon standalone (shouldn't happen normally)
        if c == PANGKON:
            i += 1
            continue

        # Cecak telu standalone
        if c == CECAK_TELU:
            i += 1
            continue

        # Vowel signs standalone (shouldn't happen, but handle)
        if _is_javanese_vowel_sign(c):
            result.append(VOWEL_SIGN_MAP.get(c, ''))
            i += 1
            continue

        # Pass through anything else (Latin chars, symbols, etc.)
        result.append(c)
        i += 1

    return ''.join(result)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        text = ' '.join(sys.argv[1:])
    else:
        text = sys.stdin.read()
    print(transliterate_to_latin(text))
