from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tokenizer_jawa import (
    BaseTokenizer,
    ConstrainedSyllableBPETokenizer,
    compute_text_metrics,
    filter_to_javanese_text,
    segment_javanese,
    tokenizer_from_name,
)


def pick_storage_dtype(vocab_size: int) -> np.dtype:
    if vocab_size <= np.iinfo(np.uint16).max:
        return np.uint16
    if vocab_size <= np.iinfo(np.uint32).max:
        return np.uint32
    raise ValueError(f"Vocabulary too large for uint32 storage: {vocab_size}")


def build_tokenizer(args: argparse.Namespace) -> BaseTokenizer:
    if args.tokenizer == "syllable_bpe":
        return tokenizer_from_name(
            args.tokenizer,
            target_vocab_size=args.bpe_target_vocab_size,
            min_pair_freq=args.bpe_min_pair_freq,
            max_merges=args.bpe_max_merges,
        )
    return tokenizer_from_name(args.tokenizer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Hanacaraka-only corpus for a nanoGPT-style model.")
    parser.add_argument("--input", type=Path, required=True, help="UTF-8 text corpus.")
    parser.add_argument("--out_dir", type=Path, required=True, help="Output directory.")
    parser.add_argument(
        "--tokenizer",
        choices=["syllable_bpe", "syllable", "char"],
        default="syllable_bpe",
        help=(
            "syllable_bpe = constrained BPE over orthographic syllables; "
            "syllable = pure orthographic syllables; char = codepoint baseline"
        ),
    )
    parser.add_argument("--val_frac", type=float, default=0.1, help="Validation split fraction.")
    parser.add_argument("--keep_non_javanese", action="store_true", help="Do not filter to Hanacaraka-only.")
    parser.add_argument("--keep_ascii_punct", action="store_true", help="Keep ASCII punctuation too.")
    parser.add_argument("--write_clean_text", action="store_true", help="Write cleaned.txt for inspection.")
    parser.add_argument(
        "--bpe_target_vocab_size",
        type=int,
        default=4096,
        help="Target vocabulary size for syllable_bpe.",
    )
    parser.add_argument(
        "--bpe_min_pair_freq",
        type=int,
        default=2,
        help="Minimum pair frequency to merge in syllable_bpe.",
    )
    parser.add_argument(
        "--bpe_max_merges",
        type=int,
        default=None,
        help="Optional hard cap on the number of merges for syllable_bpe.",
    )
    args = parser.parse_args()

    if not 0.0 < args.val_frac < 0.5:
        raise ValueError("--val_frac should be in (0, 0.5).")
    if args.bpe_target_vocab_size <= 0:
        raise ValueError("--bpe_target_vocab_size must be > 0.")
    if args.bpe_min_pair_freq <= 1:
        raise ValueError("--bpe_min_pair_freq should be >= 2.")
    if args.bpe_max_merges is not None and args.bpe_max_merges < 0:
        raise ValueError("--bpe_max_merges must be >= 0 when provided.")

    raw_text = args.input.read_text(encoding="utf-8")
    clean_text = filter_to_javanese_text(
        raw_text,
        keep_whitespace=True,
        keep_ascii_punct=args.keep_ascii_punct,
        keep_non_javanese=args.keep_non_javanese,
    )

    tokenizer = build_tokenizer(args)
    freq = tokenizer.fit_text(clean_text)
    tokens = tokenizer.segment(clean_text)
    ids = tokenizer.encode_tokens(tokens, allow_unk=False)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    arr_dtype = pick_storage_dtype(len(tokenizer))
    ids_arr = np.asarray(ids, dtype=arr_dtype)
    split_idx = int(len(ids_arr) * (1.0 - args.val_frac))
    train_ids = ids_arr[:split_idx]
    val_ids = ids_arr[split_idx:]

    (out_dir / "train.bin").write_bytes(train_ids.tobytes())
    (out_dir / "val.bin").write_bytes(val_ids.tobytes())
    tokenizer.save(out_dir / "tokenizer.json")

    if args.write_clean_text:
        (out_dir / "cleaned.txt").write_text(clean_text, encoding="utf-8")

    text_metrics = compute_text_metrics(clean_text).to_dict()
    top_tokens = [
        {"token": tok, "count": count}
        for tok, count in sorted(freq.items(), key=lambda item: (-item[1], item[0]))[:50]
    ]

    atomic_syllables = segment_javanese(clean_text, keep_whitespace=True, keep_unknown=False)
    atomic_token_count = len(atomic_syllables)
    compression = {
        "atomic_syllable_count": atomic_token_count,
        "final_token_count": int(ids_arr.size),
        "atomic_to_final_ratio": (atomic_token_count / int(ids_arr.size)) if ids_arr.size else 0.0,
    }

    tokenizer_config: dict[str, object] = {
        "type": tokenizer.tokenizer_type,
    }
    if isinstance(tokenizer, ConstrainedSyllableBPETokenizer):
        tokenizer_config.update(
            {
                "atomic_unit": "orthographic_syllable",
                "target_vocab_size": tokenizer.target_vocab_size,
                "min_pair_freq": tokenizer.min_pair_freq,
                "max_merges": tokenizer.max_merges,
                "merge_count": len(tokenizer.merges),
            }
        )

    meta = {
        "tokenizer_type": tokenizer.tokenizer_type,
        "tokenizer_config": tokenizer_config,
        "vocab_size": len(tokenizer),
        "storage_dtype": np.dtype(arr_dtype).name,
        "train_num_ids": int(train_ids.size),
        "val_num_ids": int(val_ids.size),
        "total_num_ids": int(ids_arr.size),
        "input_file": str(args.input),
        "write_clean_text": args.write_clean_text,
        "cleaning": {
            "keep_non_javanese": args.keep_non_javanese,
            "keep_ascii_punct": args.keep_ascii_punct,
            "keep_whitespace": True,
        },
        "metrics": text_metrics,
        "compression": compression,
        "tokenizer_json": "tokenizer.json",
        "train_bin": "train.bin",
        "val_bin": "val.bin",
        "top_tokens": top_tokens,
        "sample_tokens": tokens[:100],
        "sample_atomic_syllables": atomic_syllables[:100],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Prepared dataset")
    print(f"  input             : {args.input}")
    print(f"  out_dir           : {out_dir}")
    print(f"  tokenizer         : {tokenizer.tokenizer_type}")
    print(f"  vocab_size        : {len(tokenizer)}")
    print(f"  total_ids         : {ids_arr.size}")
    print(f"  train_ids         : {train_ids.size}")
    print(f"  val_ids           : {val_ids.size}")
    print(f"  storage_dtype     : {np.dtype(arr_dtype).name}")
    print(f"  atomic_syllables  : {atomic_token_count}")
    print(f"  atomic/final ratio: {compression['atomic_to_final_ratio']:.4f}")
    if isinstance(tokenizer, ConstrainedSyllableBPETokenizer):
        print(f"  bpe_merge_count   : {len(tokenizer.merges)}")
    print(f"  invalid_token_rate: {text_metrics['invalid_token_rate']:.6f}")
    print(f"  orphan_mark_rate  : {text_metrics['orphan_mark_rate']:.6f}")
    print(f"  dotted_circle_rate: {text_metrics['dotted_circle_rate']:.6f}")


if __name__ == "__main__":
    main()
