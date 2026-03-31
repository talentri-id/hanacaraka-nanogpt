"""Evaluate different sampling strategies to find optimal parameters."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import torch

from model import GPT, GPTConfig
from tokenizer_jawa import BaseTokenizer, compute_text_metrics


def ngram_repetition_rate(tokens: list[str], n: int = 3) -> float:
    """Fraction of n-grams that are repeated."""
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / max(len(ngrams), 1)


def unique_token_ratio(tokens: list[str]) -> float:
    non_ws = [t for t in tokens if t.strip()]
    if not non_ws:
        return 0.0
    return len(set(non_ws)) / len(non_ws)


def evaluate_config(
    model: GPT,
    tokenizer: BaseTokenizer,
    prompt_ids: torch.Tensor,
    device: str,
    *,
    num_samples: int = 5,
    max_tokens: int = 128,
    **gen_kwargs,
) -> dict:
    results = []
    for i in range(num_samples):
        torch.manual_seed(42 + i)
        out = model.generate(
            prompt_ids.clone(),
            max_new_tokens=max_tokens,
            **gen_kwargs,
        )
        text = tokenizer.decode_ids(out[0].tolist())
        tokens = tokenizer.segment(text)
        metrics = compute_text_metrics(text)

        results.append({
            "text": text,
            "total_tokens": metrics.total_tokens,
            "invalid_tokens": metrics.invalid_tokens,
            "invalid_rate": metrics.invalid_token_rate,
            "unique_ratio": unique_token_ratio(tokens),
            "rep3": ngram_repetition_rate(tokens, 3),
            "rep5": ngram_repetition_rate(tokens, 5),
        })

    avg = lambda key: sum(r[key] for r in results) / len(results)
    return {
        "config": gen_kwargs,
        "avg_invalid_rate": avg("invalid_rate"),
        "avg_unique_ratio": avg("unique_ratio"),
        "avg_rep3": avg("rep3"),
        "avg_rep5": avg("rep5"),
        "quality_score": avg("unique_ratio") * (1 - avg("rep3")) * (1 - avg("invalid_rate")),
        "samples": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep sampling parameters to find optimal settings.")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/full_v2/best.pt"))
    parser.add_argument("--tokenizer_json", type=Path, default=Path("corpora/full/prepared/tokenizer.json"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("eval_results.json"))
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = GPTConfig(**checkpoint["model_args"])
    model = GPT(config)
    model.load_state_dict(checkpoint["model"])
    model.eval().to(device)

    tokenizer = BaseTokenizer.load(args.tokenizer_json)
    prompt = "\ua9b2"  # ha
    ids = tokenizer.encode_tokens(tokenizer.segment(prompt))
    prompt_ids = torch.tensor(ids, dtype=torch.long, device=device)[None, :]

    configs = [
        # Baseline: just temperature
        {"temperature": 0.5, "top_k": None},
        {"temperature": 0.7, "top_k": None},
        {"temperature": 0.8, "top_k": None},
        {"temperature": 1.0, "top_k": None},

        # Top-k sweep
        {"temperature": 0.8, "top_k": 10},
        {"temperature": 0.8, "top_k": 20},
        {"temperature": 0.8, "top_k": 40},
        {"temperature": 0.8, "top_k": 80},
        {"temperature": 0.8, "top_k": 150},

        # Top-p sweep
        {"temperature": 0.8, "top_p": 0.7},
        {"temperature": 0.8, "top_p": 0.85},
        {"temperature": 0.8, "top_p": 0.92},
        {"temperature": 0.8, "top_p": 0.95},

        # Min-p sweep
        {"temperature": 0.8, "min_p": 0.02},
        {"temperature": 0.8, "min_p": 0.05},
        {"temperature": 0.8, "min_p": 0.1},

        # Repetition penalty
        {"temperature": 0.8, "top_k": 40, "repetition_penalty": 1.1},
        {"temperature": 0.8, "top_k": 40, "repetition_penalty": 1.2},
        {"temperature": 0.8, "top_k": 40, "repetition_penalty": 1.5},

        # XTC
        {"temperature": 0.8, "top_k": 40, "xtc_threshold": 0.1, "xtc_probability": 0.3},
        {"temperature": 0.8, "top_k": 40, "xtc_threshold": 0.1, "xtc_probability": 0.5},
        {"temperature": 0.8, "top_k": 40, "xtc_threshold": 0.2, "xtc_probability": 0.5},

        # Combined best guesses
        {"temperature": 0.8, "top_k": 40, "top_p": 0.92, "repetition_penalty": 1.1},
        {"temperature": 0.8, "top_p": 0.9, "min_p": 0.05, "repetition_penalty": 1.1},
        {"temperature": 0.85, "top_k": 50, "top_p": 0.92, "repetition_penalty": 1.15},
        {"temperature": 0.8, "min_p": 0.05, "repetition_penalty": 1.2, "xtc_threshold": 0.1, "xtc_probability": 0.3},
    ]

    all_results = []
    for i, cfg in enumerate(configs):
        label = " ".join(f"{k}={v}" for k, v in cfg.items())
        print(f"[{i+1}/{len(configs)}] {label} ... ", end="", flush=True)
        result = evaluate_config(
            model, tokenizer, prompt_ids, device,
            num_samples=args.num_samples,
            max_tokens=args.max_tokens,
            **cfg,
        )
        print(
            f"quality={result['quality_score']:.4f} "
            f"unique={result['avg_unique_ratio']:.3f} "
            f"rep3={result['avg_rep3']:.3f} "
            f"invalid={result['avg_invalid_rate']:.4f}"
        )
        all_results.append(result)

    # Sort by quality score
    ranked = sorted(all_results, key=lambda r: -r["quality_score"])

    print("\n=== TOP 10 CONFIGURATIONS ===")
    for i, r in enumerate(ranked[:10]):
        cfg = " ".join(f"{k}={v}" for k, v in r["config"].items())
        print(
            f"#{i+1}  quality={r['quality_score']:.4f}  "
            f"unique={r['avg_unique_ratio']:.3f}  "
            f"rep3={r['avg_rep3']:.3f}  "
            f"invalid={r['avg_invalid_rate']:.4f}  "
            f"| {cfg}"
        )

    # Save full results (without sample text for brevity)
    save_results = []
    for r in ranked:
        save_r = {k: v for k, v in r.items() if k != "samples"}
        save_r["sample_texts"] = [s["text"] for s in r["samples"][:2]]
        save_results.append(save_r)

    args.output.write_text(json.dumps(save_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nFull results saved to {args.output}")


if __name__ == "__main__":
    main()
