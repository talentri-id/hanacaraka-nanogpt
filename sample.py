from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model import GPT, GPTConfig
from tokenizer_jawa import BaseTokenizer, compute_text_metrics


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample text from a trained Hanacaraka nanoGPT.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="best.pt or last.pt")
    parser.add_argument("--tokenizer_json", type=Path, required=True)
    parser.add_argument("--prompt", type=str, default="ꦲ")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = choose_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = GPTConfig(**checkpoint["model_args"])
    model = GPT(config)
    model.load_state_dict(checkpoint["model"])
    model.eval().to(device)

    tokenizer = BaseTokenizer.load(args.tokenizer_json)
    prompt_tokens = tokenizer.segment(args.prompt)
    prompt_ids = tokenizer.encode_tokens(prompt_tokens, allow_unk=False)
    if not prompt_ids:
        raise ValueError("Prompt produced zero tokens after Hanacaraka filtering; use a non-empty Javanese prompt.")

    idx = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, :]

    print(f"Checkpoint   : {args.checkpoint}")
    print(f"Device       : {device}")
    print(f"Tokenizer    : {tokenizer.tokenizer_type}")
    print(f"Prompt       : {args.prompt}")
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Prompt ids   : {prompt_ids}\n")

    with torch.no_grad():
        for sample_i in range(args.num_samples):
            out = model.generate(
                idx.clone(),
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            text = tokenizer.decode_ids(out[0].tolist())
            metrics = compute_text_metrics(text).to_dict()
            print(f"=== SAMPLE {sample_i + 1} ===")
            print(text)
            print(json.dumps(metrics, ensure_ascii=False, indent=2))
            print()


if __name__ == "__main__":
    main()
