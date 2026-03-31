from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from model import GPT, GPTConfig


def get_storage_dtype(name: str) -> np.dtype:
    if name == "uint16":
        return np.uint16
    if name == "uint32":
        return np.uint32
    raise ValueError(f"Unsupported storage dtype: {name}")


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_meta(data_dir: Path) -> dict:
    meta_path = data_dir / "meta.json"
    return json.loads(meta_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny GPT on Hanacaraka tokens.")
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda/mps")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bias", action="store_true", help="Enable Linear/LayerNorm bias.")
    parser.add_argument("--max_iters", type=int, default=2000)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--eval_iters", type=int, default=50)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument("--lr_decay_iters", type=int, default=2000)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--compile", action="store_true", help="Use torch.compile when available.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = choose_device(args.device)
    device_type = "cuda" if device.startswith("cuda") else device

    data_dir = args.data_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_meta(data_dir)
    storage_dtype = get_storage_dtype(meta["storage_dtype"])

    train_data = np.memmap(data_dir / meta["train_bin"], dtype=storage_dtype, mode="r")
    val_data = np.memmap(data_dir / meta["val_bin"], dtype=storage_dtype, mode="r")

    if len(train_data) <= args.block_size + 1 or len(val_data) <= args.block_size + 1:
        raise ValueError(
            "Dataset too small for the chosen block size. Decrease --block_size or use a larger corpus."
        )

    gpt_config = GPTConfig(
        vocab_size=int(meta["vocab_size"]),
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        bias=args.bias,
    )
    model = GPT(gpt_config).to(device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        betas=(args.beta1, args.beta2),
        device_type=device_type,
    )

    args_payload = vars(args).copy()
    args_payload["data_dir"] = str(args_payload["data_dir"])
    args_payload["out_dir"] = str(args_payload["out_dir"])
    (out_dir / "train_args.json").write_text(json.dumps(args_payload, indent=2), encoding="utf-8")

    def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
        data = train_data if split == "train" else val_data
        ix = torch.randint(len(data) - args.block_size - 1, (args.batch_size,))
        x = torch.stack(
            [torch.from_numpy(np.asarray(data[i : i + args.block_size], dtype=np.int64)) for i in ix]
        )
        y = torch.stack(
            [torch.from_numpy(np.asarray(data[i + 1 : i + 1 + args.block_size], dtype=np.int64)) for i in ix]
        )
        x = x.to(device)
        y = y.to(device)
        return x, y

    @torch.no_grad()
    def estimate_loss() -> dict[str, float]:
        model.eval()
        out = {}
        for split in ["train", "val"]:
            losses = torch.zeros(args.eval_iters)
            for k in range(args.eval_iters):
                xb, yb = get_batch(split)
                _, loss = model(xb, yb)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    def get_lr(iter_num: int) -> float:
        if iter_num < args.warmup_iters:
            return args.learning_rate * iter_num / max(1, args.warmup_iters)
        if iter_num > args.lr_decay_iters:
            return args.min_lr
        decay_ratio = (iter_num - args.warmup_iters) / max(1, args.lr_decay_iters - args.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return args.min_lr + coeff * (args.learning_rate - args.min_lr)

    best_val_loss = float("inf")
    running_mfu_note = "n/a"
    t0 = time.time()

    print(f"Device            : {device}")
    print(f"Parameters        : {model.get_num_params(non_embedding=True):,}")
    print(f"Vocab size        : {gpt_config.vocab_size}")
    print(f"Train ids         : {len(train_data):,}")
    print(f"Val ids           : {len(val_data):,}")
    print(f"Gradient accum    : {args.gradient_accumulation_steps}")

    model.train()
    for iter_num in range(args.max_iters + 1):
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if iter_num % args.eval_interval == 0 or iter_num == args.max_iters:
            losses = estimate_loss()
            print(
                f"step {iter_num:05d} | train {losses['train']:.4f} | val {losses['val']:.4f} | lr {lr:.6f}"
            )
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
            checkpoint = {
                "model": model.state_dict(),
                "model_args": asdict(gpt_config),
                "iter_num": iter_num,
                "best_val_loss": best_val_loss,
                "meta": meta,
            }
            torch.save(checkpoint, out_dir / "last.pt")
            if losses["val"] <= best_val_loss:
                torch.save(checkpoint, out_dir / "best.pt")

        optimizer.zero_grad(set_to_none=True)
        lossf = 0.0
        for micro_step in range(args.gradient_accumulation_steps):
            xb, yb = get_batch("train")
            _, loss = model(xb, yb)
            loss = loss / args.gradient_accumulation_steps
            lossf += loss.item()
            loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if iter_num % args.log_interval == 0:
            dt = time.time() - t0
            toks_per_iter = args.batch_size * args.block_size * args.gradient_accumulation_steps
            toks_per_sec = toks_per_iter / max(dt, 1e-9)
            print(
                f"iter {iter_num:05d} | loss {lossf:.4f} | lr {lr:.6f} | {toks_per_sec:,.0f} tok/s | mfu {running_mfu_note}"
            )
            t0 = time.time()

    print(f"Done. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
