"""Export the nanoGPT model to ONNX for WebGPU inference."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
import torch.nn as nn

from model import GPT, GPTConfig
from tokenizer_jawa import BaseTokenizer


class GPTForExport(nn.Module):
    """Wrapper that takes token ids and returns logits for the last position."""

    def __init__(self, model: GPT):
        super().__init__()
        self.model = model

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(idx)
        return logits[:, -1, :]  # only last position


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/full_v3_para/best.pt"))
    parser.add_argument("--tokenizer_json", type=Path, default=Path("corpora/full/prepared_para/tokenizer.json"))
    parser.add_argument("--out_dir", type=Path, default=Path("webgpu_demo"))
    parser.add_argument("--quantize_fp16", action="store_true", default=True)
    args = parser.parse_args()

    args.out_dir.mkdir(exist_ok=True)

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = GPTConfig(**checkpoint["model_args"])
    model = GPT(config)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    wrapper = GPTForExport(model)

    print(f"Model: {config.n_layer}L/{config.n_head}H/{config.n_embd}E, {model.get_num_params():,} params")
    print(f"Block size: {config.block_size}")

    # Export to ONNX
    dummy_input = torch.randint(0, config.vocab_size, (1, 16), dtype=torch.long)
    onnx_path = args.out_dir / "model.onnx"

    print(f"Exporting to {onnx_path} ...")
    torch.onnx.export(
        wrapper,
        (dummy_input,),
        str(onnx_path),
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq_len"},
            "logits": {0: "batch"},
        },
        opset_version=18,
    )

    # Quantize to FP16
    if args.quantize_fp16:
        import onnx
        from onnxruntime.transformers.float16 import convert_float_to_float16

        print("Converting to FP16 ...")
        model_fp32 = onnx.load(str(onnx_path))
        model_fp16 = convert_float_to_float16(model_fp32, keep_io_types=True)
        fp16_path = args.out_dir / "model_fp16.onnx"
        onnx.save(model_fp16, str(fp16_path))

        size_32 = onnx_path.stat().st_size / 1024 / 1024
        size_16 = fp16_path.stat().st_size / 1024 / 1024
        print(f"FP32: {size_32:.1f} MB → FP16: {size_16:.1f} MB")

    # Export tokenizer vocab for JS
    tokenizer = BaseTokenizer.load(args.tokenizer_json)
    vocab = dict(tokenizer.stoi)
    id2token = {v: k for k, v in vocab.items()}

    tok_data = {
        "token2id": vocab,
        "id2token": {str(k): v for k, v in id2token.items()},
        "vocab_size": config.vocab_size,
        "block_size": config.block_size,
    }
    tok_path = args.out_dir / "tokenizer.json"
    tok_path.write_text(json.dumps(tok_data, ensure_ascii=False), encoding="utf-8")
    print(f"Tokenizer exported: {len(vocab)} tokens → {tok_path}")

    # Export model config
    model_config = {
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "n_embd": config.n_embd,
        "block_size": config.block_size,
        "vocab_size": config.vocab_size,
        "params": model.get_num_params(),
    }
    (args.out_dir / "config.json").write_text(json.dumps(model_config, indent=2))

    print("Done!")


if __name__ == "__main__":
    main()
