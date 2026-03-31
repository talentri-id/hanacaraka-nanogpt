"""Gradio demo for Hanacaraka nanoGPT."""
from __future__ import annotations

import torch
import gradio as gr

from model import GPT, GPTConfig
from tokenizer_jawa import BaseTokenizer, compute_text_metrics

CHECKPOINT = "runs/full_v2/best.pt"
TOKENIZER_JSON = "corpora/full/prepared/tokenizer.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
checkpoint = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
config = GPTConfig(**checkpoint["model_args"])
model = GPT(config)
model.load_state_dict(checkpoint["model"])
model.eval().to(DEVICE)

# Load tokenizer
tokenizer = BaseTokenizer.load(TOKENIZER_JSON)

print(f"Model loaded: {config.n_layer}L/{config.n_head}H/{config.n_embd}E, {model.get_num_params():,} params")
print(f"Device: {DEVICE}")


def generate(
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    min_p: float,
    repetition_penalty: float,
    xtc_threshold: float,
    xtc_probability: float,
) -> tuple[str, str]:
    if not prompt.strip():
        prompt = "\ua9b2"  # ꦲ (ha)

    try:
        tokens = tokenizer.segment(prompt)
        ids = tokenizer.encode_tokens(tokens, allow_unk=False)
    except KeyError:
        return "Error: prompt contains tokens not in vocabulary. Use Javanese script.", ""

    if not ids:
        return "Error: prompt produced zero tokens. Use Javanese script (Hanacaraka).", ""

    idx = torch.tensor(ids, dtype=torch.long, device=DEVICE)[None, :]

    gen_kwargs = {
        "temperature": float(temperature),
        "top_k": int(top_k) if top_k > 0 else None,
        "repetition_penalty": float(repetition_penalty),
    }
    if top_p < 1.0:
        gen_kwargs["top_p"] = float(top_p)
    if min_p > 0.0:
        gen_kwargs["min_p"] = float(min_p)
    if xtc_threshold > 0.0 and xtc_probability > 0.0:
        gen_kwargs["xtc_threshold"] = float(xtc_threshold)
        gen_kwargs["xtc_probability"] = float(xtc_probability)

    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=int(max_tokens), **gen_kwargs)

    text = tokenizer.decode_ids(out[0].tolist())
    metrics = compute_text_metrics(text).to_dict()
    stats = (
        f"Tokens: {metrics['total_tokens']} | "
        f"Syllables: {metrics['syllable_like_tokens']} | "
        f"Invalid: {metrics['invalid_tokens']} ({metrics['invalid_token_rate']:.4f})"
    )
    return text, stats


with gr.Blocks(title="Hanacaraka nanoGPT") as app:
    gr.Markdown("# Hanacaraka nanoGPT")
    gr.Markdown(
        "Generate text in **Javanese script (Aksara Jawa)** using a "
        f"GPT model ({config.n_layer}L/{config.n_head}H/{config.n_embd}E, "
        f"{model.get_num_params():,} params) trained from scratch on 176K sentences."
    )

    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(
                label="Prompt (Hanacaraka)",
                value="\ua9b2",
                placeholder="Enter Javanese script...",
                lines=2,
            )
            max_tokens = gr.Slider(32, 512, value=128, step=16, label="Max tokens")

            gr.Markdown("### Sampling")
            temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.05, label="Temperature")
            top_k = gr.Slider(0, 200, value=40, step=1, label="Top-k (0=off)")
            top_p = gr.Slider(0.0, 1.0, value=1.0, step=0.01, label="Top-p / nucleus (1.0=off)")
            min_p = gr.Slider(0.0, 0.5, value=0.0, step=0.01, label="Min-p (0=off)")

            gr.Markdown("### Anti-repetition")
            repetition_penalty = gr.Slider(1.0, 2.0, value=1.2, step=0.05, label="Repetition penalty (1.0=off, 1.2=recommended)")

            gr.Markdown("### XTC (Exclude Top Choices)")
            xtc_threshold = gr.Slider(0.0, 0.5, value=0.0, step=0.01, label="XTC threshold (0=off)")
            xtc_probability = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="XTC probability (0=off)")

            btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=2):
            output = gr.Textbox(label="Output", lines=16)
            stats = gr.Textbox(label="Stats", lines=1)

    all_inputs = [prompt, max_tokens, temperature, top_k, top_p, min_p, repetition_penalty, xtc_threshold, xtc_probability]
    btn.click(fn=generate, inputs=all_inputs, outputs=[output, stats])

    gr.Examples(
        examples=[
            # prompt, max_tokens, temp, top_k, top_p, min_p, rep_pen, xtc_thresh, xtc_prob
            ["\ua9b2", 128, 0.8, 40, 1.0, 0.0, 1.2, 0.0, 0.0],  # Best balanced
            ["\ua9b2", 128, 0.8, 40, 1.0, 0.0, 1.5, 0.0, 0.0],  # Max diversity
            ["\ua9b2", 128, 0.85, 50, 0.92, 0.0, 1.15, 0.0, 0.0],  # Combined
            ["\ua9b2", 128, 0.8, 0, 1.0, 0.05, 1.2, 0.1, 0.3],  # XTC + min-p
        ],
        inputs=all_inputs,
    )

if __name__ == "__main__":
    app.launch(share=False)
