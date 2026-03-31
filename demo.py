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


def generate(prompt: str, max_tokens: int, temperature: float, top_k: int) -> tuple[str, str]:
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

    with torch.no_grad():
        out = model.generate(
            idx,
            max_new_tokens=int(max_tokens),
            temperature=float(temperature),
            top_k=int(top_k),
        )

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
        f"{model.get_num_params():,} params) trained on 176K sentences from "
        "Leipzig Corpora, UD Javanese-CSUI, and Javanese Wikisource."
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
            temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.05, label="Temperature (0.7=safe, 0.8=balanced, 1.0=creative)")
            top_k = gr.Slider(1, 200, value=40, step=1, label="Top-k (30=safe, 40=balanced, 80=creative)")
            btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=2):
            output = gr.Textbox(label="Output", lines=12)
            stats = gr.Textbox(label="Stats", lines=1)

    btn.click(fn=generate, inputs=[prompt, max_tokens, temperature, top_k], outputs=[output, stats])

    gr.Examples(
        examples=[
            ["\ua9b2", 200, 0.85, 50],
            ["\ua9b2\ua9a4\ua995\ua9ab\ua98f", 200, 0.85, 50],  # ꦲꦤꦕꦫꦏ
            ["\ua9a5\ua9a2\ua997\ua9aa\ua99a", 200, 0.9, 40],   # ꦥꦢꦗꦪꦚ
        ],
        inputs=[prompt, max_tokens, temperature, top_k],
    )

if __name__ == "__main__":
    app.launch(share=False)
