# hanacaraka-nanogpt

Tiny GPT project for **Hanacaraka / Aksara Jawa** with a **syllable-aware constrained BPE tokenizer**.

This repo keeps the model simple, nanoGPT-style, while making the tokenizer aware of the abugida structure:

- corpus is **Hanacaraka-only** by default
- atomic unit is **Javanese orthographic syllable**
- recommended tokenizer is **`syllable_bpe`**
- merges happen **only across adjacent syllables inside a word-like span**
- tokenizer **never splits inside a syllable**

That gives a better tradeoff than pure syllable tokens when the corpus starts getting bigger, while still respecting the script structure.

## What's inside

- `tokenizer_jawa.py`
  - orthographic-syllable segmentation
  - validation helpers
  - text quality metrics (`invalid_token_rate`, `orphan_mark_rate`, `dotted_circle_rate`)
  - `syllable_bpe`: constrained BPE over syllable boundaries
  - `syllable`: pure orthographic-syllable baseline
  - `char`: codepoint baseline
- `prepare.py`
  - cleans corpus to Hanacaraka-only by default
  - trains tokenizer vocab
  - writes `train.bin`, `val.bin`, `meta.json`, `tokenizer.json`
- `build_corpus.py`
  - corpus builder from Leipzig / UD / optional Wikisource
  - can call `prepare.py` automatically after build
- `model.py`
  - minimal GPT decoder in PyTorch
- `train.py`
  - small training loop
- `sample.py`
  - loads checkpoint and generates text
- `data/sample_hanacaraka.txt`
  - tiny smoke-test corpus
- `tests/test_tokenizer.py`
  - sanity checks for segmentation + constrained BPE behavior

## Tokenization design

### Recommended: `syllable_bpe`

Pipeline:

1. filter to Hanacaraka-only text
2. segment into **orthographic syllables**
3. treat each syllable as an atomic symbol
4. learn **constrained BPE merges** only across adjacent syllables
5. never merge across whitespace, punctuation, digits, or joiners

So the tokenizer gets shorter sequences than pure syllable tokenization, but still stays abugida-aware.

### Baselines

- `syllable`: one token = one orthographic syllable
- `char`: one token = one Unicode codepoint

## Quickstart

Create a venv and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1) Prepare data with the recommended tokenizer

Smoke test with the bundled tiny corpus:

```bash
python prepare.py \
  --input data/sample_hanacaraka.txt \
  --out_dir runs/sample_data_bpe \
  --tokenizer syllable_bpe \
  --bpe_target_vocab_size 128 \
  --bpe_min_pair_freq 2 \
  --val_frac 0.1 \
  --write_clean_text
```

For a real corpus, replace `data/sample_hanacaraka.txt` with your own UTF-8 corpus.

Default behavior:
- keeps Javanese block characters
- keeps whitespace
- drops non-Javanese characters

If you want to keep ASCII punctuation too:

```bash
python prepare.py \
  --input my_corpus.txt \
  --out_dir runs/my_data \
  --tokenizer syllable_bpe \
  --keep_ascii_punct
```

### 2) Train

Tiny CPU-friendly smoke run:

```bash
python train.py \
  --data_dir runs/sample_data_bpe \
  --out_dir runs/sample_ckpt \
  --device cpu \
  --batch_size 8 \
  --block_size 16 \
  --n_layer 2 \
  --n_head 2 \
  --n_embd 32 \
  --dropout 0.0 \
  --max_iters 50 \
  --eval_interval 25 \
  --eval_iters 10
```

A more realistic small GPU run:

```bash
python train.py \
  --data_dir runs/my_data \
  --out_dir runs/my_ckpt \
  --device cuda \
  --batch_size 32 \
  --block_size 128 \
  --n_layer 6 \
  --n_head 6 \
  --n_embd 384 \
  --dropout 0.1 \
  --max_iters 5000 \
  --eval_interval 250 \
  --eval_iters 50
```

### 3) Sample

```bash
python sample.py \
  --checkpoint runs/my_ckpt/best.pt \
  --tokenizer_json runs/my_data/tokenizer.json \
  --prompt "ꦲꦤ" \
  --max_new_tokens 128 \
  --temperature 0.9 \
  --top_k 20
```

## Build corpus + prepare in one go

```bash
python build_corpus.py \
  --out_dir corpora/jawa_ready \
  --backend auto \
  --prepare_after_build \
  --prepared_dir runs/jawa_ready_bpe \
  --prepared_tokenizer syllable_bpe \
  --prepared_bpe_target_vocab_size 4096 \
  --prepared_bpe_min_pair_freq 2
```

## Inspect token efficiency

`prepare.py` now writes extra metadata for the recommended tokenizer:

- `atomic_syllable_count`
- `final_token_count`
- `atomic_to_final_ratio`
- BPE merge count and config

If `atomic_to_final_ratio > 1`, the constrained BPE tokenizer is compressing the original syllable stream.

## Run tests

```bash
python -m unittest tests/test_tokenizer.py
```

## Notes on corpus quality

For a real model, the biggest wins still come from corpus quality.

Good corpus checklist:
- modern, valid Javanese Unicode text
- consistent normalization
- minimal OCR noise
- enough examples with real vowels, medials, and conjuncts
- enough total tokens that the tokenizer learns useful merges instead of memorizing tiny fragments

## Known limitations

- This is a **CBPE-style pure Python implementation**, not an exact reproduction of any specific external tokenizer library.
- `dotted_circle_rate` here is an **explicit-codepoint proxy**. It counts literal U+25CC in text; it does not shape-render samples through HarfBuzz.
- Very small corpora will overfit fast and generate repetitive text.
- If your prompt contains unseen syllables that never appeared in training, strict no-UNK encoding will reject them.

## Files produced by `prepare.py`

- `train.bin`: token ids for training
- `val.bin`: token ids for validation
- `meta.json`: dataset and tokenizer metadata
- `tokenizer.json`: vocab and tokenizer type
- `cleaned.txt`: optional cleaned corpus dump

## Suggested next upgrade

The next good upgrade after this is usually one of these:

1. add AMP / bf16 training for faster runs
2. add a held-out transliteration or spelling benchmark
3. compare `syllable_bpe` vs `syllable` on the same corpus and context length
