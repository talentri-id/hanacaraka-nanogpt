# hanacaraka-nanogpt

A small GPT language model trained to generate text in **Javanese script (Hanacaraka/Aksara Jawa)**, Unicode block U+A980--A9DF.

The project provides an end-to-end pipeline: corpus collection, Latin-to-Javanese transliteration, syllable-aware tokenization, training, and sampling.

## Pipeline overview

```
[Leipzig / UD / Wikisource]
        |
   build_corpus.py      -- download, clean, transliterate Latin Jawa -> Hanacaraka
        |
   prepare.py            -- tokenize (syllable BPE), split train/val, write .bin
        |
   train.py              -- train a GPT model (nanoGPT-style)
        |
   sample.py             -- generate Hanacaraka text from a checkpoint
```

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

PyTorch will be installed as CPU-only by default. For CUDA:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

### 2. Build the corpus

```bash
python build_corpus.py \
  --out_dir corpora/full \
  --backend builtin \
  --use_wikisource \
  --prepare_after_build \
  --prepared_bpe_target_vocab_size 16000
```

This downloads Javanese text from Leipzig Corpora, UD Javanese-CSUI, and Javanese Wikisource, cleans it, transliterates Latin to Hanacaraka, and prepares tokenized train/val splits.

Sources:
- **Leipzig Wikipedia Jawa 2021** (~100K sentences)
- **Leipzig Community Jawa 2017** (~75K sentences)
- **UD Javanese-CSUI** (~1K sentences)
- **Javanese Wikisource** (native script pages)

### 3. Prepare data only (if corpus already built)

```bash
python prepare.py \
  --input corpora/full/corpus_hanacaraka.txt \
  --out_dir corpora/full/prepared \
  --tokenizer syllable_bpe \
  --bpe_target_vocab_size 16000 \
  --write_clean_text
```

### 4. Train

```bash
python train.py \
  --data_dir corpora/full/prepared \
  --out_dir runs/my_run \
  --device cuda \
  --batch_size 64 \
  --block_size 128 \
  --n_layer 8 \
  --n_head 8 \
  --n_embd 512 \
  --max_iters 10000 \
  --eval_interval 500 \
  --learning_rate 3e-4 \
  --bias
```

### 5. Sample

```bash
python sample.py \
  --checkpoint runs/my_run/best.pt \
  --tokenizer_json corpora/full/prepared/tokenizer.json \
  --device cuda \
  --temperature 0.85 \
  --top_k 40 \
  --num_samples 5
```

On Windows, prefix commands with `PYTHONIOENCODING=utf-8` to print Javanese characters correctly.

## Tokenizer

Three tokenizer modes are available:

| Mode | Description |
|---|---|
| `syllable_bpe` | Constrained BPE over orthographic syllables (default). Never splits inside a valid Javanese syllable. |
| `syllable` | Pure orthographic syllable segmentation per Unicode TN47. |
| `char` | Codepoint-level baseline. |

## Transliteration backends

| Backend | Description |
|---|---|
| `builtin` | Pure Python fallback. Handles standard consonants, vowels, sandangan, pangkon, rekan. |
| `carakanjs` | Uses [Carakan.js](https://github.com/nicatquliyev/carakan.js) via a Node bridge for higher accuracy. Requires `npm install`. |

## Model

Standard GPT (nanoGPT-style): causal self-attention + MLP + LayerNorm, with flash attention support. Weight tying between token embeddings and output head.

## Project structure

```
build_corpus.py       Corpus builder (Leipzig, UD, Wikisource)
latin_to_javanese.py  Latin-to-Hanacaraka transliteration
tokenizer_jawa.py     Syllable segmentation, BPE, validation
model.py              GPT model definition
prepare.py            Tokenize & split corpus into train/val .bin
train.py              Training loop
sample.py             Text generation from checkpoint
data/                 Sample Hanacaraka data
tests/                Unit tests
```

## Tests

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## License

This project uses publicly available linguistic data from Leipzig Corpora Collection, Universal Dependencies, and Javanese Wikisource.
