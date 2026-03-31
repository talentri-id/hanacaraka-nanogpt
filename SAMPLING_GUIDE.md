# Sampling Guide for Hanacaraka nanoGPT

Panduan parameter sampling untuk model nanoGPT 30M params yang dilatih dari scratch pada 176K kalimat Jawa.

## Presets

### Creative / Cerita
Untuk generasi kreatif, cerita, narasi bebas. Prioritas: **diversity tinggi, minim repetisi**.

```
temperature: 0.95
top_k: 80
repetition_penalty: 1.4
```
- Quality score: **0.75** | Unique: 76% | Repetition: 1% | Invalid: 0%
- Alternatif lebih liar: `temp=1.0, rep=1.3` (quality 0.72)
- Dengan XTC untuk variasi ekstra: `temp=0.9, k=50, rep=1.3, xtc_threshold=0.1, xtc_prob=0.5`

### Factual / Fakta
Untuk output mirip ensiklopedia/Wikipedia. Prioritas: **koheren, terstruktur**.

```
temperature: 0.7
top_k: 25
repetition_penalty: 1.2
```
- Quality score: **0.60** | Unique: 63% | Repetition: 5% | Invalid: 0%
- Lebih aman: `temp=0.65, top_p=0.85, rep=1.15`
- Tanpa rep penalty: `temp=0.7, k=30` (tapi repetisi naik ke 10%)

### Balanced
Untuk general-purpose. Tidak terlalu repetitif, tidak terlalu liar.

```
temperature: 0.8
top_k: 40
repetition_penalty: 1.2
```
- Quality score: **0.61** | Unique: 64% | Repetition: 5% | Invalid: 0%
- Dengan XTC: `temp=0.8, k=40, rep=1.2, xtc_threshold=0.1, xtc_prob=0.3`

## Parameter Reference

### Temperature
Mengontrol "keacakan" distribusi probabilitas.
- `0.5-0.7`: Konservatif, repetitif, cocok untuk fakta
- `0.8`: Sweet spot balanced
- `0.9-1.0`: Kreatif, beragam
- `>1.1`: Terlalu acak, mulai nonsense

### Top-k
Hanya sampling dari k token teratas.
- `10-20`: Sangat terbatas, repetitif
- `30-50`: Balanced
- `80-150`: Luas, cocok dengan high temperature
- `0` (off): Tidak ada filter, bergantung pada temperature saja

### Top-p (Nucleus Sampling)
Sampling dari token yang cumulative probability-nya <= p.
- `0.85`: Ketat
- `0.92`: Balanced
- `0.95-1.0`: Longgar/off
- Untuk model kecil ini, top-k lebih efektif dari top-p

### Min-p
Buang token yang probabilitasnya < min_p * max_prob.
- `0.02-0.05`: Ringan, buang noise
- `0.1`: Agresif
- Kurang impactful di model ini dibanding rep penalty

### Repetition Penalty
**Parameter paling penting untuk model kecil.** Penalize token yang sudah muncul.
- `1.0`: Off
- `1.1-1.15`: Ringan, masih ada repetisi
- `1.2`: Recommended default
- `1.3-1.4`: Kuat, bagus untuk creative
- `1.5+`: Terlalu kuat, bisa muncul invalid tokens

### XTC (Exclude Top Choices)
Secara random exclude beberapa top token untuk memaksa diversity.
- `threshold=0.1, probability=0.3`: Ringan
- `threshold=0.1, probability=0.5`: Sedang
- `threshold=0.2, probability=0.5`: Agresif
- Paling efektif dikombinasikan dengan repetition penalty

## Tips untuk Model Kecil (nanoGPT)

1. **Repetition penalty adalah game changer.** Model kecil sangat rentan repetisi. Selalu pakai minimal 1.1-1.2.

2. **Jangan pakai temperature terlalu rendah.** Di bawah 0.6 model kecil cenderung stuck di loop yang sama. Lebih baik temp agak tinggi + rep penalty.

3. **Top-k > top-p untuk model kecil.** Distribusi probabilitas model kecil kurang smooth, top-p bisa terlalu agresif atau terlalu longgar. Top-k lebih predictable.

4. **Max tokens 100-150 optimal.** Model 30M params kehilangan koherensi setelah ~150 token. Lebih pendek = lebih bagus.

5. **Kombinasi > single method.** `temp + top_k + rep_penalty` selalu lebih baik dari salah satu saja.

6. **XTC untuk menghindari "mode collapse".** Kalau model selalu menghasilkan output yang mirip-mirip, tambahkan XTC ringan (threshold=0.1, prob=0.3).

7. **Min-p berguna untuk trim noise.** `min_p=0.02-0.05` bersihkan token-token sangat rendah tanpa mengubah output secara signifikan.

## Quality Metric

Score dihitung sebagai: `unique_ratio * (1 - trigram_repetition) * (1 - invalid_rate)`

- **unique_ratio**: Persentase token unik (non-whitespace). Tinggi = beragam.
- **trigram_repetition**: Fraksi trigram yang diulang. Rendah = tidak repetitif.
- **invalid_rate**: Token yang melanggar aturan ortografi Jawa. Harus 0.
