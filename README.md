# HealthQA-ID Evaluator

Toolkit benchmark untuk menguji kemampuan AI model mendiagnosis penyakit
dalam Bahasa Indonesia, menggunakan dataset **HealthQA-ID** (format JSON).

---

## Apa yang diukur?

Setiap respons model dinilai oleh **LLM-as-a-Judge** pada empat dimensi:

| Dimensi                 | Bobot | Deskripsi                                                                   |
| ----------------------- | ----- | --------------------------------------------------------------------------- |
| **Akurasi Diagnosis**   | 40%   | Seberapa tepat diagnosis model dibanding `question_type` ground truth (0–4) |
| **Penalaran Diagnosis** | 25%   | Kualitas alasan klinis yang menghubungkan gejala dengan diagnosis (0–4)     |
| **Cakupan Follow-up**   | 20%   | Proporsi pertanyaan klarifikasi referensi yang tercakup oleh model          |
| **Akurasi Triase**      | 15%   | Ketepatan label triage: `rendah`, `menengah`, atau `darurat` (exact-match)  |

Skor akhir adalah **rata-rata berbobot** keempat dimensi di atas (0.0–1.0).

---

## Model yang diuji

| Kunci                   | Nama                  | Provider     | Thinking default |
| ----------------------- | --------------------- | ------------ | ---------------- |
| `gemini-2.5-flash`      | Gemini 2.5 Flash      | Google API   | `high`           |
| `gemini-2.5-flash-lite` | Gemini 2.5 Flash Lite | Google API   | `none`           |
| `gemini-3.1-flash-lite` | Gemini 3.1 Flash Lite | Google API   | `none`           |
| `gemini-3.5-flash`      | Gemini 3.5 Flash      | Google API   | `high`           |
| `gemma4-e4b-local`      | gemma4:e4b            | Ollama lokal | —                |

**Model default:** `gemini-2.5-flash`  
**Judge default:** `gemma4-e4b-local` (model reasoning lokal).  
Jika judge lokal tidak tersedia, fallback otomatis ke model yang sedang diuji.

---

## Rate Limit

Seluruh panggilan ke semua provider — model yang diuji **dan** judge —
berbagi satu sliding-window rate limiter. Default: **2 RPM**.

Artinya dalam satu menit maksimal ada 2 panggilan API total:
satu untuk mendapat jawaban, satu untuk menilai jawaban tersebut.

Ubah via env var:

```bash
export HEALTHQA_RPM_LIMIT=2   # default
```

---

## Instalasi

```bash
# 1. Buat virtual environment (opsional tapi direkomendasikan)
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

# 2. Pasang dependensi
pip install -r requirements.txt
```

---

## Konfigurasi API Key

Buat file `.env` di root project, atau set env var langsung:

```bash
# Google (Gemini via Google AI Studio atau Vertex)
GOOGLE_API_KEY=your_google_key
# atau
GEMINI_API_KEY=your_gemini_key

# Ollama API (opsional, jika menggunakan Ollama cloud)
OLLAMA_API_KEY=your_ollama_key
```

Model lokal (`gemma4:e4b`) tidak butuh API key —
cukup pastikan Ollama berjalan di `http://localhost:11434`.

---

## Konfigurasi Thinking Level

**Thinking level** hanya berlaku untuk model Google. Nilai yang valid:
`none` | `low` | `medium` | `high`.

### Override global (berlaku untuk semua model di satu run)

```bash
export HEALTHQA_THINKING_LEVEL=high   # paksa semua model pakai thinking high
```

### Override per model (via env var di `.env`)

Setiap model punya env var tersendiri:

```bash
THINKING_LEVEL_GEMINI_25_FLASH=high        # default: high
THINKING_LEVEL_GEMINI_25_FLASH_LITE=none   # default: none
THINKING_LEVEL_GEMINI_31_FLASH_LITE=none   # default: none
THINKING_LEVEL_GEMINI_35_FLASH=high        # default: high
```

### Prioritas (tertinggi ke terendah)

1. `HEALTHQA_THINKING_LEVEL` — override global saat runtime
2. `THINKING_LEVEL_<MODEL>` — konfigurasi per model di `.env`
3. Nilai default di `configs.py` (per model, lihat tabel di atas)

> **Catatan:** Model `gemma4-e4b-local` (Ollama) tidak mendukung
> thinking level — parameter ini diabaikan secara otomatis.

---

## Dataset

Letakkan dataset di `data/HealthQA-ID.json`.
Format yang diharapkan: JSON object dengan key `data` berisi array of objects:

```json
{
    "total": 15,
    "data": [
        {
            "index": 1,
            "question_type": "Appendicitis",
            "question": "Perut kanan bawah saya sakit sejak kemarin...",
            "follow_up": [
                {
                    "pertanyaan": "Apakah ada demam?",
                    "jawaban": ["Ya", "Tidak"]
                }
            ],
            "triage": {
                "level": "darurat",
                "label": "Merah – Segera",
                "deskripsi": "Kondisi mengancam jiwa, perlu penanganan segera."
            }
        }
    ]
}
```

---

## Cara Menjalankan

### Lihat daftar model

```bash
python -m healthqa.evaluate --list
```

### Uji satu model

```bash
# Uji 10 item pertama dengan model default (gemini-2.5-flash)
python -m healthqa.evaluate --limit 10

# Uji model tertentu
python -m healthqa.evaluate --model gemini-2.5-flash-lite --limit 10

# Uji model lokal
python -m healthqa.evaluate --model gemma4-e4b-local --limit 5

# Tentukan judge secara eksplisit
python -m healthqa.evaluate --model gemini-2.5-flash --judge gemma4-e4b-local
```

### Uji semua model sekaligus

```bash
python -m healthqa.evaluate --all --limit 5
```

### Gunakan file dataset lain

```bash
python -m healthqa.evaluate --model gemini-2.5-flash --file data/dataset_lain.json
```

### Mode senyap (kurangi output per item)

```bash
python -m healthqa.evaluate --all --quiet
```

### Gunakan sebagai entrypoint (setelah `pip install -e .`)

```bash
healthqa-eval --model gemini-2.5-flash --limit 10
healthqa-eval --all --limit 5
```

---

## Output

| File                               | Isi                                             |
| ---------------------------------- | ----------------------------------------------- |
| `healthqa_results.json`            | Hasil evaluasi terakhir (selalu diperbarui)     |
| `results/<model>_<timestamp>.json` | Hasil per model + timestamp                     |
| `results/comparison_summary.json`  | Perbandingan ringkas semua model (saat `--all`) |

Struktur `healthqa_results.json`:

```json
{
    "timestamp": "2026-01-01T00:00:00",
    "model_key": "gemini-2.5-flash",
    "model_display": "Gemini 2.5 Flash",
    "judge_key": "gemma4-e4b-local",
    "metrics": {
        "avg_diagnosis_accuracy": 0.75,
        "avg_diagnosis_reasoning": 0.68,
        "avg_follow_up_coverage": 0.6,
        "avg_triage_accuracy": 0.8,
        "avg_composite_score": 0.712,
        "pass_rate": 0.85,
        "triage_accuracy_by_label": { "...": "..." }
    },
    "results_detail": ["..."]
}
```

---

## Override model via env var

Slug model dapat di-override tanpa mengubah kode:

```bash
# Ganti model string yang dipanggil ke Google API
GOOGLE_MODEL_GEMINI_25_FLASH=gemini-2.5-flash-exp
GOOGLE_MODEL_GEMINI_25_FLASH_LITE=gemini-2.5-flash-lite-preview-06-17

# Ganti model default CLI
export HEALTHQA_MODEL=gemini-2.5-flash-lite

# Ganti judge default
export HEALTHQA_JUDGE_MODEL=gemini-2.5-flash-lite
```

---

## Struktur proyek

```
healthqa/
├── __init__.py        # ekspor utama
├── configs.py         # registry model, bobot metrik, konstanta
├── presets.py         # prompt sistem & template untuk model + judge
├── models.py          # gateway Google / Ollama + rate limiter
├── utilities.py       # loader dataset JSON + helper field
├── metrics.py         # judge call, scoring, agregasi
└── evaluate.py        # loop evaluasi utama + CLI

data/
└── HealthQA-ID.json   # ← letakkan dataset di sini

results/               # ← dibuat otomatis saat evaluasi
```

---

## Catatan teknis

- **Rate limiter bersama:** satu `SlidingWindowRateLimiter` singleton digunakan
  oleh model yang diuji dan judge, memastikan total ≤ RPM_LIMIT per menit.
- **Judge fallback:** jika `gemma4-e4b-local` tidak tersedia (Ollama tidak
  jalan atau model belum di-pull), evaluator otomatis menggunakan model yang
  sedang diuji sebagai judge.
- **Parse JSON judge:** jika output judge bukan JSON valid, heuristik regex
  mengekstrak angka dari teks mentah dan mencatat `judge_error` di hasil.
- **Follow-up referensi kosong:** jika item dataset tidak memiliki follow-up
  question, skor `follow_up_coverage` diisi 0.5 (netral).
- **Thinking level:** parameter ini dikirim ke Google genai client hanya jika
  nilainya bukan `none`. Model non-Google mengabaikan parameter ini sepenuhnya.

---

## Lisensi

Gunakan sesuai kebutuhan riset dan pengembangan.
