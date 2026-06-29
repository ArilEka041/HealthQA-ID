"""Konfigurasi terpusat untuk evaluator diagnosis HealthQA-ID."""
from __future__ import annotations
import os
from dotenv import load_dotenv
load_dotenv(override=False)



# ---------------------------------------------------------------------------
# Path file
# ---------------------------------------------------------------------------

DATA_FILE = "data/HealthQA-ID.json"
RESULTS_FILE = "healthqa_results.json"
RESULTS_DIR = "results"

# ---------------------------------------------------------------------------
# Rate limiter
# Batas ini dipakai oleh satu rate limiter bersama untuk SEMUA panggilan model
# (jawaban target dan LLM-as-a-judge), sehingga total tidak pernah > RPM_LIMIT.
# ---------------------------------------------------------------------------

RPM_LIMIT = int(os.getenv("HEALTHQA_RPM_LIMIT", "2"))

# ---------------------------------------------------------------------------
# Label triage yang valid
# ---------------------------------------------------------------------------

TRIAGE_LABELS = ("rendah", "menengah", "darurat")

# ---------------------------------------------------------------------------
# Field yang wajib ada di setiap item dataset JSON
# ---------------------------------------------------------------------------

EXPECTED_FIELDS = (
    "index",
    "question_type",
    "question",
    "follow_up",
    "triage",
    "ground_truth",
)

FOLLOW_UP_EXPECTED_FIELDS = ("pertanyaan", "jawaban")

# ---------------------------------------------------------------------------
# Bobot metrik final (harus berjumlah 1.0)
# ---------------------------------------------------------------------------

METRIC_WEIGHTS = {
    "diagnosis_accuracy": 0.40,
    "diagnosis_reasoning": 0.25,
    "follow_up_coverage": 0.20,
    "triage_accuracy": 0.15,
}

# ---------------------------------------------------------------------------
# Thinking level yang valid untuk Google provider
# Nilai: "none" | "low" | "medium" | "high"
# Referensi: https://ai.google.dev/gemini-api/docs/thinking
# ---------------------------------------------------------------------------

THINKING_LEVELS = ("none", "low", "medium", "high")

DEFAULT_THINKING_LEVEL = os.getenv("HEALTHQA_THINKING_LEVEL", "none")

# ---------------------------------------------------------------------------
# Registry model
# Setiap entry mendefinisikan provider dan nama model sesungguhnya.
# Slug dapat di-override via env var tanpa menyentuh kode.
#
# Field opsional `thinking_level` hanya berlaku untuk provider "google".
# Jika tidak di-set, nilai DEFAULT_THINKING_LEVEL dipakai saat runtime.
# Override per-run juga bisa dilakukan via env var HEALTHQA_THINKING_LEVEL.
# ---------------------------------------------------------------------------

MODEL_SPECS = {
    "gemini-2.5-flash": {
        "display_name": "Gemini 2.5 Flash",
        "provider": "google",
        "model": os.getenv("GOOGLE_MODEL_GEMINI_25_FLASH", "gemini-2.5-flash"),
        "thinking_level": os.getenv("THINKING_LEVEL_GEMINI_25_FLASH", "high"),
    },
    "gemini-2.5-flash-lite": {
        "display_name": "Gemini 2.5 Flash Lite",
        "provider": "google",
        "model": os.getenv("GOOGLE_MODEL_GEMINI_25_FLASH_LITE", "gemini-2.5-flash-lite"),
        "thinking_level": os.getenv("THINKING_LEVEL_GEMINI_25_FLASH_LITE", "none"),
    },
    "gemini-3.1-flash-lite": {
        "display_name": "Gemini 3.1 Flash Lite",
        "provider": "google",
        "model": os.getenv("GOOGLE_MODEL_GEMINI_31_FLASH_LITE", "gemini-3.1-flash-lite"),
        "thinking_level": os.getenv("THINKING_LEVEL_GEMINI_31_FLASH_LITE", "none"),
    },
    "gemini-3.5-flash": {
        "display_name": "Gemini 3.5 Flash",
        "provider": "google",
        "model": os.getenv("GOOGLE_MODEL_GEMINI_35_FLASH", "gemini-3.5-flash"),
        "thinking_level": os.getenv("THINKING_LEVEL_GEMINI_35_FLASH", "high"),
    },
    "gemma4-e4b-local": {
        "display_name": "gemma4:e4b (lokal)",
        "provider": "ollama_local",
        "model": os.getenv("OLLAMA_MODEL_GEMMA4_E4B", "gemma4:e4b"),
        # thinking_level tidak berlaku untuk provider ollama_local
    },
}

# ---------------------------------------------------------------------------
# Model default untuk pengujian dan judge
# ---------------------------------------------------------------------------

DEFAULT_MODEL = os.getenv("HEALTHQA_MODEL", "gemini-2.5-flash")

# deepseek-r1:8b dipilih sebagai judge lokal default karena merupakan model
# reasoning terbesar dari model lokal yang tersedia (model reasoning).
# Jika tidak tersedia atau gagal, evaluator akan fallback ke model yang sedang diuji.
DEFAULT_JUDGE_MODEL = os.getenv("HEALTHQA_JUDGE_MODEL", "gemma4-e4b-local")

# ---------------------------------------------------------------------------
# Base URL untuk Ollama
# ---------------------------------------------------------------------------

OLLAMA_LOCAL_BASE_URL = os.getenv("OLLAMA_LOCAL_BASE_URL", "http://localhost:11434/api")
OLLAMA_API_BASE_URL = os.getenv("OLLAMA_API_BASE_URL", "https://ollama.com/api")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def get_model_spec(model_key: str) -> dict:
    """Kembalikan salinan spesifikasi model atau raise dengan pesan yang jelas."""
    try:
        spec = MODEL_SPECS[model_key]
    except KeyError as exc:
        choices = ", ".join(MODEL_SPECS)
        raise ValueError(f"Model '{model_key}' tidak dikenal. Pilihan: {choices}") from exc
    return {"key": model_key, **spec}


def get_thinking_level(model_key: str) -> str | None:
    """
    Kembalikan thinking_level efektif untuk model tertentu.

    Prioritas (tertinggi ke terendah):
      1. Env var HEALTHQA_THINKING_LEVEL  — override global saat runtime
      2. Field 'thinking_level' di MODEL_SPECS[model_key]
      3. DEFAULT_THINKING_LEVEL           — fallback terakhir ("none")

    Mengembalikan None jika provider model tidak mendukung thinking_level
    (ollama_local, openai, dll.) agar caller tidak meneruskan parameter
    yang tidak dikenal ke provider tersebut.
    """
    spec = get_model_spec(model_key)

    # Hanya Google provider yang mendukung thinking_level
    if spec.get("provider") != "google":
        return None

    # Env var global menang atas konfigurasi per-model
    runtime_override = os.getenv("HEALTHQA_THINKING_LEVEL")
    if runtime_override:
        level = runtime_override.lower()
    else:
        level = spec.get("thinking_level", DEFAULT_THINKING_LEVEL)

    if level not in THINKING_LEVELS:
        raise ValueError(
            f"thinking_level '{level}' tidak valid untuk model '{model_key}'. "
            f"Pilihan: {', '.join(THINKING_LEVELS)}"
        )
    return level


def build_google_generation_config(model_key: str, **extra) -> dict:
    """
    Bangun dict generation_config siap pakai untuk Google genai client.

    Contoh penggunaan:
        interaction = client.interactions.create(
            model=spec["model"],
            input=prompt,
            generation_config=build_google_generation_config("gemini-2.5-flash"),
        )

    Parameter tambahan (mis. temperature, max_output_tokens) bisa diteruskan
    via **extra dan akan di-merge ke dalam config.
    """
    config: dict = {}

    thinking_level = get_thinking_level(model_key)
    if thinking_level and thinking_level != "none":
        config["thinking_level"] = thinking_level

    config.update(extra)
    return config