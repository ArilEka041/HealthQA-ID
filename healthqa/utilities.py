"""Utility helper untuk memuat dataset JSON dan menyimpan hasil evaluasi."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from .configs import DATA_FILE, EXPECTED_FIELDS


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_dataset(path: str = DATA_FILE) -> list[dict]:
    """Muat dataset HealthQA-ID dari file JSON.

    Format yang didukung:
    - Array langsung: [{...}, {...}]
    - Wrapper dict:   {"total": N, "data": [{...}, {...}]}
    - Wrapper dict generik: {"key": [{...}, {...}]}

    Field aktual dataset (huruf kecil):
      index, question_type, question, follow_up, triage, ground_truth

    Kembalikan list kosong jika file tidak ditemukan atau format tidak valid.
    """
    p = _find_file(path, extensions=[".json"])
    if p is None:
        print(f"[PERINGATAN] Dataset tidak ditemukan: {path}")
        return []

    try:
        with open(p, encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        print(f"[ERROR] Gagal membaca JSON: {e}")
        return []

    # Normalisasi: jika berupa dict bungkus, ambil nilai list-nya
    # Prioritaskan key "data" (format {"total": N, "data": [...]})
    if isinstance(raw, dict):
        if "data" in raw and isinstance(raw["data"], list):
            raw = raw["data"]
        else:
            for v in raw.values():
                if isinstance(v, list):
                    raw = v
                    break
            else:
                print("[PERINGATAN] Format JSON tidak dikenali (dict tanpa key list).")
                return []

    if not isinstance(raw, list):
        print("[PERINGATAN] Dataset harus berupa JSON array.")
        return []

    # Validasi minimal field — cek nama lowercase sesuai dataset aktual
    ACTUAL_FIELDS = ("index", "question_type", "question", "follow_up", "triage", "ground_truth")
    if raw:
        first_keys = set(raw[0].keys())
        missing = [f for f in ACTUAL_FIELDS if f not in first_keys]
        if missing:
            print(f"[PERINGATAN] Field tidak ditemukan di item pertama: {missing}")
            print(f"             Field yang ada: {sorted(first_keys)}")

    return raw


def _find_file(path: str, extensions: list[str]) -> Optional[Path]:
    """Cari file; jika tidak ada, cari di beberapa lokasi fallback.

    Urutan pencarian:
    1. Path persis seperti yang diberikan (relatif ke CWD atau absolut)
    2. Path relatif terhadap direktori file ini (healthqa/../data/)
    3. Folder data/ di CWD
    4. Folder data/ satu level di atas CWD  ← penting untuk struktur:
         project/
           healthqa/   ← package (.py files)
           data/       ← JSON files
    5. Direktori saat ini (CWD)
    """
    p = Path(path)
    if p.exists():
        return p

    # Coba path relatif terhadap lokasi file ini (healthqa/utilities.py → naik ke project root)
    here = Path(__file__).resolve().parent  # direktori package (healthqa/)
    p_rel = here / path
    if p_rel.exists():
        print(f"[INFO] Menggunakan file: {p_rel}")
        return p_rel

    # Coba satu level di atas direktori package (project root)
    project_root = here.parent
    p_from_root = project_root / path
    if p_from_root.exists():
        print(f"[INFO] Menggunakan file: {p_from_root}")
        return p_from_root

    # Cari folder data/ di CWD
    d_cwd = Path("data")
    if d_cwd.exists():
        for ext in extensions:
            candidates = sorted(d_cwd.glob(f"*{ext}"))
            if candidates:
                print(f"[INFO] Menggunakan file: {candidates[0]}")
                return candidates[0]

    # Cari folder data/ satu level di atas CWD (struktur: project/healthqa/ + project/data/)
    d_parent = Path("..") / "data"
    if d_parent.exists():
        for ext in extensions:
            candidates = sorted(d_parent.glob(f"*{ext}"))
            if candidates:
                print(f"[INFO] Menggunakan file: {candidates[0]}")
                return candidates[0]

    # Cari folder data/ relatif ke project root (package parent)
    d_root = project_root / "data"
    if d_root.exists():
        for ext in extensions:
            candidates = sorted(d_root.glob(f"*{ext}"))
            if candidates:
                print(f"[INFO] Menggunakan file: {candidates[0]}")
                return candidates[0]

    # Coba direktori saat ini
    for ext in extensions:
        candidates = sorted(Path(".").glob(f"*{ext}"))
        if candidates:
            print(f"[INFO] Menggunakan file: {candidates[0]}")
            return candidates[0]

    return None


# ---------------------------------------------------------------------------
# Dataset field accessors
# Nama key mengikuti dataset aktual (huruf kecil):
#   index, question_type, question, follow_up, triage
# ---------------------------------------------------------------------------

def get_question(item: dict) -> str:
    return str(item.get("question", "")).strip()


def get_question_type(item: dict) -> str:
    """Kembalikan question_type — kategori penyakit (lebih luas dari ground_truth).

    Contoh: "Infeksi Saluran Pernapasan Atas (ISPA)"
    Lihat juga get_ground_truth() untuk diagnosis spesifik yang dikonfirmasi.
    """
    return str(item.get("question_type", "")).strip()


def get_follow_up_questions(item: dict) -> list[dict]:
    """Kembalikan list follow-up questions dari field 'follow_up'.
    
    Dinormalisasi ke format internal:
      [{"question": "...", "jawaban": <string atau list>}, ...]
    """
    raw = item.get("follow_up", [])

    # Sudah list — normalisasi key jika perlu
    if isinstance(raw, list):
        return _normalize_follow_up(raw)

    # Disimpan sebagai JSON string
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return _normalize_follow_up(parsed)
        except json.JSONDecodeError:
            pass

    return []


def _normalize_follow_up(items: list) -> list[dict]:
    """Normalisasi key follow-up dari format dataset ke format internal judge.
    """
    normalized = []
    for entry in items:
        if not isinstance(entry, dict):
            continue
        question = (
            entry.get("pertanyaan")
            or entry.get("question")
            or ""
        )
        jawaban = (
            entry.get("jawaban")
            or entry.get("answer")
            or ""
        )

        normalized.append({
            "question": str(question),
            "jawaban":  str(jawaban),
        })
    return normalized


def get_triage(item: dict) -> str:
    """Kembalikan label triage ground truth dalam huruf kecil.
    """
    triage = item.get("triage", "")
    return str(triage).strip().lower()


def get_ground_truth(item: dict) -> str:
    """Kembalikan ground_truth — diagnosis spesifik yang dikonfirmasi untuk kasus ini.

    Contoh: "Common Cold (Rhinitis Viral Akut)"
    Berbeda dari question_type yang merupakan kategori penyakit lebih luas.
    """
    return str(item.get("ground_truth", "")).strip()


def get_index(item: dict, fallback: int = 0) -> int:
    try:
        return int(item.get("index", fallback))
    except (ValueError, TypeError):
        return fallback


# ---------------------------------------------------------------------------
# Result saver
# ---------------------------------------------------------------------------

def save_results(results: dict, path: str = "healthqa_results.json") -> None:
    """Simpan hasil evaluasi ke file JSON dengan encoding UTF-8."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Hasil disimpan ke: {path}")


def save_results_dir(
    results: dict,
    model_key: str,
    judge_key: str = "",
    run_dir: str = "",
    results_dir: str = "results",
) -> str:
    """Simpan hasil evaluasi per model ke folder run di dalam results/.

    Nama file: <model_key>_<judge_key>.json
    Folder run: results/<YYYYMMDD_HHMMSS>/ (dibuat oleh caller, atau dibuat di sini).

    Parameters
    ----------
    results     : dict hasil evaluasi.
    model_key   : kunci model yang diuji (untuk nama file).
    judge_key   : kunci model judge (untuk nama file).
    run_dir     : path folder run yang sudah dibuat oleh caller.
                  Jika kosong, folder baru dibuat dengan timestamp sekarang.
    results_dir : folder induk (default: "results").
    """
    import datetime

    if run_dir:
        target_dir = Path(run_dir)
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        target_dir = Path(results_dir) / ts

    target_dir.mkdir(parents=True, exist_ok=True)

    safe_model = re.sub(r"[^\w\-]", "_", model_key)
    safe_judge = re.sub(r"[^\w\-]", "_", judge_key) if judge_key else "unknown_judge"
    filename = f"{safe_model}_{safe_judge}.json"

    path = str(target_dir / filename)
    save_results(results, path)
    return path