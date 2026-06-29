"""Evaluator HealthQA-ID.

Menjalankan satu model (atau semua model) pada dataset JSON, menilai setiap
respons menggunakan LLM-as-a-Judge, dan menyimpan hasil lengkap.

Penggunaan CLI:
  # Uji satu model
  python -m healthqa.evaluate --model gemini-3.5-flash --limit 5

  # Uji semua model yang terdaftar
  python -m healthqa.evaluate --all --limit 5

  # Lihat daftar model
  python -m healthqa.evaluate --list

  # Tentukan model judge secara eksplisit
  python -m healthqa.evaluate --model gemini-3.5-flash --judge gemma4-e4b-local
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

from .configs import (
    DATA_FILE,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_MODEL,
    METRIC_WEIGHTS,
    MODEL_SPECS,
    RESULTS_DIR,
    RESULTS_FILE,
    get_model_spec,
)
from .metrics import (
    compute_item_scores,
    compute_metrics,
    judge_response,
    print_metrics_summary,
)
from .models import ModelGateway, build_gateway, get_shared_limiter
from .presets import DIAGNOSIS_SYSTEM_PROMPT, DIAGNOSIS_USER_TEMPLATE
from .utilities import (
    get_follow_up_questions,
    get_ground_truth,
    get_index,
    get_question,
    get_question_type,
    get_triage,
    load_dataset,
    save_results,
    save_results_dir,
)


# ---------------------------------------------------------------------------
# Core: evaluasi satu model
# ---------------------------------------------------------------------------

def evaluate_model(
    model_key: str,
    judge_key: Optional[str] = None,
    data_file: str = DATA_FILE,
    limit: Optional[int] = None,
    save_to_dir: bool = True,
    verbose: bool = True,
    run_dir: Optional[str] = None,
) -> dict:
    """Evaluasi satu model terhadap dataset dan kembalikan dict hasil lengkap.

    Parameters
    ----------
    model_key : str
        Kunci model dari configs.MODEL_SPECS.
    judge_key : str | None
        Kunci model judge. Jika None, pakai DEFAULT_JUDGE_MODEL.
        Jika judge gagal diinisialisasi, fallback ke model yang diuji.
    data_file : str
        Path ke file JSON dataset.
    limit : int | None
        Jumlah item yang diuji (None = semua).
    save_to_dir : bool
        Jika True, simpan hasil ke folder run di RESULTS_DIR/.
        Selalu simpan salinan ke RESULTS_FILE (flat).
    verbose : bool
        Cetak progress per item.
    run_dir : str | None
        Path folder run yang sudah dibuat oleh evaluate_all_models().
        Jika None dan save_to_dir=True, folder baru dibuat dengan timestamp sekarang
        (perilaku saat evaluate_model dipanggil standalone).
    """
    from typing import Optional  # local import untuk kompatibilitas

    # --- Setup model yang diuji ---
    model_spec = get_model_spec(model_key)
    model_gateway = build_gateway(model_key)

    # --- Setup judge ---
    effective_judge_key = judge_key or DEFAULT_JUDGE_MODEL
    judge_gateway = _build_judge_gateway(effective_judge_key, fallback_gateway=model_gateway)
    judge_display = (
        get_model_spec(effective_judge_key).get("display_name", effective_judge_key)
        if effective_judge_key in MODEL_SPECS
        else f"{model_spec['display_name']} (fallback)"
    )

    # --- Dataset ---
    dataset = load_dataset(data_file)
    if not dataset:
        print("[ERROR] Dataset kosong atau tidak ditemukan.")
        return {}

    total = min(limit, len(dataset)) if limit else len(dataset)
    items = dataset[:total]

    if verbose:
        print("=" * 70)
        print(f"  Model diuji   : {model_spec['display_name']}")
        print(f"  Judge         : {judge_display}")
        print(f"  Total item    : {total}")
        print(f"  RPM limit     : {get_shared_limiter().rpm}")
        print(f"  Dataset       : {data_file}")
        print("=" * 70)

    results_detail = []

    for i, item in enumerate(items):
        idx          = get_index(item, fallback=i + 1)
        question     = get_question(item)
        qtype        = get_question_type(item)
        triage       = get_triage(item)
        ground_truth = get_ground_truth(item)
        followups    = get_follow_up_questions(item)
        n_ref_followups = len(followups)

        prompt = DIAGNOSIS_USER_TEMPLATE.format(question=question)

        if verbose:
            print(f"\n[{i+1}/{total}] #{idx} | triage={triage.upper()} | {question[:65]}…")

        # --- Panggil model yang diuji ---
        start = time.time()
        try:
            raw_answer = model_gateway.generate(
                prompt=prompt,
                system_prompt=DIAGNOSIS_SYSTEM_PROMPT,
            )
            latency = time.time() - start
            model_error = None
        except Exception as exc:
            raw_answer = f"ERROR: {exc}"
            latency = None
            model_error = str(exc)
            if verbose:
                print(f"  [!] Model error: {exc}")

        # --- Panggil judge (lewati jika model gagal) ---
        if model_error:
            judge_result = {
                "diagnosis_score":         0,
                "reasoning_score":         0,
                "matched_follow_up_count": 0,
                "asked_follow_up_count":   0,
                "predicted_triage":        "tidak_ditemukan",
                "diagnosis_reason":        "",
                "reasoning_reason":        "",
                "follow_up_reason":        "",
                "triage_reason":           "",
                "judge_error":             f"Model gagal: {model_error}",
            }
            item_scores = {
                "diagnosis_accuracy":  0.0,
                "diagnosis_reasoning": 0.0,
                "follow_up_coverage":  0.0,
                "triage_accuracy":     0.0,
                "composite_score":     0.0,
            }
        else:
            judge_result = judge_response(
                reference_item=item,
                model_answer=raw_answer,
                judge_gateway=judge_gateway,
            )
            item_scores = compute_item_scores(
                judge_result=judge_result,
                ground_truth_triage=triage,
                total_reference_follow_ups=n_ref_followups,
            )

        if verbose and not model_error:
            _print_item_result(item_scores, judge_result)

        results_detail.append({
            # --- Identifikasi ---
            "index":               idx,
            "question_type":       qtype,
            "ground_truth":        ground_truth,
            "question":            question,
            "ground_truth_triage": triage,
            "n_reference_followups": n_ref_followups,
            # --- Output model ---
            "raw_answer":          raw_answer,
            "latency":             latency,
            "model_error":         model_error,
            # --- Output judge ---
            "judge_diagnosis_score":  judge_result["diagnosis_score"],
            "judge_reasoning_score":  judge_result["reasoning_score"],
            "matched_follow_up_count": judge_result["matched_follow_up_count"],
            "asked_follow_up_count":  judge_result["asked_follow_up_count"],
            "predicted_triage":       judge_result["predicted_triage"],
            "judge_diagnosis_reason": judge_result["diagnosis_reason"],
            "judge_reasoning_reason": judge_result["reasoning_reason"],
            "judge_follow_up_reason": judge_result["follow_up_reason"],
            "judge_triage_reason":    judge_result["triage_reason"],
            "judge_error":            judge_result.get("judge_error"),
            # --- Skor ternormalisasi ---
            **item_scores,
        })

    # --- Agregasi metrik ---
    metrics = compute_metrics(results_detail)

    results = {
        "timestamp":         datetime.now().isoformat(),
        "model_key":         model_key,
        "model_display":     model_spec["display_name"],
        "judge_key":         effective_judge_key,
        "judge_display":     judge_display,
        "total_items":       total,
        "metric_weights":    METRIC_WEIGHTS,
        "metrics":           metrics,
        "results_detail":    results_detail,
    }

    # --- Simpan hasil ---
    if save_to_dir:
        path = save_results_dir(
            results,
            model_key=model_key,
            judge_key=effective_judge_key,
            run_dir=run_dir or "",
            results_dir=RESULTS_DIR,
        )
    save_results(results, RESULTS_FILE)

    if verbose:
        print_metrics_summary(metrics, model_display_name=model_spec["display_name"])

    return results


# ---------------------------------------------------------------------------
# Evaluasi semua model sekaligus
# ---------------------------------------------------------------------------

def evaluate_all_models(
    judge_key: Optional[str] = None,
    data_file: str = DATA_FILE,
    limit: Optional[int] = None,
    verbose: bool = True,
) -> dict[str, dict]:
    """Jalankan evaluate_model() untuk setiap model di MODEL_SPECS.

    Semua hasil disimpan ke satu folder run bersama:
      results/<YYYYMMDD_HHMMSS>/

    Setiap model menghasilkan file:
      <model_key>_<judge_key>.json

    Ringkasan perbandingan disimpan ke:
      results/<YYYYMMDD_HHMMSS>/comparison_summary.json

    Kembalikan dict {model_key: results_dict}.
    """
    from typing import Optional

    # Buat folder run sekali di awal agar semua model masuk folder yang sama
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = str(Path(RESULTS_DIR) / run_ts)
    Path(run_dir).mkdir(parents=True, exist_ok=True)

    all_results = {}
    model_keys = list(MODEL_SPECS.keys())

    print(f"\n{'='*70}")
    print(f"  EVALUASI SEMUA MODEL ({len(model_keys)} model)")
    print(f"  Folder run: {run_dir}")
    print(f"{'='*70}\n")

    for i, key in enumerate(model_keys):
        spec = MODEL_SPECS[key]
        print(f"\n[Model {i+1}/{len(model_keys)}] {spec['display_name']} ({key})")
        print("-" * 70)
        try:
            results = evaluate_model(
                model_key=key,
                judge_key=judge_key,
                data_file=data_file,
                limit=limit,
                save_to_dir=True,
                verbose=verbose,
                run_dir=run_dir,
            )
            all_results[key] = results
        except Exception as exc:
            print(f"[ERROR] Model '{key}' gagal seluruhnya: {exc}")
            all_results[key] = {"error": str(exc)}

    # Cetak perbandingan ringkas
    _print_comparison_table(all_results)

    # Simpan ringkasan perbandingan ke dalam folder run yang sama
    summary_path = str(Path(run_dir) / "comparison_summary.json")
    _save_comparison_summary(all_results, summary_path)

    return all_results


# ---------------------------------------------------------------------------
# Helper privat
# ---------------------------------------------------------------------------

def _build_judge_gateway(
    judge_key: str,
    fallback_gateway: ModelGateway,
) -> ModelGateway:
    """Buat gateway judge; jika gagal, kembalikan gateway model yang diuji."""
    if judge_key in MODEL_SPECS:
        try:
            return build_gateway(judge_key)
        except Exception as exc:
            print(f"[PERINGATAN] Judge '{judge_key}' gagal dikonfigurasi ({exc}). "
                  "Menggunakan model yang diuji sebagai judge.")
    else:
        print(f"[PERINGATAN] Judge key '{judge_key}' tidak ada di registry. "
              "Menggunakan model yang diuji sebagai judge.")
    return fallback_gateway


def _print_item_result(item_scores: dict, judge_result: dict) -> None:
    """Cetak ringkasan skor satu item."""
    print(
        f"  diag={item_scores['diagnosis_accuracy']:.2f} "
        f"reas={item_scores['diagnosis_reasoning']:.2f} "
        f"fu={item_scores['follow_up_coverage']:.2f} "
        f"triage={item_scores['triage_accuracy']:.0f} "
        f"→ composite={item_scores['composite_score']:.2f}"
        + (f"  [judge_error]" if judge_result.get("judge_error") else "")
    )


def _print_comparison_table(all_results: dict) -> None:
    """Cetak tabel perbandingan semua model."""
    print(f"\n{'='*70}")
    print("  PERBANDINGAN SEMUA MODEL")
    print(f"{'='*70}")
    print(
        f"  {'Model':<35} {'Diag':>6} {'Reas':>6} {'FU':>6} "
        f"{'Triage':>8} {'Komposit':>10} {'Pass':>6}"
    )
    print(f"  {'-'*35} {'-'*6} {'-'*6} {'-'*6} {'-'*8} {'-'*10} {'-'*6}")

    rows = []
    for key, res in all_results.items():
        if "error" in res:
            continue
        m = res.get("metrics", {})
        rows.append((
            res.get("model_display", key)[:35],
            m.get("avg_diagnosis_accuracy", 0),
            m.get("avg_diagnosis_reasoning", 0),
            m.get("avg_follow_up_coverage", 0),
            m.get("avg_triage_accuracy", 0),
            m.get("avg_composite_score", 0),
            m.get("pass_rate", 0),
        ))

    # Urutkan dari komposit tertinggi
    rows.sort(key=lambda r: r[5], reverse=True)

    for row in rows:
        print(
            f"  {row[0]:<35} {row[1]:>6.3f} {row[2]:>6.3f} {row[3]:>6.3f} "
            f"{row[4]:>8.3f} {row[5]:>10.3f} {row[6]:>5.1%}"
        )
    print(f"{'='*70}\n")


def _save_comparison_summary(all_results: dict, path: str) -> None:
    """Simpan ringkasan perbandingan ke JSON."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "models": {},
    }
    for key, res in all_results.items():
        if "error" in res:
            summary["models"][key] = {"error": res["error"]}
        else:
            m = res.get("metrics", {})
            summary["models"][key] = {
                "display_name":            res.get("model_display", key),
                "avg_diagnosis_accuracy":  m.get("avg_diagnosis_accuracy"),
                "avg_diagnosis_reasoning": m.get("avg_diagnosis_reasoning"),
                "avg_follow_up_coverage":  m.get("avg_follow_up_coverage"),
                "avg_triage_accuracy":     m.get("avg_triage_accuracy"),
                "avg_composite_score":     m.get("avg_composite_score"),
                "pass_rate":               m.get("pass_rate"),
                "judge_errors":            m.get("judge_errors"),
            }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Ringkasan perbandingan disimpan ke: {path}")


# ---------------------------------------------------------------------------
# Type hints backward-compat
# ---------------------------------------------------------------------------
from typing import Optional  # noqa: E402 — diletakkan setelah definisi fungsi


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="HealthQA-ID Evaluator — uji AI model untuk diagnosis medis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_build_epilog(),
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Kunci model yang diuji (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--judge", "-j",
        type=str,
        default=DEFAULT_JUDGE_MODEL,
        help=f"Kunci model judge (default: {DEFAULT_JUDGE_MODEL})",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Uji semua model yang terdaftar",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="Tampilkan daftar model yang tersedia dan keluar",
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        default=DATA_FILE,
        help=f"Path ke dataset JSON (default: {DATA_FILE})",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="Jumlah item yang diuji (default: semua)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Kurangi output per item",
    )
    args = parser.parse_args()

    if args.list:
        _print_model_list()
        return

    print("=" * 70)
    print("  HealthQA-ID Evaluator — Benchmark Diagnosis AI")
    print("=" * 70)

    if args.all:
        evaluate_all_models(
            judge_key=args.judge,
            data_file=args.file,
            limit=args.limit,
            verbose=not args.quiet,
        )
    else:
        evaluate_model(
            model_key=args.model,
            judge_key=args.judge,
            data_file=args.file,
            limit=args.limit,
            verbose=not args.quiet,
        )


def _print_model_list() -> None:
    print("\nModel yang tersedia:\n")
    print(f"  {'Kunci':<30} {'Nama Tampil':<35} {'Provider'}")
    print(f"  {'-'*30} {'-'*35} {'-'*15}")
    for key, spec in MODEL_SPECS.items():
        print(f"  {key:<30} {spec['display_name']:<35} {spec['provider']}")
    print()


def _build_epilog() -> str:
    keys = list(MODEL_SPECS.keys())
    lines = [
        "Model tersedia:",
        *[f"  {k}" for k in keys],
        "",
        "Contoh:",
        "  python -m healthqa.evaluate --model gemini-3.5-flash --limit 10",
        "  python -m healthqa.evaluate --all --limit 5",
        "  python -m healthqa.evaluate --list",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()