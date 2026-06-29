"""Modul metrik evaluasi untuk HealthQA-ID.

Semua penilaian dilakukan oleh LLM-as-a-Judge menggunakan JUDGE_SYSTEM_PROMPT
dan JUDGE_USER_TEMPLATE dari presets.py.

Fungsi utama:
- judge_response()     : kirim respons model ke judge, kembalikan skor terstruktur
- compute_item_scores(): hitung semua skor satu item dari output judge
- compute_metrics()    : agregasi semua item menjadi statistik final
"""
from __future__ import annotations

import json
import re
from typing import Optional

from .configs import METRIC_WEIGHTS, TRIAGE_LABELS
from .presets import JUDGE_SYSTEM_PROMPT, JUDGE_USER_TEMPLATE


# ---------------------------------------------------------------------------
# Judge: panggil LLM untuk menilai satu respons model
# ---------------------------------------------------------------------------

def judge_response(
    reference_item: dict,
    model_answer: str,
    judge_gateway,
) -> dict:
    """Kirim pasangan (referensi, jawaban model) ke judge dan kembalikan dict skor.

    Parameters
    ----------
    reference_item : dict
        Satu item dari dataset JSON (berisi question_type, follow_up, Triase).
    model_answer : str
        Respons JSON mentah dari model yang diuji.
    judge_gateway : ModelGateway
        Gateway yang sudah dikonfigurasi untuk model judge.

    Returns
    -------
    dict dengan key:
        diagnosis_score         (int, 0-4)
        reasoning_score         (int, 0-4)
        matched_follow_up_count (int)
        asked_follow_up_count   (int)
        predicted_triage        (str)
        diagnosis_reason        (str)
        reasoning_reason        (str)
        follow_up_reason        (str)
        triage_reason           (str)
        judge_error             (str | None)  — diisi jika parse/call gagal
    """
    # ---------------------------------------------------------------------------
    # Bangun referensi kasus yang dikirim ke judge
    # ---------------------------------------------------------------------------
    triase_raw = reference_item.get("triage", "")
    if isinstance(triase_raw, dict):
        triase_str = triase_raw.get("level", "")
    else:
        triase_str = str(triase_raw)

    # Ground truth — diagnosis spesifik yang dikonfirmasi
    ground_truth = str(reference_item.get("ground_truth", "")).strip()

    # Normalisasi follow_up untuk judge
    raw_followups = reference_item.get("follow_up", [])
    enriched_followups = []
    for entry in (raw_followups if isinstance(raw_followups, list) else []):
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

        enriched_followups.append({
            "question": str(question),
            "jawaban":  str(jawaban),
        })

    reference_json = json.dumps(
        {
            "question_type": reference_item.get("question_type", ""),
            "ground_truth":  ground_truth,
            "question":      reference_item.get("question", ""),
            "follow_up":     enriched_followups,
            "triage":        triase_str,
        },
        ensure_ascii=False,
        indent=2,
    )

    prompt = JUDGE_USER_TEMPLATE.format(
        reference_json=reference_json,
        model_answer=model_answer,
    )

    try:
        raw = judge_gateway.generate(prompt=prompt, system_prompt=JUDGE_SYSTEM_PROMPT)
    except Exception as exc:
        return _empty_judge_result(error=str(exc))

    return _parse_judge_output(raw)


def _parse_judge_output(raw: str) -> dict:
    """Parse JSON output judge; fallback heuristik jika gagal."""
    # Coba langsung parse JSON
    cleaned = raw.strip()

    # Hapus markdown code fences jika ada
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE).strip()

    # Ambil blok JSON pertama yang ditemukan
    json_match = re.search(r"\{[\s\S]*\}", cleaned)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            return {
                "diagnosis_score":         int(data.get("diagnosis_score", 0)),
                "reasoning_score":         int(data.get("reasoning_score", 0)),
                "matched_follow_up_count": int(data.get("matched_follow_up_count", 0)),
                "asked_follow_up_count":   int(data.get("asked_follow_up_count", 0)),
                "predicted_triage":        str(data.get("predicted_triage", "tidak_ditemukan")).lower(),
                "diagnosis_reason":        str(data.get("diagnosis_reason", "")),
                "reasoning_reason":        str(data.get("reasoning_reason", "")),
                "follow_up_reason":        str(data.get("follow_up_reason", "")),
                "triage_reason":           str(data.get("triage_reason", "")),
                "judge_error":             None,
            }
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: ekstrak angka dari teks
    diag  = _extract_score(raw, "diagnosis_score")
    reas  = _extract_score(raw, "reasoning_score")
    match = _extract_int(raw, "matched_follow_up_count")
    asked = _extract_int(raw, "asked_follow_up_count")
    triage = _extract_triage(raw)

    return {
        "diagnosis_score":         diag,
        "reasoning_score":         reas,
        "matched_follow_up_count": match,
        "asked_follow_up_count":   asked,
        "predicted_triage":        triage,
        "diagnosis_reason":        "",
        "reasoning_reason":        "",
        "follow_up_reason":        "",
        "triage_reason":           "",
        "judge_error":             "JSON parse fallback — angka diekstrak secara heuristik",
    }


def _empty_judge_result(error: str = "") -> dict:
    return {
        "diagnosis_score":         0,
        "reasoning_score":         0,
        "matched_follow_up_count": 0,
        "asked_follow_up_count":   0,
        "predicted_triage":        "tidak_ditemukan",
        "diagnosis_reason":        "",
        "reasoning_reason":        "",
        "follow_up_reason":        "",
        "triage_reason":           "",
        "judge_error":             error or "judge call gagal",
    }


def _extract_score(text: str, key: str) -> int:
    m = re.search(rf'"{key}"\s*:\s*([0-4])', text)
    if m:
        return int(m.group(1))
    return 0


def _extract_int(text: str, key: str) -> int:
    m = re.search(rf'"{key}"\s*:\s*(\d+)', text)
    if m:
        return int(m.group(1))
    return 0


def _extract_triage(text: str) -> str:
    for label in TRIAGE_LABELS:
        if re.search(rf'\b{label}\b', text, re.IGNORECASE):
            return label
    return "tidak_ditemukan"


# ---------------------------------------------------------------------------
# Konversi skor judge → metrik ternormalisasi (0.0–1.0)
# ---------------------------------------------------------------------------

def compute_item_scores(
    judge_result: dict,
    ground_truth_triage: str,
    total_reference_follow_ups: int,
) -> dict:
    """Hitung skor ternormalisasi satu item dari output judge.

    Returns
    -------
    dict dengan key:
        diagnosis_accuracy   (float 0-1)  : diagnosis_score / 4
        diagnosis_reasoning  (float 0-1)  : reasoning_score / 4
        follow_up_coverage   (float 0-1)  : matched / total_ref (atau 0.5 jika ref kosong)
        triage_accuracy      (float 0/1)  : exact-match label triage
        composite_score      (float 0-1)  : rata-rata berbobot keempat metrik
    """
    diag_acc  = round(judge_result["diagnosis_score"] / 4, 3)
    diag_reas = round(judge_result["reasoning_score"] / 4, 3)

    if total_reference_follow_ups > 0:
        matched = min(
            judge_result["matched_follow_up_count"],
            total_reference_follow_ups,
        )
        fu_cov = round(matched / total_reference_follow_ups, 3)
    else:
        # Tidak ada follow-up referensi → skor netral
        fu_cov = 0.5

    pred_triage = judge_result["predicted_triage"].lower().strip()
    gt_triage   = ground_truth_triage.lower().strip()
    triage_acc  = 1.0 if pred_triage == gt_triage else 0.0

    w = METRIC_WEIGHTS
    composite = round(
        diag_acc  * w["diagnosis_accuracy"]
        + diag_reas * w["diagnosis_reasoning"]
        + fu_cov    * w["follow_up_coverage"]
        + triage_acc * w["triage_accuracy"],
        3,
    )

    return {
        "diagnosis_accuracy":  diag_acc,
        "diagnosis_reasoning": diag_reas,
        "follow_up_coverage":  fu_cov,
        "triage_accuracy":     triage_acc,
        "composite_score":     composite,
    }


# ---------------------------------------------------------------------------
# Agregasi seluruh item
# ---------------------------------------------------------------------------

def compute_metrics(results_detail: list[dict]) -> dict:
    """Hitung semua metrik agregat dari list hasil evaluasi per item.

    Setiap item diharapkan memiliki:
        diagnosis_accuracy, diagnosis_reasoning,
        follow_up_coverage, triage_accuracy, composite_score,
        latency (float detik, boleh None)

    Returns
    -------
    dict metrik agregat, termasuk distribusi per dimensi dan per label triage.
    """
    n = len(results_detail)
    if n == 0:
        return _empty_metrics()

    def _avg(key: str) -> float:
        vals = [r.get(key) or 0.0 for r in results_detail]
        return round(sum(vals) / len(vals), 3)

    avg_diag_acc  = _avg("diagnosis_accuracy")
    avg_diag_reas = _avg("diagnosis_reasoning")
    avg_fu_cov    = _avg("follow_up_coverage")
    avg_triage    = _avg("triage_accuracy")
    avg_composite = _avg("composite_score")

    # Pass rate: composite >= 0.5
    pass_rate = round(
        sum(1 for r in results_detail if (r.get("composite_score") or 0) >= 0.5) / n, 3
    )

    # Latensi model yang diuji (bukan judge)
    latencies = [r["latency"] for r in results_detail if r.get("latency") is not None]
    avg_latency_ms = round(sum(latencies) / len(latencies) * 1000, 1) if latencies else None

    # Distribusi skor diagnosis (0-4) → bucket
    diag_dist = _score_distribution(results_detail, "judge_diagnosis_score", max_val=4)
    reas_dist = _score_distribution(results_detail, "judge_reasoning_score", max_val=4)

    # Akurasi triage per label
    triage_breakdown = _triage_breakdown(results_detail)

    # Hitung berapa item judge gagal
    judge_errors = sum(1 for r in results_detail if r.get("judge_error"))

    return {
        # --- Rata-rata per dimensi ---
        "avg_diagnosis_accuracy":  avg_diag_acc,
        "avg_diagnosis_reasoning": avg_diag_reas,
        "avg_follow_up_coverage":  avg_fu_cov,
        "avg_triage_accuracy":     avg_triage,
        "avg_composite_score":     avg_composite,
        # --- Ringkasan ---
        "pass_rate":               pass_rate,
        "avg_latency_ms":          avg_latency_ms,
        "total_items":             n,
        "judge_errors":            judge_errors,
        # --- Distribusi detail ---
        "diagnosis_score_distribution": diag_dist,
        "reasoning_score_distribution": reas_dist,
        "triage_accuracy_by_label":     triage_breakdown,
    }


def _empty_metrics() -> dict:
    return {
        "avg_diagnosis_accuracy":       0.0,
        "avg_diagnosis_reasoning":      0.0,
        "avg_follow_up_coverage":       0.0,
        "avg_triage_accuracy":          0.0,
        "avg_composite_score":          0.0,
        "pass_rate":                    0.0,
        "avg_latency_ms":               None,
        "total_items":                  0,
        "judge_errors":                 0,
        "diagnosis_score_distribution": {},
        "reasoning_score_distribution": {},
        "triage_accuracy_by_label":     {},
    }


def _score_distribution(results: list[dict], key: str, max_val: int) -> dict:
    """Hitung distribusi nilai integer 0..max_val untuk field `key`."""
    dist: dict[str, int] = {str(i): 0 for i in range(max_val + 1)}
    for r in results:
        val = r.get(key)
        if val is not None:
            dist[str(int(val))] = dist.get(str(int(val)), 0) + 1
    return dist


def _triage_breakdown(results: list[dict]) -> dict:
    """Hitung akurasi triage per label (rendah/menengah/darurat)."""
    counts: dict[str, dict[str, int]] = {
        label: {"correct": 0, "total": 0} for label in TRIAGE_LABELS
    }
    counts["lainnya"] = {"correct": 0, "total": 0}

    for r in results:
        gt = (r.get("ground_truth_triage") or "").lower().strip()
        pred = (r.get("predicted_triage") or "").lower().strip()
        bucket = gt if gt in TRIAGE_LABELS else "lainnya"
        counts[bucket]["total"] += 1
        if pred == gt:
            counts[bucket]["correct"] += 1

    result = {}
    for label, c in counts.items():
        if c["total"] > 0:
            result[label] = {
                "total":    c["total"],
                "correct":  c["correct"],
                "accuracy": round(c["correct"] / c["total"], 3),
            }
    return result


# ---------------------------------------------------------------------------
# Cetak ringkasan ke stdout
# ---------------------------------------------------------------------------

def print_metrics_summary(metrics: dict, model_display_name: str = "") -> None:
    """Cetak ringkasan metrik ke stdout."""
    header = f"RINGKASAN METRIK — {model_display_name}" if model_display_name else "RINGKASAN METRIK"
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  {header}")
    print(sep)
    print(f"  Total item diuji     : {metrics.get('total_items', 0)}")
    print(f"  Judge errors         : {metrics.get('judge_errors', 0)}")
    print()
    print(f"  Akurasi Diagnosis    : {metrics.get('avg_diagnosis_accuracy', 0):.3f}  (bobot 40%)")
    print(f"  Penalaran Diagnosis  : {metrics.get('avg_diagnosis_reasoning', 0):.3f}  (bobot 25%)")
    print(f"  Cakupan Follow-up    : {metrics.get('avg_follow_up_coverage', 0):.3f}  (bobot 20%)")
    print(f"  Akurasi Triase       : {metrics.get('avg_triage_accuracy', 0):.3f}  (bobot 15%)")
    print(f"  ─────────────────────────────────────────")
    print(f"  Skor Komposit        : {metrics.get('avg_composite_score', 0):.3f}")
    print(f"  Pass Rate (≥0.5)     : {metrics.get('pass_rate', 0):.1%}")
    if metrics.get("avg_latency_ms") is not None:
        print(f"  Latensi rata-rata    : {metrics['avg_latency_ms']:.0f} ms")

    # Distribusi skor diagnosis
    dd = metrics.get("diagnosis_score_distribution", {})
    if dd:
        print(f"\n  Distribusi Skor Diagnosis (0-4):")
        for score in ["4", "3", "2", "1", "0"]:
            label = {
                "4": "Tepat/Ekuivalen",
                "3": "Lebih umum",
                "2": "Diagnosa banding",
                "1": "Lemah/Kabur",
                "0": "Salah/Berbahaya",
            }.get(score, score)
            print(f"    {score} ({label:20s}): {dd.get(score, 0)}")

    # Akurasi triage per label
    tb = metrics.get("triage_accuracy_by_label", {})
    if tb:
        print(f"\n  Akurasi Triase per Label:")
        for label in [*TRIAGE_LABELS, "lainnya"]:
            if label in tb:
                d = tb[label]
                print(
                    f"    {label:12s}: {d['correct']}/{d['total']} "
                    f"({d['accuracy']:.0%})"
                )
    print(sep + "\n")