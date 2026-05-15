"""
AFDB Pre-AFib + AFib + Combined RR Analysis (ML-focused)
=========================================================

Uses AFDB SR->AF transition definitions from `afdb/sr_to_af_segments.csv`.
RR intervals are computed from beat annotations with preference order:
  1) manual corrected .qrsc (if available)
  2) automated .qrs
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import wfdb

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASET_PATH = Path("/home/intellisense01/EML-Labs/datasets/afdb")
SEGMENTS_CSV = Path(__file__).resolve().parent / "sr_to_af_segments.csv"
BASE_RESULTS_DIR = Path(__file__).resolve().parent.parent / "Results"
OUT_DIR = BASE_RESULTS_DIR / "afdb_pre_afib_afib_combined_analysis"
PLOTS_DIR = OUT_DIR / "plots"

HORIZON_MINUTES = [10, 20, 30, 60, 120, 180]
PHYSIOLOGICAL_RR_MIN_MS = 250
PHYSIOLOGICAL_RR_MAX_MS = 2500
IQR_OUTLIER_FACTOR = 1.5
CALIBRATION_SKIP_MS = 60_000
SEGMENT_TYPES = ("pre_afib", "afib", "both")

sns.set_theme(style="whitegrid", font_scale=1.0)


@dataclass
class EpisodeMeta:
    record_id: str
    episode_index: int
    fs_hz: float
    sr_start_sample: int
    af_start_sample: int
    af_end_sample: int
    sr_duration_sec: float
    af_duration_sec: float


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def rr_features(rr_seg: np.ndarray) -> Dict[str, float]:
    if len(rr_seg) == 0:
        return {
            "rr_count": 0,
            "duration_sec": 0.0,
            "mean_rr_ms": 0.0,
            "median_rr_ms": 0.0,
            "std_rr_ms": 0.0,
            "iqr_rr_ms": 0.0,
            "min_rr_ms": 0.0,
            "max_rr_ms": 0.0,
            "mean_hr_bpm": 0.0,
            "sdnn_ms": 0.0,
            "rmssd_ms": 0.0,
            "pnn50": 0.0,
            "outlier_count_low_phys": 0,
            "outlier_count_high_phys": 0,
            "outlier_count_total_phys": 0,
            "outlier_pct_phys": 100.0,
            "outlier_count_iqr": 0,
            "outlier_pct_iqr": 0.0,
            "phys_out_of_range_pct": 100.0,
            "duplicate_pct": 0.0,
            "rr_diff_std_ms": 0.0,
            "trend_slope_ms_per_beat": 0.0,
        }

    rr = rr_seg.astype(float)
    rr_diff = np.diff(rr)
    out_of_range = np.logical_or(rr < PHYSIOLOGICAL_RR_MIN_MS, rr > PHYSIOLOGICAL_RR_MAX_MS)
    outlier_low = rr < PHYSIOLOGICAL_RR_MIN_MS
    outlier_high = rr > PHYSIOLOGICAL_RR_MAX_MS

    q1 = np.percentile(rr, 25)
    q3 = np.percentile(rr, 75)
    iqr = q3 - q1
    low_iqr_bound = q1 - IQR_OUTLIER_FACTOR * iqr
    high_iqr_bound = q3 + IQR_OUTLIER_FACTOR * iqr
    outlier_iqr = np.logical_or(rr < low_iqr_bound, rr > high_iqr_bound)

    duplicate_pct = safe_div(np.sum(np.diff(rr) == 0), max(len(rr) - 1, 1)) * 100.0
    x = np.arange(len(rr), dtype=float)
    slope = np.polyfit(x, rr, 1)[0] if len(rr) >= 2 else 0.0

    return {
        "rr_count": int(len(rr)),
        "duration_sec": float(np.sum(rr) / 1000.0),
        "mean_rr_ms": float(np.mean(rr)),
        "median_rr_ms": float(np.median(rr)),
        "std_rr_ms": float(np.std(rr)),
        "iqr_rr_ms": float(np.percentile(rr, 75) - np.percentile(rr, 25)),
        "min_rr_ms": float(np.min(rr)),
        "max_rr_ms": float(np.max(rr)),
        "mean_hr_bpm": float(60000.0 / np.mean(rr)),
        "sdnn_ms": float(np.std(rr)),
        "rmssd_ms": float(np.sqrt(np.mean(rr_diff**2))) if len(rr_diff) > 0 else 0.0,
        "pnn50": float(np.mean(np.abs(rr_diff) > 50.0) * 100.0) if len(rr_diff) > 0 else 0.0,
        "outlier_count_low_phys": int(np.sum(outlier_low)),
        "outlier_count_high_phys": int(np.sum(outlier_high)),
        "outlier_count_total_phys": int(np.sum(out_of_range)),
        "outlier_pct_phys": float(np.mean(out_of_range) * 100.0),
        "outlier_count_iqr": int(np.sum(outlier_iqr)),
        "outlier_pct_iqr": float(np.mean(outlier_iqr) * 100.0),
        "phys_out_of_range_pct": float(np.mean(out_of_range) * 100.0),
        "duplicate_pct": float(duplicate_pct),
        "rr_diff_std_ms": float(np.std(rr_diff)) if len(rr_diff) > 0 else 0.0,
        "trend_slope_ms_per_beat": float(slope),
    }


def quality_score(features: Dict[str, float], target_duration_sec: int) -> tuple[float, bool]:
    if target_duration_sec <= 0:
        return 0.0, False
    coverage_ratio = min(1.0, safe_div(features["duration_sec"], target_duration_sec))
    coverage_pts = 40.0 * coverage_ratio
    phys_valid_ratio = max(0.0, 1.0 - features["phys_out_of_range_pct"] / 100.0)
    phys_pts = 20.0 * phys_valid_ratio
    count_target = max(200.0, target_duration_sec / 0.9)
    count_ratio = min(1.0, safe_div(features["rr_count"], count_target))
    count_pts = 15.0 * count_ratio
    duplicate_penalty = min(1.0, features["duplicate_pct"] / 20.0)
    artifact_pts = 15.0 * (1.0 - duplicate_penalty)
    std = features["std_rr_ms"]
    variability_pts = 2.0 if std < 10 else (4.0 if std > 400 else 10.0)
    score = coverage_pts + phys_pts + count_pts + artifact_pts + variability_pts
    ml_ready = (
        coverage_ratio >= 0.95
        and features["rr_count"] >= 200
        and features["phys_out_of_range_pct"] <= 5.0
        and features["duplicate_pct"] <= 10.0
        and score >= 70.0
    )
    return float(score), bool(ml_ready)


def calibration_cut_index(rr: np.ndarray, skip_ms: int = CALIBRATION_SKIP_MS) -> int:
    if len(rr) == 0:
        return 0
    csum_ms = np.cumsum(rr.astype(np.int64))
    cut = int(np.searchsorted(csum_ms, skip_ms, side="left") + 1)
    return min(cut, len(rr))


def extract_pre_afib_segment(rr: np.ndarray, af_start_idx: int, horizon_sec: int, calibration_start_idx: int) -> np.ndarray:
    if af_start_idx <= 0 or len(rr) == 0:
        return np.array([], dtype=rr.dtype)
    valid_start = min(calibration_start_idx, len(rr))
    prior = rr[valid_start:af_start_idx]
    if len(prior) == 0:
        return np.array([], dtype=rr.dtype)
    rev = prior[::-1]
    rev_cumsum_sec = np.cumsum(rev) / 1000.0
    cut = np.searchsorted(rev_cumsum_sec, horizon_sec, side="left")
    return rev[::-1] if cut >= len(rev) else rev[: cut + 1][::-1]


def extract_afib_segment_horizon(
    rr: np.ndarray, af_start_idx: int, af_end_idx: int, horizon_sec: int, calibration_start_idx: int
) -> np.ndarray:
    if af_start_idx < 0 or af_end_idx <= af_start_idx or len(rr) == 0:
        return np.array([], dtype=rr.dtype)
    lo = max(calibration_start_idx, af_start_idx)
    hi = min(len(rr), af_end_idx)
    seg = rr[lo:hi]
    if len(seg) == 0:
        return np.array([], dtype=rr.dtype)
    cum_sec = np.cumsum(seg) / 1000.0
    cut = np.searchsorted(cum_sec, horizon_sec, side="left")
    return seg if cut >= len(seg) else seg[: cut + 1]


def add_traceability_pre(feat: Dict, af_start: int, calibration_start_idx: int) -> Dict[str, int | str]:
    n = feat["rr_count"]
    start = max(calibration_start_idx, af_start - n)
    return {
        "segment_start_global_idx": int(start),
        "segment_end_global_idx": int(af_start),
        "segment_global_index_range": f"{start}:{af_start}",
    }


def add_traceability_af(feat: Dict, af_start: int) -> Dict[str, int | str]:
    n = feat["rr_count"]
    end = af_start + n
    return {
        "segment_start_global_idx": int(af_start),
        "segment_end_global_idx": int(end),
        "segment_global_index_range": f"{af_start}:{end}",
    }


def add_traceability_both(pre_feat: Dict, af_feat: Dict, pre_start: int, pre_end: int, af_start: int) -> Dict[str, int | str]:
    n_pre = pre_feat["rr_count"]
    n_af = af_feat["rr_count"]
    if n_pre == 0 and n_af == 0:
        return {
            "segment_start_global_idx": int(pre_start),
            "segment_end_global_idx": int(pre_start),
            "segment_global_index_range": f"{pre_start}:{pre_start}",
        }
    combined_start = pre_start if n_pre > 0 else af_start
    combined_end = (af_start + n_af) if n_af > 0 else pre_end
    return {
        "segment_start_global_idx": int(combined_start),
        "segment_end_global_idx": int(combined_end),
        "segment_global_index_range": f"{combined_start}:{combined_end}",
    }


def build_episode_list(segments_csv: Path) -> List[EpisodeMeta]:
    seg = pd.read_csv(segments_csv, dtype={"record_id": str})
    seg["record_id"] = seg["record_id"].str.zfill(5)
    seg = seg.sort_values(["record_id", "af_start_sample"]).reset_index(drop=True)
    seg["episode_index"] = seg.groupby("record_id").cumcount()
    episodes: List[EpisodeMeta] = []
    for _, r in seg.iterrows():
        episodes.append(
            EpisodeMeta(
                record_id=str(r["record_id"]).zfill(5),
                episode_index=int(r["episode_index"]),
                fs_hz=float(r["fs_hz"]),
                sr_start_sample=int(r["sr_start_sample"]),
                af_start_sample=int(r["af_start_sample"]),
                af_end_sample=int(r["af_end_sample"]),
                sr_duration_sec=float(r["sr_duration_sec"]),
                af_duration_sec=float(r["af_duration_sec"]),
            )
        )
    return episodes


def load_peaks(dataset_dir: Path, rid: str) -> tuple[np.ndarray, float, str]:
    header = wfdb.rdheader(str(dataset_dir / rid))
    fs = float(header.fs)
    ann_ext = "qrsc" if (dataset_dir / f"{rid}.qrsc").exists() else "qrs"
    ann = wfdb.rdann(str(dataset_dir / rid), ann_ext)
    peaks = np.asarray(ann.sample, dtype=np.int64)
    peaks = peaks[peaks >= 0]
    peaks = np.sort(peaks)
    return peaks, fs, ann_ext


def rr_from_peaks(peaks: np.ndarray, fs_hz: float) -> np.ndarray:
    if len(peaks) < 2:
        return np.array([], dtype=np.float64)
    return np.diff(peaks).astype(np.float64) * (1000.0 / fs_hz)


def sample_to_rr_index(peaks: np.ndarray, sample_idx: int) -> int:
    if len(peaks) < 2:
        return 0
    return int(np.searchsorted(peaks[:-1], sample_idx, side="left"))


def generate_plots(df: pd.DataFrame) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    agg = df.groupby(["segment_type", "horizon_min"])[["is_sufficient_duration", "ml_ready"]].mean().reset_index()

    fig, ax = plt.subplots(figsize=(12, 5))
    agg.pivot(index="horizon_min", columns="segment_type", values="ml_ready").plot(kind="bar", ax=ax, rot=0)
    ax.set_xlabel("Horizon (minutes)")
    ax.set_ylabel("Fraction ML-ready")
    ax.set_title("AFDB ML-ready rate by horizon and segment")
    ax.legend(title="segment_type")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "01_ml_ready_by_horizon_segment.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=df, x="horizon_min", y="quality_score", hue="segment_type", ax=ax)
    ax.set_title("AFDB quality score by horizon and segment")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "02_quality_score_by_horizon_segment.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=df, x="horizon_min", y="mean_rr_ms", hue="segment_type", ax=ax, showfliers=False)
    ax.set_title("AFDB mean RR by horizon and segment")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "03_mean_rr_by_horizon_segment.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=df, x="horizon_min", y="outlier_pct_phys", hue="segment_type", ax=ax, showfliers=False)
    ax.set_title("AFDB outlier rate by horizon and segment")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "04_outlier_pct_by_horizon_segment.png", dpi=150)
    plt.close(fig)

    heat = df.groupby(["segment_type", "horizon_min"])["ml_ready"].mean().unstack(0) * 100.0
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(heat.T, annot=True, fmt=".1f", cmap="YlGnBu", vmin=0, vmax=100, ax=ax, cbar_kws={"label": "ML-ready %"})
    ax.set_title("AFDB ML-ready % heatmap")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "05_ml_ready_heatmap.png", dpi=150)
    plt.close(fig)

    ne = df[df["rr_count"] > 0].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    for seg in SEGMENT_TYPES:
        vals = np.sort(ne.loc[ne["segment_type"] == seg, "outlier_pct_phys"].values)
        if len(vals) == 0:
            continue
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, y, label=seg)
    ax.set_title("AFDB ECDF of physical outlier %")
    ax.legend(title="segment_type")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "06_outlier_pct_ecdf.png", dpi=150)
    plt.close(fig)

    band_df = ne.copy()
    bins = [-1e-9, 0, 0.5, 1.0, 5.0, 100.0]
    labels = ["0%", "0-0.5%", "0.5-1%", "1-5%", ">5%"]
    band_df["outlier_band"] = pd.cut(band_df["outlier_pct_phys"], bins=bins, labels=labels)
    band = band_df.groupby(["segment_type", "outlier_band"], observed=False).size().reset_index(name="n")
    band["pct"] = band.groupby("segment_type")["n"].transform(lambda x: x / x.sum() * 100.0)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=band, x="outlier_band", y="pct", hue="segment_type", ax=ax)
    ax.set_title("AFDB outlier band distribution")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "07_outlier_band_distribution.png", dpi=150)
    plt.close(fig)


def build_cleanliness_verification(df: pd.DataFrame) -> pd.DataFrame:
    ne = df[df["rr_count"] > 0].copy()
    rows = []
    for (seg, h), g in ne.groupby(["segment_type", "horizon_min"]):
        total_out = float(g["outlier_count_total_phys"].sum())
        total_rr = float(g["rr_count"].sum())
        rows.append(
            {
                "segment_type": seg,
                "horizon_min": int(h),
                "n_nonempty": int(len(g)),
                "global_outlier_pct": (total_out / total_rr * 100.0) if total_rr > 0 else np.nan,
                "mean_outlier_pct_nonempty": float(g["outlier_pct_phys"].mean()),
                "median_outlier_pct_nonempty": float(g["outlier_pct_phys"].median()),
                "p95_outlier_pct_nonempty": float(np.percentile(g["outlier_pct_phys"], 95)),
                "p99_outlier_pct_nonempty": float(np.percentile(g["outlier_pct_phys"], 99)),
                "fail_outlier_gt_2_pct": float(np.mean(g["outlier_pct_phys"] > 2.0) * 100.0),
                "fail_outlier_gt_5_pct": float(np.mean(g["outlier_pct_phys"] > 5.0) * 100.0),
                "fail_duplicate_gt_10_pct": float(np.mean(g["duplicate_pct"] > 10.0) * 100.0),
                "fail_rr_count_lt_200_pct": float(np.mean(g["rr_count"] < 200) * 100.0),
                "fail_min_rr_lt_250_pct": float(np.mean(g["min_rr_ms"] < PHYSIOLOGICAL_RR_MIN_MS) * 100.0),
                "fail_max_rr_gt_2500_pct": float(np.mean(g["max_rr_ms"] > PHYSIOLOGICAL_RR_MAX_MS) * 100.0),
                "fail_any_rule_pct": float(np.mean((g["outlier_pct_phys"] > 5.0) | (g["duplicate_pct"] > 10.0) | (g["rr_count"] < 200)) * 100.0),
            }
        )
    return pd.DataFrame(rows).sort_values(["segment_type", "horizon_min"]).reset_index(drop=True)


def write_report(df: pd.DataFrame) -> None:
    lines: List[str] = []
    lines.append("# AFDB Pre-AFib + AFib + Combined Episode Analysis")
    lines.append("")
    lines.append(f"> Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append("- Episodes loaded from `afdb/sr_to_af_segments.csv`.")
    lines.append("- RR derived from beat annotations with preference `.qrsc` then `.qrs`.")
    lines.append("- pre_afib: last H minutes before AF onset; afib: first H inside AF segment.")
    lines.append("- both: concatenation pre_afib + afib, target duration 2H.")
    lines.append("")

    summary = (
        df.groupby(["segment_type", "horizon_min"])
        .agg(
            n=("episode_index", "count"),
            sufficient_pct=("is_sufficient_duration", lambda x: float(np.mean(x) * 100)),
            ml_ready_pct=("ml_ready", lambda x: float(np.mean(x) * 100)),
            mean_quality=("quality_score", "mean"),
            mean_rr_count=("rr_count", "mean"),
            mean_mean_rr=("mean_rr_ms", "mean"),
            mean_std_rr=("std_rr_ms", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(OUT_DIR / "horizon_segment_summary.csv", index=False)
    with open(OUT_DIR / "horizon_segment_summary.json", "w") as f:
        json.dump(summary.to_dict(orient="records"), f, indent=2)

    lines.append("## Summary")
    lines.append("")
    lines.append("| segment_type | horizon_min | n_rows | sufficient_% | ml_ready_% | mean_quality | mean_rr_count | mean_mean_rr_ms | mean_std_rr_ms |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in summary.sort_values(["segment_type", "horizon_min"]).iterrows():
        lines.append(
            f"| {r['segment_type']} | {int(r['horizon_min'])} | {int(r['n'])} | {r['sufficient_pct']:.2f} | "
            f"{r['ml_ready_pct']:.2f} | {r['mean_quality']:.2f} | {r['mean_rr_count']:.1f} | "
            f"{r['mean_mean_rr']:.1f} | {r['mean_std_rr']:.1f} |"
        )

    ver = build_cleanliness_verification(df)
    ver.to_csv(OUT_DIR / "cleanliness_verification_summary.csv", index=False)
    with open(OUT_DIR / "cleanliness_verification_summary.json", "w") as f:
        json.dump(ver.to_dict(orient="records"), f, indent=2)

    lines.append("")
    lines.append("## Visualizations")
    lines.append("")
    for name in [
        "01_ml_ready_by_horizon_segment.png",
        "02_quality_score_by_horizon_segment.png",
        "03_mean_rr_by_horizon_segment.png",
        "04_outlier_pct_by_horizon_segment.png",
        "05_ml_ready_heatmap.png",
        "06_outlier_pct_ecdf.png",
        "07_outlier_band_distribution.png",
    ]:
        lines.append(f"![{name}](plots/{name})")

    (OUT_DIR / "AFDB_PreAFib_AFib_Combined_Report.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="AFDB pre/af/both RR analysis.")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_PATH)
    parser.add_argument("--segments-csv", type=Path, default=SEGMENTS_CSV)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    print("=" * 72)
    print("AFDB Pre-AFib + AFib + Combined analysis")
    print("=" * 72)

    episodes = build_episode_list(args.segments_csv)
    print(f"Total SR->AF episodes: {len(episodes)}")

    rr_cache: Dict[str, tuple[np.ndarray, np.ndarray, float, str]] = {}
    rows: List[Dict] = []

    for idx, ep in enumerate(episodes):
        rid = ep.record_id
        if rid not in rr_cache:
            peaks, fs, ann_ext = load_peaks(args.dataset_dir, rid)
            rr_all = rr_from_peaks(peaks, fs)
            rr_cache[rid] = (peaks, rr_all, fs, ann_ext)

        peaks, rr_all, fs, ann_ext = rr_cache[rid]
        if len(rr_all) == 0:
            continue

        calibration_start_idx = calibration_cut_index(rr_all, CALIBRATION_SKIP_MS)
        af_start = sample_to_rr_index(peaks, ep.af_start_sample)
        af_end = sample_to_rr_index(peaks, ep.af_end_sample)
        if af_end <= af_start:
            af_end = min(len(rr_all), af_start + 1)

        for horizon_min in HORIZON_MINUTES:
            target_sec = horizon_min * 60
            target_both_sec = 2 * target_sec

            seg_pre = extract_pre_afib_segment(rr_all, af_start, target_sec, calibration_start_idx)
            seg_af = extract_afib_segment_horizon(rr_all, af_start, af_end, target_sec, calibration_start_idx)
            feat_pre = rr_features(seg_pre)
            feat_af = rr_features(seg_af)

            if len(seg_pre) > 0 and len(seg_af) > 0:
                seg_both = np.concatenate([seg_pre, seg_af])
            elif len(seg_pre) > 0:
                seg_both = seg_pre.copy()
            elif len(seg_af) > 0:
                seg_both = seg_af.copy()
            else:
                seg_both = np.array([], dtype=rr_all.dtype)
            feat_both = rr_features(seg_both)

            pre_trace = add_traceability_pre(feat_pre, af_start, calibration_start_idx)
            af_trace = add_traceability_af(feat_af, af_start)
            both_trace = add_traceability_both(feat_pre, feat_af, pre_trace["segment_start_global_idx"], af_start, af_start)

            sufficient_pre = feat_pre["duration_sec"] >= target_sec * 0.95
            sufficient_af = feat_af["duration_sec"] >= target_sec * 0.95
            sufficient_both = feat_both["duration_sec"] >= target_both_sec * 0.95

            score_pre, ready_pre = quality_score(feat_pre, target_sec)
            score_af, ready_af = quality_score(feat_af, target_sec)
            score_both, ready_both = quality_score(feat_both, target_both_sec)

            base = {
                "record_id": rid,
                "episode_index": ep.episode_index,
                "annotation_source": ann_ext,
                "fs_hz": fs,
                "af_start_sample": ep.af_start_sample,
                "af_end_sample_exclusive": ep.af_end_sample,
                "afib_start_global_idx": af_start,
                "afib_end_global_idx_exclusive": af_end,
                "horizon_min": horizon_min,
                "target_duration_sec_pre_or_af": target_sec,
                "target_duration_sec_both": target_both_sec,
                "af_labeled_duration_sec": float(np.sum(rr_all[af_start:af_end]) / 1000.0),
                "af_duration_sec_meta": ep.af_duration_sec,
                "nsr_before_sec": ep.sr_duration_sec,
            }

            for seg_type, feat, suff, sc, ml, trace in [
                ("pre_afib", feat_pre, sufficient_pre, score_pre, ready_pre, pre_trace),
                ("afib", feat_af, sufficient_af, score_af, ready_af, af_trace),
                ("both", feat_both, sufficient_both, score_both, ready_both, both_trace),
            ]:
                rows.append(
                    {
                        **base,
                        "segment_type": seg_type,
                        "is_sufficient_duration": bool(suff),
                        "quality_score": sc,
                        "ml_ready": bool(ml),
                        "outlier_pct_phys_nonempty": feat["outlier_pct_phys"] if feat["rr_count"] > 0 else np.nan,
                        "outlier_pct_iqr_nonempty": feat["outlier_pct_iqr"] if feat["rr_count"] > 0 else np.nan,
                        **trace,
                        **feat,
                    }
                )

        if (idx + 1) % 50 == 0 or idx + 1 == len(episodes):
            print(f"Processed episodes: {idx + 1}/{len(episodes)}")

    df = pd.DataFrame(rows)
    df.sort_values(["record_id", "episode_index", "horizon_min", "segment_type"], inplace=True)
    df.to_csv(OUT_DIR / "episode_pre_afib_afib_combined_features.csv", index=False)

    trace_cols = [
        "record_id",
        "episode_index",
        "annotation_source",
        "segment_type",
        "horizon_min",
        "af_start_sample",
        "af_end_sample_exclusive",
        "afib_start_global_idx",
        "afib_end_global_idx_exclusive",
        "segment_start_global_idx",
        "segment_end_global_idx",
        "segment_global_index_range",
        "target_duration_sec_pre_or_af",
        "target_duration_sec_both",
        "duration_sec",
        "is_sufficient_duration",
        "rr_count",
        "outlier_count_low_phys",
        "outlier_count_high_phys",
        "outlier_count_total_phys",
        "outlier_pct_phys",
        "outlier_count_iqr",
        "outlier_pct_iqr",
        "mean_rr_ms",
        "median_rr_ms",
        "std_rr_ms",
        "iqr_rr_ms",
        "min_rr_ms",
        "max_rr_ms",
        "mean_hr_bpm",
        "sdnn_ms",
        "rmssd_ms",
        "pnn50",
        "duplicate_pct",
        "quality_score",
        "ml_ready",
        "af_labeled_duration_sec",
    ]
    df[trace_cols].to_csv(OUT_DIR / "episode_segment_traceability.csv", index=False)

    generate_plots(df)
    write_report(df)

    run_summary = {
        "n_episodes": int(df[["record_id", "episode_index"]].drop_duplicates().shape[0]),
        "rows": int(len(df)),
        "segment_types": list(SEGMENT_TYPES),
        "horizons_min": HORIZON_MINUTES,
        "output_dir": str(OUT_DIR),
    }
    with open(OUT_DIR / "run_summary.json", "w") as f:
        json.dump(run_summary, f, indent=2)

    print("\nOutputs:")
    print(f"- {OUT_DIR / 'episode_pre_afib_afib_combined_features.csv'}")
    print(f"- {OUT_DIR / 'episode_segment_traceability.csv'}")
    print(f"- {OUT_DIR / 'horizon_segment_summary.csv'}")
    print(f"- {OUT_DIR / 'AFDB_PreAFib_AFib_Combined_Report.md'}")
    print(f"- {PLOTS_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()
