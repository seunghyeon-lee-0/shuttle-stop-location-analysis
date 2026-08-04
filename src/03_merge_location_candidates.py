from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from distance_utils import compute_distance_matrix
from validation import load_csv_robust, validate_columns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("merge")


@dataclass
class MergeConfig:
    mclp_csv: str
    pmedian_csv: str
    original_candidate_csv: str | None = None
    ahp_csv: str | None = None

    id_col: str = "candidate_id"
    name_col: str = "candidate_name"
    lat_col: str = "latitude"
    lon_col: str = "longitude"
    district_col: str = "district"
    ahp_score_col: str = "ahp_score"

    enable_coordinate_fuzzy_match: bool = False
    coordinate_match_tolerance_m: float = 20.0

    priority_weights: dict = field(default_factory=lambda: {
        "selected_by_both_bonus": 1.0,
        "mclp_coverage_norm_weight": 0.3,
        "pmedian_distance_quality_norm_weight": 0.3,
        "ahp_score_norm_weight": 0.4,
    })
    compute_preliminary_priority: bool = True

    output_dir: str = "../outputs/merged"


def load_model_results(cfg: MergeConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    mclp = load_csv_robust(cfg.mclp_csv)
    pmedian = load_csv_robust(cfg.pmedian_csv)
    return mclp, pmedian


def validate_model_result_columns(mclp: pd.DataFrame, pmedian: pd.DataFrame, cfg: MergeConfig) -> None:
    validate_columns(mclp, [cfg.id_col, cfg.name_col, cfg.lat_col, cfg.lon_col, "mclp_selected"], "mclp_selected_stops")
    validate_columns(pmedian, [cfg.id_col, cfg.name_col, cfg.lat_col, cfg.lon_col, "pmedian_selected"], "pmedian_selected_stops")


def normalize_stop_identifiers(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    df = df.copy()
    df[id_col] = df[id_col].astype(str).str.strip()
    return df


def exact_id_merge(mclp: pd.DataFrame, pmedian: pd.DataFrame, cfg: MergeConfig) -> pd.DataFrame:
    left = mclp[[cfg.id_col, cfg.name_col, cfg.lat_col, cfg.lon_col,
                 "mclp_selected", "covered_demand_weight", "covered_demand_count"]].rename(
        columns={"covered_demand_weight": "mclp_covered_demand_weight",
                 "covered_demand_count": "mclp_covered_demand_count"}
    )
    if cfg.district_col in mclp.columns:
        left[cfg.district_col] = mclp[cfg.district_col]

    right = pmedian[[cfg.id_col, "pmedian_selected", "assigned_demand_weight", "mean_assignment_distance_m"]].rename(
        columns={"assigned_demand_weight": "pmedian_assigned_demand_weight",
                 "mean_assignment_distance_m": "pmedian_mean_assignment_distance_m"}
    )
    if "pmp_selected" in pmedian.columns:
        right["pmp_selected"] = pmedian["pmp_selected"]

    merged = left.merge(right, on=cfg.id_col, how="outer", indicator="_merge_src")
    merged["source_record_count"] = merged["_merge_src"].map({"left_only": 1, "right_only": 1, "both": 2})
    merged = merged.drop(columns=["_merge_src"])

    for c in ["mclp_selected", "pmedian_selected", "pmp_selected"]:
        if c in merged.columns:
            merged[c] = merged[c].fillna(False).astype(bool)

    merged["merge_method"] = "exact_id"
    merged["merge_warning"] = ""
    return merged


def optional_coordinate_match(merged: pd.DataFrame, cfg: MergeConfig) -> pd.DataFrame:
    if not cfg.enable_coordinate_fuzzy_match:
        return merged

    lat = merged[cfg.lat_col].to_numpy()
    lon = merged[cfg.lon_col].to_numpy()
    dist = compute_distance_matrix(lat, lon, lat, lon, method="haversine")
    np.fill_diagonal(dist, np.inf)

    close_pairs = np.argwhere(dist <= cfg.coordinate_match_tolerance_m)
    ids = merged[cfg.id_col].to_numpy()
    for i, j in close_pairs:
        if i < j and ids[i] != ids[j]:
            msg = f"potential duplicate within {dist[i, j]:.1f}m: {ids[j]}"
            merged.at[i, "merge_warning"] = (merged.at[i, "merge_warning"] + "; " + msg).strip("; ")
    return merged


def detect_merge_conflicts(merged: pd.DataFrame, cfg: MergeConfig) -> pd.DataFrame:
    conflicts = []
    if cfg.name_col not in merged.columns:
        return pd.DataFrame(conflicts)

    for name, group in merged.groupby(cfg.name_col):
        if len(group) < 2:
            continue
        lat = group[cfg.lat_col].to_numpy()
        lon = group[cfg.lon_col].to_numpy()
        dist = compute_distance_matrix(lat, lon, lat, lon, method="haversine")
        np.fill_diagonal(dist, 0)
        if dist.max() > 500:
            conflicts.append({
                "candidate_name": name,
                "candidate_ids": ";".join(group[cfg.id_col].astype(str)),
                "max_distance_m": float(dist.max()),
                "reason": "same name, coordinates differ by >500m — not auto-merged",
            })
    return pd.DataFrame(conflicts)


def attach_original_candidate_features(merged: pd.DataFrame, cfg: MergeConfig) -> pd.DataFrame:
    if not cfg.original_candidate_csv:
        return merged
    orig = load_csv_robust(cfg.original_candidate_csv)
    extra_cols = [c for c in orig.columns if c not in merged.columns and c != cfg.id_col]
    return merged.merge(orig[[cfg.id_col] + extra_cols], on=cfg.id_col, how="left")


def attach_ahp_features(merged: pd.DataFrame, cfg: MergeConfig) -> pd.DataFrame:
    if cfg.ahp_score_col in merged.columns:
        merged = merged.rename(columns={cfg.ahp_score_col: "ahp_score"})
        return merged
    if cfg.ahp_csv:
        ahp = load_csv_robust(cfg.ahp_csv)
        if cfg.ahp_score_col in ahp.columns:
            merged = merged.merge(ahp[[cfg.id_col, cfg.ahp_score_col]], on=cfg.id_col, how="left")
            merged = merged.rename(columns={cfg.ahp_score_col: "ahp_score"})
            return merged
    logger.warning("AHP score not found; keeping ahp_score as NaN (not fabricated).")
    merged["ahp_score"] = np.nan
    return merged


def _minmax_norm(s: pd.Series) -> pd.Series:
    if s.notna().sum() == 0:
        return s
    lo, hi = s.min(), s.max()
    if hi == lo:
        return s.apply(lambda v: 0.5 if pd.notna(v) else np.nan)
    return (s - lo) / (hi - lo)


def calculate_normalized_scores(merged: pd.DataFrame) -> pd.DataFrame:
    merged["selected_by_both"] = merged["mclp_selected"] & merged["pmedian_selected"]
    merged["selected_model_count"] = merged["mclp_selected"].astype(int) + merged["pmedian_selected"].astype(int)

    merged["mclp_coverage_norm"] = _minmax_norm(merged.get("mclp_covered_demand_weight", pd.Series(dtype=float)))
    if "pmedian_mean_assignment_distance_m" in merged.columns:
        dist_norm = _minmax_norm(merged["pmedian_mean_assignment_distance_m"])
        merged["pmedian_distance_quality_norm"] = 1 - dist_norm
    else:
        merged["pmedian_distance_quality_norm"] = np.nan
    merged["ahp_score_norm"] = _minmax_norm(merged["ahp_score"]) if "ahp_score" in merged.columns else np.nan
    return merged


def calculate_preliminary_priority(merged: pd.DataFrame, cfg: MergeConfig) -> pd.DataFrame:
    if not cfg.compute_preliminary_priority:
        merged["preliminary_priority_score"] = np.nan
        return merged

    w = cfg.priority_weights
    components = {
        "mclp_coverage_norm": w.get("mclp_coverage_norm_weight", 0.0),
        "pmedian_distance_quality_norm": w.get("pmedian_distance_quality_norm_weight", 0.0),
        "ahp_score_norm": w.get("ahp_score_norm_weight", 0.0),
    }
    available = {k: w for k, w in components.items() if merged[k].notna().any()}
    total_w = sum(available.values())
    if total_w == 0:
        merged["preliminary_priority_score"] = np.nan
        return merged
    available = {k: v / total_w for k, v in available.items()}

    score = pd.Series(0.0, index=merged.index)
    for col, weight in available.items():
        score = score + merged[col].fillna(0) * weight
    score = score + merged["selected_by_both"].astype(int) * w.get("selected_by_both_bonus", 0.0)
    merged["preliminary_priority_score"] = score
    return merged


def save_merge_outputs(merged: pd.DataFrame, conflicts: pd.DataFrame, cfg: MergeConfig) -> None:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    final_cols = [
        "candidate_id", "candidate_name", "latitude", "longitude", "district",
        "mclp_selected", "pmedian_selected", "pmp_selected", "selected_by_both", "selected_model_count",
        "mclp_covered_demand_weight", "mclp_covered_demand_count",
        "pmedian_assigned_demand_weight", "pmedian_mean_assignment_distance_m",
        "ahp_score", "source_record_count", "merge_method", "merge_warning",
        "mclp_coverage_norm", "pmedian_distance_quality_norm", "ahp_score_norm",
        "preliminary_priority_score",
    ]
    rename = {cfg.id_col: "candidate_id", cfg.name_col: "candidate_name", cfg.lat_col: "latitude", cfg.lon_col: "longitude"}
    out = merged.rename(columns={k: v for k, v in rename.items() if k in merged.columns})
    if cfg.district_col in out.columns and cfg.district_col != "district":
        out = out.rename(columns={cfg.district_col: "district"})
    for c in final_cols:
        if c not in out.columns:
            out[c] = np.nan
    out[final_cols].to_csv(out_dir / "combined_candidate_stops.csv", index=False, encoding="utf-8-sig")

    conflicts.to_csv(out_dir / "merge_conflicts.csv", index=False, encoding="utf-8-sig")

    metrics = {
        "mclp_selected_count": int(merged["mclp_selected"].sum()),
        "pmedian_selected_count": int(merged["pmedian_selected"].sum()),
        "unique_candidate_count": int(merged[cfg.id_col].nunique()),
        "selected_by_both_count": int(merged["selected_by_both"].sum()),
        "mclp_only_count": int(((merged["mclp_selected"]) & (~merged["pmedian_selected"])).sum()),
        "pmedian_only_count": int(((~merged["mclp_selected"]) & (merged["pmedian_selected"])).sum()),
        "merge_conflict_count": int(len(conflicts)),
        "coordinate_match_count": int((merged["merge_warning"] != "").sum()),
        "exact_id_match_count": int((merged["source_record_count"] == 2).sum()),
    }
    with open(out_dir / "merge_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(out_dir / "merge_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    logger.info("Merge done: %d unique candidates (MCLP %d + P-Median %d, common %d) -> %s",
                metrics["unique_candidate_count"], metrics["mclp_selected_count"],
                metrics["pmedian_selected_count"], metrics["selected_by_both_count"], out_dir.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge MCLP and P-Median results")
    parser.add_argument("--mclp-csv", required=True)
    parser.add_argument("--pmedian-csv", required=True)
    parser.add_argument("--original-candidate-csv", default=None)
    parser.add_argument("--ahp-csv", default=None)
    parser.add_argument("--output-dir", default="../outputs/merged")
    return parser.parse_args()


def run(cfg: MergeConfig) -> dict:
    mclp, pmedian = load_model_results(cfg)
    validate_model_result_columns(mclp, pmedian, cfg)
    mclp = normalize_stop_identifiers(mclp, cfg.id_col)
    pmedian = normalize_stop_identifiers(pmedian, cfg.id_col)

    before_mclp, before_pmedian = int(mclp["mclp_selected"].sum()), int(pmedian["pmedian_selected"].sum())

    merged = exact_id_merge(mclp, pmedian, cfg)
    merged = optional_coordinate_match(merged, cfg)
    conflicts = detect_merge_conflicts(merged, cfg)
    merged = attach_original_candidate_features(merged, cfg)
    merged = attach_ahp_features(merged, cfg)
    merged = calculate_normalized_scores(merged)
    merged = calculate_preliminary_priority(merged, cfg)

    assert not merged[cfg.id_col].duplicated().any(), "duplicate ID after merge"
    assert int(merged["mclp_selected"].sum()) == before_mclp, "MCLP selected count not preserved after merge"
    assert int(merged["pmedian_selected"].sum()) == before_pmedian, "P-Median selected count not preserved after merge"
    for c in ["mclp_coverage_norm", "pmedian_distance_quality_norm", "ahp_score_norm"]:
        vals = merged[c].dropna()
        assert vals.between(0, 1).all(), f"{c} value out of [0, 1] range"

    save_merge_outputs(merged, conflicts, cfg)
    return merged


def main() -> None:
    args = parse_args()
    cfg = MergeConfig(
        mclp_csv=args.mclp_csv,
        pmedian_csv=args.pmedian_csv,
        original_candidate_csv=args.original_candidate_csv,
        ahp_csv=args.ahp_csv,
        output_dir=args.output_dir,
    )
    run(cfg)


if __name__ == "__main__":
    main()
