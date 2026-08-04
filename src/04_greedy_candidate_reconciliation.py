from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from distance_utils import compute_distance_matrix
from validation import load_csv_robust, validate_columns, DataValidationError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("greedy")


@dataclass
class GreedyConfig:
    combined_csv: str
    id_col: str = "candidate_id"
    name_col: str = "candidate_name"
    lat_col: str = "latitude"
    lon_col: str = "longitude"
    district_col: str = "district"

    target_stop_count: int = 22
    min_stop_spacing_m: float = 400.0
    distance_method: str = "haversine"

    score_mode: str = "weighted_components"
    existing_score_col: str = "preliminary_priority_score"
    priority_weights: dict = field(default_factory=lambda: {
        "ahp_score_norm": 0.50,
        "mclp_coverage_norm": 0.20,
        "pmedian_distance_quality_norm": 0.20,
        "selected_by_both": 0.10,
    })
    tie_break_columns: list[str] = field(default_factory=lambda: [
        "priority_score:desc", "selected_by_both:desc", "ahp_score:desc",
        "mclp_coverage_norm:desc", "pmedian_distance_quality_norm:desc", "candidate_id:asc",
    ])

    must_include_stop_ids: list[str] = field(default_factory=list)
    must_exclude_stop_ids: list[str] = field(default_factory=list)
    forced_conflict_policy: str = "error"

    minimum_one_per_district: bool = False
    district_minimums: dict[str, int] = field(default_factory=dict)
    district_maximums: dict[str, int] = field(default_factory=dict)
    repair_max_iterations: int = 100

    output_dir: str = "../outputs/greedy"


STATUS_SELECTED = {"SELECTED_FORCED", "SELECTED_DISTRICT_MINIMUM", "SELECTED_GREEDY", "SELECTED_REPAIR"}


def load_combined_candidates(cfg: GreedyConfig) -> pd.DataFrame:
    return load_csv_robust(cfg.combined_csv)


def validate_candidate_columns(df: pd.DataFrame, cfg: GreedyConfig) -> None:
    validate_columns(df, [cfg.id_col, cfg.name_col, cfg.lat_col, cfg.lon_col], "combined_candidates")
    if cfg.target_stop_count > len(df):
        raise DataValidationError(
            f"target_stop_count({cfg.target_stop_count}) exceeds candidate count ({len(df)})"
        )


def normalize_priority_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "selected_by_both" in df.columns:
        df["selected_by_both"] = df["selected_by_both"].fillna(False).astype(bool)
    for c in ["ahp_score_norm", "mclp_coverage_norm", "pmedian_distance_quality_norm"]:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)
    return df


def calculate_priority_score(df: pd.DataFrame, cfg: GreedyConfig) -> pd.DataFrame:
    df = df.copy()
    if cfg.score_mode == "existing_score":
        if cfg.existing_score_col not in df.columns:
            raise DataValidationError(f"existing_score_col '{cfg.existing_score_col}' not found in data")
        df["priority_score"] = df[cfg.existing_score_col].fillna(0.0)
    elif cfg.score_mode == "weighted_components":
        w = cfg.priority_weights
        score = pd.Series(0.0, index=df.index)
        for col, weight in w.items():
            if col == "selected_by_both":
                score = score + df.get(col, False).astype(int) * weight
            else:
                score = score + df.get(col, 0.0).fillna(0.0) * weight
        df["priority_score"] = score
    else:
        raise DataValidationError(f"unknown score_mode: {cfg.score_mode}")

    if df["priority_score"].isna().any():
        raise DataValidationError("priority_score contains NaN")
    return df


def calculate_candidate_distance_matrix(df: pd.DataFrame, cfg: GreedyConfig) -> np.ndarray:
    return compute_distance_matrix(
        df[cfg.lat_col].to_numpy(), df[cfg.lon_col].to_numpy(),
        df[cfg.lat_col].to_numpy(), df[cfg.lon_col].to_numpy(),
        method=cfg.distance_method,
    )


def build_conflict_pairs(dist: np.ndarray, ids: np.ndarray, cfg: GreedyConfig) -> pd.DataFrame:
    n = len(ids)
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            if dist[i, j] < cfg.min_stop_spacing_m:
                rows.append({"stop_id_a": ids[i], "stop_id_b": ids[j], "distance_m": float(dist[i, j])})
    return pd.DataFrame(rows)


def build_conflict_components(conflict_pairs: pd.DataFrame, ids: np.ndarray) -> dict:
    parent = {cid: cid for cid in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for _, row in conflict_pairs.iterrows():
        union(row["stop_id_a"], row["stop_id_b"])

    roots = {cid: find(cid) for cid in ids}
    root_to_group = {}
    group_id_map = {}
    next_group = 0
    for cid, root in roots.items():
        if root not in root_to_group:
            root_to_group[root] = next_group
            next_group += 1
        group_id_map[cid] = root_to_group[root]
    return group_id_map


def _tie_break_sort(df: pd.DataFrame, tie_break_columns: list[str]) -> pd.DataFrame:
    by, asc = [], []
    for spec in tie_break_columns:
        col, _, direction = spec.partition(":")
        if col not in df.columns:
            continue
        by.append(col)
        asc.append(direction != "desc")
    return df.sort_values(by=by, ascending=asc, kind="mergesort")


def validate_forced_constraints(df: pd.DataFrame, dist: np.ndarray, id_to_idx: dict, cfg: GreedyConfig) -> None:
    for sid in cfg.must_include_stop_ids + cfg.must_exclude_stop_ids:
        if sid not in id_to_idx:
            raise DataValidationError(f"forced include/exclude ID '{sid}' not found in candidate data")
    overlap = set(cfg.must_include_stop_ids) & set(cfg.must_exclude_stop_ids)
    if overlap:
        raise DataValidationError(f"ID present in both must_include and must_exclude: {overlap}")
    if len(cfg.must_include_stop_ids) > cfg.target_stop_count:
        raise DataValidationError(
            f"must_include count ({len(cfg.must_include_stop_ids)}) exceeds target_stop_count({cfg.target_stop_count})"
        )
    for i, a in enumerate(cfg.must_include_stop_ids):
        for b in cfg.must_include_stop_ids[i + 1:]:
            d = dist[id_to_idx[a], id_to_idx[b]]
            if d < cfg.min_stop_spacing_m:
                msg = f"forced stops {a}, {b} are {d:.1f}m apart < min_stop_spacing_m({cfg.min_stop_spacing_m}m)"
                if cfg.forced_conflict_policy == "error":
                    raise DataValidationError(msg)
                logger.warning(msg)


def select_forced_candidates(
    df: pd.DataFrame, dist: np.ndarray, id_to_idx: dict, cfg: GreedyConfig, audit: list
) -> tuple[list[int], list[int]]:
    selected_idx: list[int] = []
    excluded_idx: list[int] = [id_to_idx[sid] for sid in cfg.must_exclude_stop_ids]
    for sid in cfg.must_include_stop_ids:
        idx = id_to_idx[sid]
        selected_idx.append(idx)
        audit.append({"step": len(audit), "candidate_id": sid, "action": "SELECTED_FORCED",
                      "reason": "must_include_stop_ids", "conflicting_stop_ids": "",
                      "score": float(df.loc[idx, "priority_score"]),
                      "selected_count_before": len(selected_idx) - 1, "selected_count_after": len(selected_idx)})
    for sid in cfg.must_exclude_stop_ids:
        audit.append({"step": len(audit), "candidate_id": sid, "action": "REJECTED_MUST_EXCLUDE",
                      "reason": "must_exclude_stop_ids", "conflicting_stop_ids": "",
                      "score": float(df.loc[id_to_idx[sid], "priority_score"]),
                      "selected_count_before": len(selected_idx), "selected_count_after": len(selected_idx)})
    return selected_idx, excluded_idx


def _min_dist_to_set(idx: int, selected_idx: list[int], dist: np.ndarray) -> float:
    if not selected_idx:
        return float("inf")
    return float(dist[idx, selected_idx].min())


def select_district_minimum_candidates(
    df: pd.DataFrame, dist: np.ndarray, selected_idx: list[int], excluded_idx: list[int], cfg: GreedyConfig, audit: list
) -> list[int]:
    if cfg.district_col not in df.columns:
        return selected_idx

    min_map = dict(cfg.district_minimums)
    if cfg.minimum_one_per_district:
        for d in df[cfg.district_col].dropna().unique():
            min_map.setdefault(d, max(min_map.get(d, 0), 1))
    if not min_map:
        return selected_idx

    ranked = _tie_break_sort(df, cfg.tie_break_columns)

    for district, min_n in min_map.items():
        current_count = sum(1 for i in selected_idx if df.loc[i, cfg.district_col] == district)
        district_candidates = ranked[(ranked[cfg.district_col] == district)].index.tolist()
        for idx in district_candidates:
            if current_count >= min_n:
                break
            if idx in selected_idx or idx in excluded_idx:
                continue
            if len(selected_idx) >= cfg.target_stop_count:
                break
            min_d = _min_dist_to_set(idx, selected_idx, dist)
            if min_d < cfg.min_stop_spacing_m:
                audit.append({"step": len(audit), "candidate_id": df.loc[idx, cfg.id_col],
                              "action": "REJECTED_DISTANCE_CONFLICT",
                              "reason": f"distance conflict while filling district_minimum({district}) ({min_d:.1f}m)",
                              "conflicting_stop_ids": ";".join(df.loc[selected_idx, cfg.id_col].astype(str)),
                              "score": float(df.loc[idx, "priority_score"]),
                              "selected_count_before": len(selected_idx), "selected_count_after": len(selected_idx)})
                continue
            selected_idx = selected_idx + [idx]
            current_count += 1
            audit.append({"step": len(audit), "candidate_id": df.loc[idx, cfg.id_col],
                          "action": "SELECTED_DISTRICT_MINIMUM", "reason": f"district_minimum({district})",
                          "conflicting_stop_ids": "", "score": float(df.loc[idx, "priority_score"]),
                          "selected_count_before": len(selected_idx) - 1, "selected_count_after": len(selected_idx)})
    return selected_idx


def greedy_select_remaining(
    df: pd.DataFrame, dist: np.ndarray, selected_idx: list[int], excluded_idx: list[int], cfg: GreedyConfig, audit: list
) -> list[int]:
    ranked = _tie_break_sort(df, cfg.tie_break_columns)
    district_max_count = {d: sum(1 for i in selected_idx if df.loc[i, cfg.district_col] == d)
                          for d in cfg.district_maximums} if cfg.district_col in df.columns else {}

    for idx in ranked.index:
        if len(selected_idx) >= cfg.target_stop_count:
            audit.append({"step": len(audit), "candidate_id": df.loc[idx, cfg.id_col],
                          "action": "REJECTED_TARGET_REACHED", "reason": "target_stop_count reached",
                          "conflicting_stop_ids": "", "score": float(df.loc[idx, "priority_score"]),
                          "selected_count_before": len(selected_idx), "selected_count_after": len(selected_idx)})
            continue
        if idx in selected_idx or idx in excluded_idx:
            continue

        district = df.loc[idx, cfg.district_col] if cfg.district_col in df.columns else None
        if district in cfg.district_maximums and district_max_count.get(district, 0) >= cfg.district_maximums[district]:
            audit.append({"step": len(audit), "candidate_id": df.loc[idx, cfg.id_col],
                          "action": "REJECTED_DISTRICT_MAXIMUM", "reason": f"district_maximum({district}) exceeded",
                          "conflicting_stop_ids": "", "score": float(df.loc[idx, "priority_score"]),
                          "selected_count_before": len(selected_idx), "selected_count_after": len(selected_idx)})
            continue

        min_d = _min_dist_to_set(idx, selected_idx, dist)
        if min_d < cfg.min_stop_spacing_m:
            conflicting = [df.loc[i, cfg.id_col] for i in selected_idx if dist[idx, i] < cfg.min_stop_spacing_m]
            audit.append({"step": len(audit), "candidate_id": df.loc[idx, cfg.id_col],
                          "action": "REJECTED_DISTANCE_CONFLICT",
                          "reason": f"{min_d:.1f}m from an already-selected stop (min_stop_spacing_m={cfg.min_stop_spacing_m}m)",
                          "conflicting_stop_ids": ";".join(str(c) for c in conflicting),
                          "score": float(df.loc[idx, "priority_score"]),
                          "selected_count_before": len(selected_idx), "selected_count_after": len(selected_idx)})
            continue

        selected_idx = selected_idx + [idx]
        if district in district_max_count:
            district_max_count[district] += 1
        audit.append({"step": len(audit), "candidate_id": df.loc[idx, cfg.id_col],
                      "action": "SELECTED_GREEDY", "reason": "greedy selection by descending priority_score",
                      "conflicting_stop_ids": "", "score": float(df.loc[idx, "priority_score"]),
                      "selected_count_before": len(selected_idx) - 1, "selected_count_after": len(selected_idx)})

    return selected_idx


def repair_district_constraints(
    df: pd.DataFrame, dist: np.ndarray, selected_idx: list[int], excluded_idx: list[int], cfg: GreedyConfig, audit: list
) -> list[int]:
    if cfg.district_col not in df.columns:
        return selected_idx

    min_map = dict(cfg.district_minimums)
    if cfg.minimum_one_per_district:
        for d in df[cfg.district_col].dropna().unique():
            min_map.setdefault(d, max(min_map.get(d, 0), 1))
    if not min_map:
        return selected_idx

    ranked = _tie_break_sort(df, cfg.tie_break_columns)

    for _ in range(cfg.repair_max_iterations):
        deficits = {
            d: min_n - sum(1 for i in selected_idx if df.loc[i, cfg.district_col] == d)
            for d, min_n in min_map.items()
        }
        deficits = {d: n for d, n in deficits.items() if n > 0}
        if not deficits:
            break

        repaired_any = False
        for district, _ in deficits.items():
            candidates = [i for i in ranked.index if df.loc[i, cfg.district_col] == district
                          and i not in selected_idx and i not in excluded_idx]
            for cand in candidates:
                min_d = _min_dist_to_set(cand, selected_idx, dist)
                conflicting = [i for i in selected_idx if dist[cand, i] < cfg.min_stop_spacing_m]

                if not conflicting:
                    if len(selected_idx) < cfg.target_stop_count:
                        selected_idx = selected_idx + [cand]
                        repaired_any = True
                        audit.append({"step": len(audit), "candidate_id": df.loc[cand, cfg.id_col],
                                      "action": "SELECTED_REPAIR", "reason": f"district_minimum({district}) repair",
                                      "conflicting_stop_ids": "", "score": float(df.loc[cand, "priority_score"]),
                                      "selected_count_before": len(selected_idx) - 1, "selected_count_after": len(selected_idx)})
                        break
                    lowest = min(selected_idx, key=lambda i: df.loc[i, "priority_score"])
                    if df.loc[lowest, cfg.district_col] in min_map and \
                       sum(1 for i in selected_idx if df.loc[i, cfg.district_col] == df.loc[lowest, cfg.district_col]) <= min_map[df.loc[lowest, cfg.district_col]]:
                        continue
                    new_selected = [i for i in selected_idx if i != lowest] + [cand]
                    selected_idx = new_selected
                    repaired_any = True
                    audit.append({"step": len(audit), "candidate_id": df.loc[cand, cfg.id_col],
                                  "action": "SELECTED_REPAIR", "reason": f"district_minimum({district}) repair, replaced",
                                  "conflicting_stop_ids": str(df.loc[lowest, cfg.id_col]),
                                  "score": float(df.loc[cand, "priority_score"]),
                                  "selected_count_before": len(selected_idx), "selected_count_after": len(selected_idx)})
                    audit.append({"step": len(audit), "candidate_id": df.loc[lowest, cfg.id_col],
                                  "action": "REPLACED_DURING_REPAIR", "reason": f"replaced to satisfy district_minimum({district})",
                                  "conflicting_stop_ids": str(df.loc[cand, cfg.id_col]),
                                  "score": float(df.loc[lowest, "priority_score"]),
                                  "selected_count_before": len(selected_idx), "selected_count_after": len(selected_idx)})
                    break
            if repaired_any:
                break
        if not repaired_any:
            logger.warning("district_minimum repair made no further progress (candidate shortage or distance conflict).")
            break

    return selected_idx


def validate_final_selection(df: pd.DataFrame, dist: np.ndarray, selected_idx: list[int], excluded_idx: list[int], cfg: GreedyConfig) -> None:
    if len(selected_idx) != cfg.target_stop_count:
        raise RuntimeError(
            f"final selected count {len(selected_idx)} != target_stop_count({cfg.target_stop_count}). "
            "Could not reach target due to candidate shortage or distance/district constraints — relax constraints or add candidates."
        )
    sub = dist[np.ix_(selected_idx, selected_idx)]
    np.fill_diagonal(sub, np.inf)
    if sub.min() < cfg.min_stop_spacing_m:
        raise RuntimeError(f"minimum spacing violated among final selected stops (min={sub.min():.1f}m)")
    for sid in cfg.must_exclude_stop_ids:
        if df.loc[df[cfg.id_col] == sid].index[0] in selected_idx:
            raise RuntimeError(f"must_exclude stop {sid} is in the final selection")
    for sid in cfg.must_include_stop_ids:
        idx = df.loc[df[cfg.id_col] == sid].index[0]
        if idx not in selected_idx:
            raise RuntimeError(f"must_include stop {sid} is missing from the final selection")
    if cfg.district_col in df.columns:
        for d, min_n in cfg.district_minimums.items():
            cnt = sum(1 for i in selected_idx if df.loc[i, cfg.district_col] == d)
            if cnt < min_n:
                raise RuntimeError(f"district_minimum not satisfied: {d} ({cnt}/{min_n})")
        for d, max_n in cfg.district_maximums.items():
            cnt = sum(1 for i in selected_idx if df.loc[i, cfg.district_col] == d)
            if cnt > max_n:
                raise RuntimeError(f"district_maximum exceeded: {d} ({cnt}/{max_n})")


def build_candidate_decision_table(
    df: pd.DataFrame, dist: np.ndarray, selected_idx: list[int], conflict_group_map: dict, cfg: GreedyConfig, audit: pd.DataFrame
) -> pd.DataFrame:
    df = df.copy()
    df["final_selected"] = df.index.isin(selected_idx)

    last_action = audit.groupby("candidate_id").tail(1).set_index("candidate_id")["action"] if len(audit) else pd.Series(dtype=object)
    df["selection_stage"] = df[cfg.id_col].map(last_action).fillna("NOT_EVALUATED")
    df["rejection_reason"] = df[cfg.id_col].map(
        audit.groupby("candidate_id").tail(1).set_index("candidate_id")["reason"] if len(audit) else {}
    )

    others = [i for i in df.index if i not in selected_idx]
    min_dist_to_selected = []
    blocking_ids = []
    for idx in df.index:
        if not selected_idx:
            min_dist_to_selected.append(np.nan)
            blocking_ids.append("")
            continue
        others_selected = [i for i in selected_idx if i != idx]
        if not others_selected:
            min_dist_to_selected.append(np.nan)
            blocking_ids.append("")
            continue
        d = dist[idx, others_selected]
        min_dist_to_selected.append(float(d.min()))
        blocking = [df.loc[i, cfg.id_col] for i, dd in zip(others_selected, d) if dd < cfg.min_stop_spacing_m]
        blocking_ids.append(";".join(str(b) for b in blocking))

    df["minimum_distance_to_selected_m"] = min_dist_to_selected
    df["blocking_selected_stop_ids"] = blocking_ids
    df["conflict_group_id"] = df[cfg.id_col].map(conflict_group_map)

    rank_src = df.loc[df["final_selected"]].sort_values("priority_score", ascending=False)
    rank_map = {cid: r + 1 for r, cid in enumerate(rank_src[cfg.id_col])}
    df["final_selection_rank"] = df[cfg.id_col].map(rank_map)
    df["decision_order"] = range(len(df))
    return df


def calculate_greedy_metrics(df: pd.DataFrame, dist: np.ndarray, selected_idx: list[int], cfg: GreedyConfig, runtime_sec: float, repair_iterations: int, status: str) -> dict:
    sub = dist[np.ix_(selected_idx, selected_idx)] if selected_idx else np.array([[]])
    if len(selected_idx) > 1:
        np.fill_diagonal(sub, np.inf)
        min_pairwise = float(sub.min())
    else:
        min_pairwise = float("nan")

    district_counts = {}
    if cfg.district_col in df.columns:
        for d in df[cfg.district_col].dropna().unique():
            district_counts[d] = int(sum(1 for i in selected_idx if df.loc[i, cfg.district_col] == d))

    return {
        "target_stop_count": cfg.target_stop_count,
        "final_selected_count": len(selected_idx),
        "min_stop_spacing_m": cfg.min_stop_spacing_m,
        "minimum_pairwise_selected_distance_m": min_pairwise,
        "selected_by_both_count": int(df.loc[selected_idx, "selected_by_both"].sum()) if "selected_by_both" in df.columns else None,
        "mclp_selected_count": int(df.loc[selected_idx, "mclp_selected"].sum()) if "mclp_selected" in df.columns else None,
        "pmedian_selected_count": int(df.loc[selected_idx, "pmedian_selected"].sum()) if "pmedian_selected" in df.columns else None,
        "district_counts": district_counts,
        "district_minimum_satisfied": all(
            sum(1 for i in selected_idx if df.loc[i, cfg.district_col] == d) >= n
            for d, n in cfg.district_minimums.items()
        ) if cfg.district_col in df.columns else None,
        "district_maximum_satisfied": all(
            sum(1 for i in selected_idx if df.loc[i, cfg.district_col] == d) <= n
            for d, n in cfg.district_maximums.items()
        ) if cfg.district_col in df.columns else None,
        "repair_iteration_count": repair_iterations,
        "runtime_sec": runtime_sec,
        "status": status,
    }


def save_greedy_outputs(decision_df: pd.DataFrame, audit_df: pd.DataFrame, conflict_pairs: pd.DataFrame, metrics: dict, cfg: GreedyConfig) -> None:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    decision_df.to_csv(out_dir / "greedy_priority_ranking.csv", index=False, encoding="utf-8-sig")

    final = decision_df[decision_df["final_selected"]].rename(columns={
        cfg.id_col: "candidate_id", cfg.name_col: "candidate_name", cfg.lat_col: "latitude", cfg.lon_col: "longitude",
    }).sort_values("final_selection_rank")
    final_cols = [
        "candidate_id", "candidate_name", "latitude", "longitude", cfg.district_col,
        "final_selected", "final_selection_rank", "priority_score", "ahp_score",
        "mclp_selected", "pmedian_selected", "selected_by_both",
        "minimum_distance_to_selected_m", "selection_stage",
    ]
    final_cols = [c for c in final_cols if c in final.columns]
    final[final_cols].rename(columns={cfg.district_col: "district"}).to_csv(
        out_dir / "greedy_final_selected_stops.csv", index=False, encoding="utf-8-sig"
    )

    audit_df.to_csv(out_dir / "greedy_decision_audit.csv", index=False, encoding="utf-8-sig")
    conflict_pairs.to_csv(out_dir / "greedy_conflict_pairs.csv", index=False, encoding="utf-8-sig")

    rejected = decision_df[~decision_df["final_selected"]]
    rejected.to_csv(out_dir / "greedy_rejected_candidates.csv", index=False, encoding="utf-8-sig")

    with open(out_dir / "greedy_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)
    with open(out_dir / "greedy_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    logger.info("Greedy final selection done: %d/%d -> %s",
                metrics["final_selected_count"], metrics["target_stop_count"], out_dir.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Greedy 400m dedup and final stop selection")
    parser.add_argument("--combined-csv", required=True)
    parser.add_argument("--target-stop-count", type=int, default=22)
    parser.add_argument("--min-stop-spacing-m", type=float, default=400.0)
    parser.add_argument("--output-dir", default="../outputs/greedy")
    return parser.parse_args()


def run(cfg: GreedyConfig) -> dict:
    import time
    start = time.time()

    df = load_combined_candidates(cfg)
    validate_candidate_columns(df, cfg)
    df = normalize_priority_features(df)
    df = calculate_priority_score(df, cfg)
    df = df.reset_index(drop=True)

    dist = calculate_candidate_distance_matrix(df, cfg)
    ids = df[cfg.id_col].to_numpy()
    id_to_idx = {cid: idx for idx, cid in enumerate(ids)}

    conflict_pairs = build_conflict_pairs(dist, ids, cfg)
    conflict_group_map = build_conflict_components(conflict_pairs, ids)

    validate_forced_constraints(df, dist, id_to_idx, cfg)

    audit: list = []
    selected_idx, excluded_idx = select_forced_candidates(df, dist, id_to_idx, cfg, audit)
    selected_idx = select_district_minimum_candidates(df, dist, selected_idx, excluded_idx, cfg, audit)
    selected_idx = greedy_select_remaining(df, dist, selected_idx, excluded_idx, cfg, audit)

    repair_iterations = 0
    if cfg.district_minimums or cfg.minimum_one_per_district:
        before = list(selected_idx)
        selected_idx = repair_district_constraints(df, dist, selected_idx, excluded_idx, cfg, audit)
        repair_iterations = 1 if selected_idx != before else 0

    status = "SUCCESS"
    try:
        validate_final_selection(df, dist, selected_idx, excluded_idx, cfg)
    except RuntimeError as e:
        status = "FAILED"
        logger.error(str(e))
        raise

    audit_df = pd.DataFrame(audit)
    decision_df = build_candidate_decision_table(df, dist, selected_idx, conflict_group_map, cfg, audit_df)

    assert decision_df["priority_score"].notna().all(), "priority_score contains NaN"
    non_selected_no_reason = decision_df[(~decision_df["final_selected"]) & (decision_df["rejection_reason"].isna())]
    if len(non_selected_no_reason):
        logger.warning("%d rejected candidates have no recorded reason (NOT_EVALUATED)", len(non_selected_no_reason))
    ranks = decision_df.loc[decision_df["final_selected"], "final_selection_rank"].sort_values().tolist()
    assert ranks == list(range(1, len(ranks) + 1)), "final_selection_rank is not consecutive starting from 1"
    assert set(decision_df.loc[decision_df["final_selected"], cfg.id_col]) <= set(df[cfg.id_col]), \
        "selected stops not found in original candidates"

    runtime_sec = time.time() - start
    metrics = calculate_greedy_metrics(decision_df, dist, selected_idx, cfg, runtime_sec, repair_iterations, status)
    save_greedy_outputs(decision_df, audit_df, conflict_pairs, metrics, cfg)

    logger.info("Greedy done: %d selected, min pairwise distance=%.1fm, runtime=%.2fs",
                metrics["final_selected_count"], metrics["minimum_pairwise_selected_distance_m"], runtime_sec)
    return metrics


def main() -> None:
    args = parse_args()
    cfg = GreedyConfig(
        combined_csv=args.combined_csv,
        target_stop_count=args.target_stop_count,
        min_stop_spacing_m=args.min_stop_spacing_m,
        output_dir=args.output_dir,
    )
    run(cfg)


if __name__ == "__main__":
    main()
