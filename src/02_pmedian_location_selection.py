from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pulp

from distance_utils import compute_distance_matrix
from validation import (
    DataValidationError,
    load_csv_robust,
    validate_columns,
    validate_coordinates,
    validate_no_missing,
    validate_no_negative,
    validate_unique_ids,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pmedian")


@dataclass
class PMedianConfig:
    demand_csv: str
    candidate_csv: str
    demand_id_col: str = "candidate_id"
    demand_lat_col: str = "latitude"
    demand_lon_col: str = "longitude"
    demand_weight_col: str = "demand_weight"
    demand_district_col: str | None = "district"
    candidate_id_col: str = "candidate_id"
    candidate_name_col: str = "candidate_name"
    candidate_lat_col: str = "latitude"
    candidate_lon_col: str = "longitude"
    candidate_district_col: str | None = "district"

    p: int = 22
    distance_method: str = "haversine"
    max_assignment_distance_m: float | None = None
    nearest_candidate_k: int | None = None
    use_sparse_assignment: bool = True
    solver_time_limit_sec: int = 300
    solver_msg: bool = True
    random_seed: int = 42

    must_include_stop_ids: list[str] = field(default_factory=list)
    must_exclude_stop_ids: list[str] = field(default_factory=list)
    district_minimums: dict[str, int] = field(default_factory=dict)
    district_maximums: dict[str, int] = field(default_factory=dict)
    capacity_col: str | None = None

    output_dir: str = "../outputs/pmedian"


def validate_input_data(demand_df: pd.DataFrame, candidate_df: pd.DataFrame, cfg: PMedianConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_columns(demand_df, [cfg.demand_id_col, cfg.demand_lat_col, cfg.demand_lon_col, cfg.demand_weight_col], "demand")
    validate_unique_ids(demand_df, cfg.demand_id_col, "demand")
    validate_no_missing(demand_df, [cfg.demand_weight_col], "demand")
    validate_no_negative(demand_df, cfg.demand_weight_col, "demand")
    d_invalid = validate_coordinates(demand_df, cfg.demand_lat_col, cfg.demand_lon_col, "demand")
    demand_df = demand_df[~d_invalid].reset_index(drop=True)

    validate_columns(candidate_df, [cfg.candidate_id_col, cfg.candidate_name_col, cfg.candidate_lat_col, cfg.candidate_lon_col], "candidate")
    validate_unique_ids(candidate_df, cfg.candidate_id_col, "candidate")
    c_invalid = validate_coordinates(candidate_df, cfg.candidate_lat_col, cfg.candidate_lon_col, "candidate")
    candidate_df = candidate_df[~c_invalid].reset_index(drop=True)

    if cfg.p > len(candidate_df):
        raise DataValidationError(f"p({cfg.p}) cannot exceed candidate count ({len(candidate_df)})")
    return demand_df, candidate_df


def calculate_distance_matrix(demand_df: pd.DataFrame, candidate_df: pd.DataFrame, cfg: PMedianConfig) -> np.ndarray:
    return compute_distance_matrix(
        demand_df[cfg.demand_lat_col].to_numpy(), demand_df[cfg.demand_lon_col].to_numpy(),
        candidate_df[cfg.candidate_lat_col].to_numpy(), candidate_df[cfg.candidate_lon_col].to_numpy(),
        method=cfg.distance_method,
    )


def build_allowed_assignment_pairs(dist: np.ndarray, cfg: PMedianConfig) -> np.ndarray:
    allowed = np.ones_like(dist, dtype=bool)

    if cfg.max_assignment_distance_m is not None:
        allowed &= dist <= cfg.max_assignment_distance_m

    if cfg.nearest_candidate_k is not None:
        k = min(cfg.nearest_candidate_k, dist.shape[1])
        knn_mask = np.zeros_like(allowed)
        nearest_idx = np.argsort(dist, axis=1)[:, :k]
        rows = np.repeat(np.arange(dist.shape[0]), k)
        knn_mask[rows, nearest_idx.ravel()] = True
        allowed &= knn_mask

    return allowed


def validate_assignment_feasibility(allowed: np.ndarray, demand_df: pd.DataFrame) -> None:
    no_option = ~allowed.any(axis=1)
    if no_option.any():
        raise DataValidationError(
            f"{int(no_option.sum())} demand points have no allowed candidate — "
            "relax max_assignment_distance_m or nearest_candidate_k (infeasible)."
        )


def build_pmedian_model(
    demand_df: pd.DataFrame, candidate_df: pd.DataFrame, dist: np.ndarray, allowed: np.ndarray, cfg: PMedianConfig
) -> tuple[pulp.LpProblem, dict, dict]:
    n_demand, n_candidate = dist.shape
    weights = demand_df[cfg.demand_weight_col].to_numpy()
    candidate_ids = candidate_df[cfg.candidate_id_col].tolist()

    prob = pulp.LpProblem("PMedian", pulp.LpMinimize)
    x = {j: pulp.LpVariable(f"x_{j}", cat="Binary") for j in range(n_candidate)}

    z = {}
    for i in range(n_demand):
        for j in range(n_candidate):
            if (not cfg.use_sparse_assignment) or allowed[i, j]:
                z[(i, j)] = pulp.LpVariable(f"z_{i}_{j}", cat="Binary")

    prob += pulp.lpSum(
        weights[i] * dist[i, j] * z[(i, j)] for (i, j) in z
    ), "total_weighted_distance"

    prob += pulp.lpSum(x.values()) == cfg.p, "select_exactly_p"

    for i in range(n_demand):
        js = [j for j in range(n_candidate) if (i, j) in z]
        prob += pulp.lpSum(z[(i, j)] for j in js) == 1, f"assign_exactly_one_{i}"
        for j in js:
            prob += z[(i, j)] <= x[j], f"link_z_x_{i}_{j}"

    id_to_idx = {cid: idx for idx, cid in enumerate(candidate_ids)}
    for sid in cfg.must_include_stop_ids:
        if sid not in id_to_idx:
            raise DataValidationError(f"must_include_stop_ids has unknown ID: {sid}")
        prob += x[id_to_idx[sid]] == 1, f"must_include_{sid}"
    for sid in cfg.must_exclude_stop_ids:
        if sid not in id_to_idx:
            raise DataValidationError(f"must_exclude_stop_ids has unknown ID: {sid}")
        prob += x[id_to_idx[sid]] == 0, f"must_exclude_{sid}"

    if cfg.candidate_district_col is not None:
        district_series = candidate_df[cfg.candidate_district_col]
        for d, min_n in cfg.district_minimums.items():
            idxs = np.where(district_series.values == d)[0]
            prob += pulp.lpSum(x[j] for j in idxs) >= min_n, f"district_min_{d}"
        for d, max_n in cfg.district_maximums.items():
            idxs = np.where(district_series.values == d)[0]
            prob += pulp.lpSum(x[j] for j in idxs) <= max_n, f"district_max_{d}"

    if cfg.capacity_col is not None:
        capacity = candidate_df[cfg.capacity_col].to_numpy()
        for j in range(n_candidate):
            is_ = [i for i in range(n_demand) if (i, j) in z]
            prob += pulp.lpSum(weights[i] * z[(i, j)] for i in is_) <= capacity[j] * x[j], f"capacity_{j}"

    return prob, x, z


def solve_pmedian(prob: pulp.LpProblem, cfg: PMedianConfig) -> str:
    solver = pulp.PULP_CBC_CMD(msg=cfg.solver_msg, timeLimit=cfg.solver_time_limit_sec)
    prob.solve(solver)
    return pulp.LpStatus[prob.status]


def extract_pmedian_solution(
    demand_df: pd.DataFrame, candidate_df: pd.DataFrame, dist: np.ndarray, x: dict, z: dict, cfg: PMedianConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_demand, n_candidate = dist.shape
    candidate_ids = candidate_df[cfg.candidate_id_col].to_numpy()
    demand_ids = demand_df[cfg.demand_id_col].to_numpy()
    weights = demand_df[cfg.demand_weight_col].to_numpy()

    selected = np.array([bool(round(pulp.value(x[j]))) for j in range(n_candidate)])

    assigned_stop_idx = np.full(n_demand, -1, dtype=int)
    for (i, j), var in z.items():
        if round(pulp.value(var)) == 1:
            assigned_stop_idx[i] = j

    assignment_df = pd.DataFrame({
        "demand_id": demand_ids,
        "demand_weight": weights,
        "assigned_stop_id": [candidate_ids[j] if j >= 0 else None for j in assigned_stop_idx],
        "assigned_stop_name": [
            candidate_df.iloc[j][cfg.candidate_name_col] if j >= 0 else None for j in assigned_stop_idx
        ],
        "assignment_distance_m": [dist[i, j] if j >= 0 else np.nan for i, j in enumerate(assigned_stop_idx)],
    })
    assignment_df["weighted_assignment_distance"] = assignment_df["demand_weight"] * assignment_df["assignment_distance_m"]
    assignment_df["within_400m"] = assignment_df["assignment_distance_m"] <= 400.0
    if cfg.demand_district_col and cfg.demand_district_col in demand_df.columns:
        assignment_df["district"] = demand_df[cfg.demand_district_col].to_numpy()
    else:
        assignment_df["district"] = None

    selected_df = candidate_df.copy()
    selected_df["pmedian_selected"] = selected
    selected_df["pmp_selected"] = selected
    return selected_df, assignment_df


def calculate_assignment_metrics(assignment_df: pd.DataFrame, selected_df: pd.DataFrame, status: str, objective_value: float, cfg: PMedianConfig, runtime_sec: float, sparse_used: bool) -> dict:
    dist = assignment_df["assignment_distance_m"].dropna()
    weights = assignment_df["demand_weight"]
    weighted_mean = (assignment_df["weighted_assignment_distance"].sum() / weights.sum()) if weights.sum() > 0 else float("nan")
    return {
        "solver_status": status,
        "objective_value": objective_value,
        "selected_stop_count": int(selected_df["pmedian_selected"].sum()),
        "total_demand_weight": float(weights.sum()),
        "weighted_mean_distance_m": float(weighted_mean),
        "unweighted_mean_distance_m": float(dist.mean()) if len(dist) else float("nan"),
        "max_distance_m": float(dist.max()) if len(dist) else float("nan"),
        "p50_distance_m": float(dist.quantile(0.50)) if len(dist) else float("nan"),
        "p75_distance_m": float(dist.quantile(0.75)) if len(dist) else float("nan"),
        "p90_distance_m": float(dist.quantile(0.90)) if len(dist) else float("nan"),
        "p95_distance_m": float(dist.quantile(0.95)) if len(dist) else float("nan"),
        "share_within_400m": float(assignment_df["within_400m"].mean()),
        "runtime_sec": runtime_sec,
        "sparse_restriction_used": sparse_used,
        "nearest_candidate_k": cfg.nearest_candidate_k,
    }


def build_candidate_summary(selected_df: pd.DataFrame, assignment_df: pd.DataFrame, dist: np.ndarray, cfg: PMedianConfig) -> pd.DataFrame:
    candidate_ids = selected_df[cfg.candidate_id_col].to_numpy()
    nearest_demand_count = []
    nearest_demand_weight = []
    weights = assignment_df["demand_weight"].to_numpy()
    nearest_j_per_demand = np.argmin(dist, axis=1)
    for j in range(len(selected_df)):
        mask = nearest_j_per_demand == j
        nearest_demand_count.append(int(mask.sum()))
        nearest_demand_weight.append(float(weights[mask].sum()))

    assigned_count = assignment_df.groupby("assigned_stop_id").size()
    assigned_weight = assignment_df.groupby("assigned_stop_id")["demand_weight"].sum()

    summary = pd.DataFrame({
        "candidate_id": candidate_ids,
        "pmedian_selected": selected_df["pmedian_selected"].to_numpy(),
        "nearest_demand_count": nearest_demand_count,
        "nearest_demand_weight": nearest_demand_weight,
        "selected_assigned_demand_count": [int(assigned_count.get(cid, 0)) for cid in candidate_ids],
        "selected_assigned_demand_weight": [float(assigned_weight.get(cid, 0.0)) for cid in candidate_ids],
    })
    return summary


def save_pmedian_outputs(selected_df: pd.DataFrame, assignment_df: pd.DataFrame, summary_df: pd.DataFrame, metrics: dict, cfg: PMedianConfig) -> None:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = selected_df.rename(columns={
        cfg.candidate_id_col: "candidate_id",
        cfg.candidate_name_col: "candidate_name",
        cfg.candidate_lat_col: "latitude",
        cfg.candidate_lon_col: "longitude",
    })
    if cfg.candidate_district_col and cfg.candidate_district_col in df.columns:
        df = df.rename(columns={cfg.candidate_district_col: "district"})
    else:
        df["district"] = None

    agg = assignment_df.groupby("assigned_stop_id").agg(
        assigned_demand_count=("demand_id", "count"),
        assigned_demand_weight=("demand_weight", "sum"),
        weighted_distance_sum=("weighted_assignment_distance", "sum"),
        mean_assignment_distance_m=("assignment_distance_m", "mean"),
        max_assignment_distance_m=("assignment_distance_m", "max"),
    ).reset_index().rename(columns={"assigned_stop_id": "candidate_id"})

    df = df.merge(agg, on="candidate_id", how="left")
    for c in ["assigned_demand_count", "assigned_demand_weight", "weighted_distance_sum"]:
        df[c] = df[c].fillna(0)

    rank_src = df.loc[df["pmedian_selected"], ["candidate_id", "assigned_demand_weight"]]
    rank_src = rank_src.sort_values("assigned_demand_weight", ascending=False)
    rank_map = {cid: r + 1 for r, cid in enumerate(rank_src["candidate_id"])}
    df["selection_rank"] = df["candidate_id"].map(rank_map)

    keep_cols = [
        "candidate_id", "candidate_name", "latitude", "longitude", "district",
        "pmedian_selected", "pmp_selected", "assigned_demand_count", "assigned_demand_weight",
        "weighted_distance_sum", "mean_assignment_distance_m", "max_assignment_distance_m",
        "selection_rank",
    ]
    df[keep_cols].to_csv(out_dir / "pmedian_selected_stops.csv", index=False, encoding="utf-8-sig")
    assignment_df.to_csv(out_dir / "pmedian_demand_assignment.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_dir / "pmedian_candidate_summary.csv", index=False, encoding="utf-8-sig")

    with open(out_dir / "pmedian_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(out_dir / "pmedian_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    logger.info("P-Median outputs saved -> %s", out_dir.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P-Median (PMP) based shuttle stop location selection")
    parser.add_argument("--demand-csv", required=True)
    parser.add_argument("--candidate-csv", required=True)
    parser.add_argument("--p", type=int, default=22)
    parser.add_argument("--max-assignment-distance-m", type=float, default=None)
    parser.add_argument("--nearest-candidate-k", type=int, default=None)
    parser.add_argument("--output-dir", default="../outputs/pmedian")
    return parser.parse_args()


def run(cfg: PMedianConfig) -> dict:
    start = time.time()
    demand_df, candidate_df = validate_input_data(load_csv_robust(cfg.demand_csv), load_csv_robust(cfg.candidate_csv), cfg)

    dist = calculate_distance_matrix(demand_df, candidate_df, cfg)
    allowed = build_allowed_assignment_pairs(dist, cfg)
    if cfg.max_assignment_distance_m is not None or cfg.nearest_candidate_k is not None:
        validate_assignment_feasibility(allowed, demand_df)

    prob, x, z = build_pmedian_model(demand_df, candidate_df, dist, allowed, cfg)
    status = solve_pmedian(prob, cfg)
    if status != "Optimal":
        raise RuntimeError(f"P-Median solver status is not Optimal: {status}")

    selected_df, assignment_df = extract_pmedian_solution(demand_df, candidate_df, dist, x, z, cfg)
    assert selected_df["pmedian_selected"].sum() == cfg.p, "selected stop count != p"
    assert assignment_df["assigned_stop_id"].notna().all(), "some demand points are not assigned"
    assert set(assignment_df["assigned_stop_id"]) <= set(selected_df.loc[selected_df["pmedian_selected"], cfg.candidate_id_col]), \
        "demand assigned to a non-selected stop"

    objective_value = float(pulp.value(prob.objective))
    recomputed = float(assignment_df["weighted_assignment_distance"].sum())
    assert abs(objective_value - recomputed) < 1e-4, "objective value mismatch with recomputed weighted distance"

    runtime_sec = time.time() - start
    metrics = calculate_assignment_metrics(assignment_df, selected_df, status, objective_value, cfg, runtime_sec, cfg.use_sparse_assignment)
    summary_df = build_candidate_summary(selected_df, assignment_df, dist, cfg)
    save_pmedian_outputs(selected_df, assignment_df, summary_df, metrics, cfg)

    logger.info("P-Median done: p=%d, weighted_mean_distance=%.1fm, runtime=%.2fs",
                cfg.p, metrics["weighted_mean_distance_m"], runtime_sec)
    return metrics


def main() -> None:
    args = parse_args()
    cfg = PMedianConfig(
        demand_csv=args.demand_csv,
        candidate_csv=args.candidate_csv,
        p=args.p,
        max_assignment_distance_m=args.max_assignment_distance_m,
        nearest_candidate_k=args.nearest_candidate_k,
        output_dir=args.output_dir,
    )
    run(cfg)


if __name__ == "__main__":
    main()
