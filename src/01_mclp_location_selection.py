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
logger = logging.getLogger("mclp")


@dataclass
class MCLPConfig:
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
    coverage_radius_m: float = 400.0
    distance_method: str = "haversine"
    solver_time_limit_sec: int = 300
    solver_msg: bool = True
    random_seed: int = 42

    minimum_one_per_district: bool = False
    district_minimums: dict[str, int] = field(default_factory=dict)
    district_maximums: dict[str, int] = field(default_factory=dict)
    must_include_stop_ids: list[str] = field(default_factory=list)
    must_exclude_stop_ids: list[str] = field(default_factory=list)

    output_dir: str = "../outputs/mclp"


def validate_demand_data(df: pd.DataFrame, cfg: MCLPConfig) -> pd.DataFrame:
    required = [cfg.demand_id_col, cfg.demand_lat_col, cfg.demand_lon_col, cfg.demand_weight_col]
    validate_columns(df, required, "demand")
    validate_unique_ids(df, cfg.demand_id_col, "demand")
    validate_no_missing(df, [cfg.demand_weight_col], "demand")
    validate_no_negative(df, cfg.demand_weight_col, "demand")
    invalid_coord = validate_coordinates(df, cfg.demand_lat_col, cfg.demand_lon_col, "demand")
    if invalid_coord.any():
        df = df[~invalid_coord].copy()
        logger.warning("Dropped %d demand points with invalid coordinates", int(invalid_coord.sum()))
    return df.reset_index(drop=True)


def validate_candidate_data(df: pd.DataFrame, cfg: MCLPConfig) -> pd.DataFrame:
    required = [cfg.candidate_id_col, cfg.candidate_name_col, cfg.candidate_lat_col, cfg.candidate_lon_col]
    validate_columns(df, required, "candidate")
    validate_unique_ids(df, cfg.candidate_id_col, "candidate")
    invalid_coord = validate_coordinates(df, cfg.candidate_lat_col, cfg.candidate_lon_col, "candidate")
    if invalid_coord.any():
        df = df[~invalid_coord].copy()
        logger.warning("Dropped %d candidates with invalid coordinates", int(invalid_coord.sum()))
    if cfg.p > len(df):
        raise DataValidationError(f"p({cfg.p}) cannot exceed candidate count ({len(df)})")
    return df.reset_index(drop=True)


def build_coverage_matrix(
    demand_df: pd.DataFrame, candidate_df: pd.DataFrame, cfg: MCLPConfig
) -> tuple[np.ndarray, np.ndarray]:
    dist = compute_distance_matrix(
        demand_df[cfg.demand_lat_col].to_numpy(),
        demand_df[cfg.demand_lon_col].to_numpy(),
        candidate_df[cfg.candidate_lat_col].to_numpy(),
        candidate_df[cfg.candidate_lon_col].to_numpy(),
        method=cfg.distance_method,
    )
    coverage = dist <= cfg.coverage_radius_m
    uncoverable = ~coverage.any(axis=1)
    if uncoverable.any():
        logger.warning(
            "%d demand points have no candidate within coverage_radius_m=%s",
            int(uncoverable.sum()), cfg.coverage_radius_m,
        )
    return coverage, dist


def validate_location_constraints(candidate_df: pd.DataFrame, cfg: MCLPConfig) -> None:
    if not cfg.district_minimums:
        return
    if cfg.candidate_district_col is None:
        raise DataValidationError("district_minimums is set but candidate_district_col is None")
    counts = candidate_df[cfg.candidate_district_col].value_counts()
    for dist_name, min_n in cfg.district_minimums.items():
        available = int(counts.get(dist_name, 0))
        if available < min_n:
            raise DataValidationError(
                f"district '{dist_name}' requires {min_n} but only {available} candidates available (infeasible)"
            )
    if sum(cfg.district_minimums.values()) > cfg.p:
        raise DataValidationError("sum of district_minimums exceeds p (infeasible)")


def build_mclp_model(
    demand_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    coverage: np.ndarray,
    cfg: MCLPConfig,
) -> tuple[pulp.LpProblem, dict, dict]:
    n_demand, n_candidate = coverage.shape
    candidate_ids = candidate_df[cfg.candidate_id_col].tolist()

    prob = pulp.LpProblem("MCLP", pulp.LpMaximize)
    x = {j: pulp.LpVariable(f"x_{j}", cat="Binary") for j in range(n_candidate)}
    y = {i: pulp.LpVariable(f"y_{i}", cat="Binary") for i in range(n_demand)}

    weights = demand_df[cfg.demand_weight_col].to_numpy()
    prob += pulp.lpSum(weights[i] * y[i] for i in range(n_demand)), "total_covered_weighted_demand"

    prob += pulp.lpSum(x.values()) == cfg.p, "select_exactly_p"

    for i in range(n_demand):
        covering_js = np.where(coverage[i])[0]
        if len(covering_js) == 0:
            prob += y[i] == 0, f"uncoverable_demand_{i}"
            continue
        prob += y[i] <= pulp.lpSum(x[j] for j in covering_js), f"coverage_link_{i}"

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
        districts = district_series.unique().tolist()

        min_map = dict(cfg.district_minimums)
        if cfg.minimum_one_per_district:
            for d in districts:
                min_map.setdefault(d, max(min_map.get(d, 0), 1))

        for d, min_n in min_map.items():
            idxs = np.where(district_series.values == d)[0]
            prob += pulp.lpSum(x[j] for j in idxs) >= min_n, f"district_min_{d}"

        for d, max_n in cfg.district_maximums.items():
            idxs = np.where(district_series.values == d)[0]
            prob += pulp.lpSum(x[j] for j in idxs) <= max_n, f"district_max_{d}"

    return prob, x, y


def solve_mclp(prob: pulp.LpProblem, cfg: MCLPConfig) -> str:
    solver = pulp.PULP_CBC_CMD(msg=cfg.solver_msg, timeLimit=cfg.solver_time_limit_sec)
    prob.solve(solver)
    return pulp.LpStatus[prob.status]


def extract_mclp_solution(
    candidate_df: pd.DataFrame, x: dict, cfg: MCLPConfig
) -> pd.DataFrame:
    df = candidate_df.copy()
    df["mclp_selected"] = [bool(round(pulp.value(x[j]))) for j in range(len(df))]
    return df


def assign_demand_to_selected_stops(
    demand_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    coverage: np.ndarray,
    dist: np.ndarray,
    selected_mask: np.ndarray,
    cfg: MCLPConfig,
) -> pd.DataFrame:
    out = demand_df[[cfg.demand_id_col, cfg.demand_weight_col]].copy()
    out.columns = ["demand_id", "demand_weight"]

    selected_idx = np.where(selected_mask)[0]
    candidate_ids = candidate_df[cfg.candidate_id_col].to_numpy()

    covered_list, nearest_id_list, nearest_dist_list, covering_ids_list = [], [], [], []
    for i in range(len(demand_df)):
        covering_selected = [j for j in selected_idx if coverage[i, j]]
        covered_list.append(len(covering_selected) > 0)
        covering_ids_list.append(";".join(candidate_ids[covering_selected]) if covering_selected else "")
        if len(selected_idx) > 0:
            nearest_j = selected_idx[np.argmin(dist[i, selected_idx])]
            nearest_id_list.append(candidate_ids[nearest_j])
            nearest_dist_list.append(float(dist[i, nearest_j]))
        else:
            nearest_id_list.append(None)
            nearest_dist_list.append(np.nan)

    out["covered"] = covered_list
    out["nearest_selected_stop_id"] = nearest_id_list
    out["nearest_selected_stop_distance_m"] = nearest_dist_list
    out["covering_selected_stop_ids"] = covering_ids_list
    return out


def calculate_mclp_metrics(
    demand_df: pd.DataFrame,
    assignment_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    status: str,
    objective_value: float,
    cfg: MCLPConfig,
    runtime_sec: float,
) -> dict:
    total_w = float(demand_df[cfg.demand_weight_col].sum())
    covered_w = float(assignment_df.loc[assignment_df["covered"], "demand_weight"].sum())
    return {
        "solver_status": status,
        "objective_value": objective_value,
        "total_demand_weight": total_w,
        "covered_demand_weight": covered_w,
        "uncovered_demand_weight": total_w - covered_w,
        "weighted_coverage_rate": (covered_w / total_w) if total_w > 0 else 0.0,
        "selected_stop_count": int(selected_df["mclp_selected"].sum()),
        "coverage_radius_m": cfg.coverage_radius_m,
        "runtime_sec": runtime_sec,
    }


def save_mclp_outputs(
    selected_df: pd.DataFrame,
    assignment_df: pd.DataFrame,
    coverage: np.ndarray,
    metrics: dict,
    cfg: MCLPConfig,
) -> None:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = selected_df.copy()
    df["potential_covered_demand_count"] = coverage.sum(axis=0)

    selected_mask = df["mclp_selected"].to_numpy()
    demand_weight_arr = assignment_df["demand_weight"].to_numpy()

    covered_count_selected = np.zeros(len(df), dtype=int)
    covered_weight_selected = np.zeros(len(df))
    selected_cover_count_per_demand = coverage[:, selected_mask].sum(axis=1) if selected_mask.any() else np.zeros(len(demand_weight_arr))
    exclusive_weight_selected = np.zeros(len(df))

    for j in np.where(selected_mask)[0]:
        covered_i = np.where(coverage[:, j])[0]
        covered_count_selected[j] = len(covered_i)
        covered_weight_selected[j] = float(demand_weight_arr[covered_i].sum())
        exclusive_i = covered_i[selected_cover_count_per_demand[covered_i] == 1]
        exclusive_weight_selected[j] = float(demand_weight_arr[exclusive_i].sum())

    df["covered_demand_count"] = covered_count_selected
    df["covered_demand_weight"] = covered_weight_selected
    df["exclusive_covered_demand_weight"] = exclusive_weight_selected

    rank = df.loc[selected_mask].sort_values("covered_demand_weight", ascending=False)
    rank_map = {cid: r + 1 for r, cid in enumerate(rank[cfg.candidate_id_col])}
    df["selection_rank"] = df[cfg.candidate_id_col].map(rank_map)

    out_cols = {
        cfg.candidate_id_col: "candidate_id",
        cfg.candidate_name_col: "candidate_name",
        cfg.candidate_lat_col: "latitude",
        cfg.candidate_lon_col: "longitude",
    }
    df = df.rename(columns=out_cols)
    if cfg.candidate_district_col and cfg.candidate_district_col in df.columns:
        df = df.rename(columns={cfg.candidate_district_col: "district"})
    else:
        df["district"] = None

    keep_cols = [
        "candidate_id", "candidate_name", "latitude", "longitude", "district",
        "mclp_selected", "covered_demand_count", "covered_demand_weight",
        "exclusive_covered_demand_weight", "selection_rank",
        "potential_covered_demand_count",
    ]
    df[keep_cols].to_csv(out_dir / "mclp_selected_stops.csv", index=False, encoding="utf-8-sig")
    assignment_df.to_csv(out_dir / "mclp_demand_assignment.csv", index=False, encoding="utf-8-sig")

    df[["candidate_id", "mclp_selected", "potential_covered_demand_count"]].assign(
        potential_covered_demand_weight=df["covered_demand_weight"]
    ).to_csv(out_dir / "mclp_candidate_summary.csv", index=False, encoding="utf-8-sig")

    with open(out_dir / "mclp_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(out_dir / "mclp_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    logger.info("MCLP outputs saved -> %s", out_dir.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MCLP-based shuttle stop location selection")
    parser.add_argument("--demand-csv", required=True)
    parser.add_argument("--candidate-csv", required=True)
    parser.add_argument("--p", type=int, default=22)
    parser.add_argument("--coverage-radius-m", type=float, default=400.0)
    parser.add_argument("--output-dir", default="../outputs/mclp")
    return parser.parse_args()


def run(cfg: MCLPConfig) -> dict:
    start = time.time()
    demand_df = validate_demand_data(load_csv_robust(cfg.demand_csv), cfg)
    candidate_df = validate_candidate_data(load_csv_robust(cfg.candidate_csv), cfg)
    validate_location_constraints(candidate_df, cfg)

    coverage, dist = build_coverage_matrix(demand_df, candidate_df, cfg)
    prob, x, y = build_mclp_model(demand_df, candidate_df, coverage, cfg)
    status = solve_mclp(prob, cfg)
    if status != "Optimal":
        raise RuntimeError(f"MCLP solver status is not Optimal: {status}")

    selected_df = extract_mclp_solution(candidate_df, x, cfg)
    selected_mask = selected_df["mclp_selected"].to_numpy()

    assert selected_mask.sum() == cfg.p, (
        f"selected stop count ({int(selected_mask.sum())}) != p ({cfg.p})"
    )

    assignment_df = assign_demand_to_selected_stops(demand_df, candidate_df, coverage, dist, selected_mask, cfg)

    objective_value = float(pulp.value(prob.objective))
    recomputed = float(assignment_df.loc[assignment_df["covered"], "demand_weight"].sum())
    assert abs(objective_value - recomputed) < 1e-6, "objective value mismatch with recomputed covered demand"

    runtime_sec = time.time() - start
    metrics = calculate_mclp_metrics(demand_df, assignment_df, selected_df, status, objective_value, cfg, runtime_sec)
    save_mclp_outputs(selected_df, assignment_df, coverage, metrics, cfg)

    logger.info("MCLP done: p=%d, coverage_rate=%.4f, runtime=%.2fs",
                cfg.p, metrics["weighted_coverage_rate"], runtime_sec)
    return metrics


def main() -> None:
    args = parse_args()
    cfg = MCLPConfig(
        demand_csv=args.demand_csv,
        candidate_csv=args.candidate_csv,
        p=args.p,
        coverage_radius_m=args.coverage_radius_m,
        output_dir=args.output_dir,
    )
    run(cfg)


if __name__ == "__main__":
    main()
