from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path

import yaml

SRC_DIR = Path(__file__).resolve().parent
STAGE_ORDER = ["mclp", "pmedian", "merge", "greedy"]


def _load_module(filename: str):
    path = SRC_DIR / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def _sha256_of_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return "MISSING"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def setup_logging(log_path: Path, level: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )


def build_mclp_config(cfg: dict, mclp_mod):
    c, col_d, col_c = cfg["columns"]["demand"], cfg["columns"]["demand"], cfg["columns"]["candidate"]
    return mclp_mod.MCLPConfig(
        demand_csv=cfg["paths"]["demand_csv"], candidate_csv=cfg["paths"]["candidate_csv"],
        demand_id_col=col_d["id"], demand_lat_col=col_d["latitude"], demand_lon_col=col_d["longitude"],
        demand_weight_col=col_d["weight"], demand_district_col=col_d.get("district"),
        candidate_id_col=col_c["id"], candidate_name_col=col_c["name"],
        candidate_lat_col=col_c["latitude"], candidate_lon_col=col_c["longitude"],
        candidate_district_col=col_c.get("district"),
        p=cfg["optimization"]["target_stop_count"], coverage_radius_m=cfg["distance"]["mclp_coverage_radius_m"],
        distance_method=cfg["distance"]["method"], solver_time_limit_sec=cfg["optimization"]["time_limit_sec"],
        solver_msg=False, random_seed=cfg["optimization"]["random_seed"],
        minimum_one_per_district=cfg["constraints"]["minimum_one_per_district"],
        district_minimums=cfg["constraints"]["district_minimums"], district_maximums=cfg["constraints"]["district_maximums"],
        must_include_stop_ids=cfg["constraints"]["must_include_stop_ids"], must_exclude_stop_ids=cfg["constraints"]["must_exclude_stop_ids"],
        output_dir=str(Path(cfg["paths"]["output_dir"]) / "mclp"),
    )


def build_pmedian_config(cfg: dict, pmedian_mod):
    col_d, col_c = cfg["columns"]["demand"], cfg["columns"]["candidate"]
    return pmedian_mod.PMedianConfig(
        demand_csv=cfg["paths"]["demand_csv"], candidate_csv=cfg["paths"]["candidate_csv"],
        demand_id_col=col_d["id"], demand_lat_col=col_d["latitude"], demand_lon_col=col_d["longitude"],
        demand_weight_col=col_d["weight"], demand_district_col=col_d.get("district"),
        candidate_id_col=col_c["id"], candidate_name_col=col_c["name"],
        candidate_lat_col=col_c["latitude"], candidate_lon_col=col_c["longitude"],
        candidate_district_col=col_c.get("district"),
        p=cfg["optimization"]["target_stop_count"], distance_method=cfg["distance"]["method"],
        max_assignment_distance_m=cfg["pmedian"]["max_assignment_distance_m"],
        nearest_candidate_k=cfg["pmedian"]["nearest_candidate_k"],
        solver_time_limit_sec=cfg["optimization"]["time_limit_sec"], solver_msg=False,
        random_seed=cfg["optimization"]["random_seed"],
        must_include_stop_ids=cfg["constraints"]["must_include_stop_ids"], must_exclude_stop_ids=cfg["constraints"]["must_exclude_stop_ids"],
        district_minimums=cfg["constraints"]["district_minimums"], district_maximums=cfg["constraints"]["district_maximums"],
        capacity_col=cfg["pmedian"]["capacity_col"],
        output_dir=str(Path(cfg["paths"]["output_dir"]) / "pmedian"),
    )


def build_merge_config(cfg: dict, merge_mod):
    col_c, col_ahp = cfg["columns"]["candidate"], cfg["columns"]["ahp"]
    return merge_mod.MergeConfig(
        mclp_csv=str(Path(cfg["paths"]["output_dir"]) / "mclp" / "mclp_selected_stops.csv"),
        pmedian_csv=str(Path(cfg["paths"]["output_dir"]) / "pmedian" / "pmedian_selected_stops.csv"),
        original_candidate_csv=cfg["paths"]["candidate_csv"], ahp_csv=cfg["paths"]["ahp_csv"],
        id_col="candidate_id", name_col="candidate_name", lat_col="latitude", lon_col="longitude",
        district_col=col_c.get("district") or "district", ahp_score_col=col_ahp["score"],
        priority_weights={
            "selected_by_both_bonus": cfg["greedy"]["priority_weights"].get("selected_by_both", 0.10),
            "mclp_coverage_norm_weight": cfg["greedy"]["priority_weights"].get("mclp_coverage_norm", 0.20),
            "pmedian_distance_quality_norm_weight": cfg["greedy"]["priority_weights"].get("pmedian_distance_quality_norm", 0.20),
            "ahp_score_norm_weight": cfg["greedy"]["priority_weights"].get("ahp_score_norm", 0.50),
        },
        output_dir=str(Path(cfg["paths"]["output_dir"]) / "merged"),
    )


def build_greedy_config(cfg: dict, greedy_mod):
    return greedy_mod.GreedyConfig(
        combined_csv=str(Path(cfg["paths"]["output_dir"]) / "merged" / "combined_candidate_stops.csv"),
        target_stop_count=cfg["optimization"]["target_stop_count"],
        min_stop_spacing_m=cfg["distance"]["greedy_min_stop_spacing_m"], distance_method=cfg["distance"]["method"],
        score_mode=cfg["greedy"]["score_mode"], existing_score_col=cfg["greedy"]["existing_score_col"],
        priority_weights=cfg["greedy"]["priority_weights"], tie_break_columns=cfg["greedy"]["tie_break_columns"],
        must_include_stop_ids=cfg["constraints"]["must_include_stop_ids"], must_exclude_stop_ids=cfg["constraints"]["must_exclude_stop_ids"],
        forced_conflict_policy=cfg["greedy"]["forced_conflict_policy"],
        minimum_one_per_district=cfg["constraints"]["minimum_one_per_district"],
        district_minimums=cfg["constraints"]["district_minimums"], district_maximums=cfg["constraints"]["district_maximums"],
        repair_max_iterations=cfg["greedy"]["repair_max_iterations"],
        output_dir=str(Path(cfg["paths"]["output_dir"]) / "greedy"),
    )


def run_pipeline(config_path: str, resume: bool, start_from: str, stop_after: str, dry_run: bool) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir / "logs" / "pipeline.log", cfg["runtime"]["log_level"])
    logger = logging.getLogger("pipeline")

    start_idx = STAGE_ORDER.index(start_from) if start_from else 0
    stop_idx = STAGE_ORDER.index(stop_after) if stop_after else len(STAGE_ORDER) - 1

    summary = {
        "input_file_hashes": {
            "demand_csv": _sha256_of_file(cfg["paths"]["demand_csv"]),
            "candidate_csv": _sha256_of_file(cfg["paths"]["candidate_csv"]),
        },
        "config_file_hash": _sha256_of_file(config_path),
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python_version": sys.version.split()[0],
        "pulp_version": None,
        "mclp_status": "SKIPPED", "pmedian_status": "SKIPPED",
        "merge_status": "SKIPPED", "greedy_status": "SKIPPED",
        "final_selected_stop_count": None, "final_output_path": None,
    }

    try:
        import pulp
        summary["pulp_version"] = pulp.__version__
    except ImportError:
        pass

    manifest_rows = []
    t0 = time.time()

    for i, stage in enumerate(STAGE_ORDER):
        if i < start_idx or i > stop_idx:
            continue

        stage_output_marker = {
            "mclp": out_dir / "mclp" / "mclp_metrics.json",
            "pmedian": out_dir / "pmedian" / "pmedian_metrics.json",
            "merge": out_dir / "merged" / "merge_metrics.json",
            "greedy": out_dir / "greedy" / "greedy_metrics.json",
        }[stage]

        if resume and stage_output_marker.exists():
            logger.info("[resume] stage %s already has results, skipping: %s", stage, stage_output_marker)
            summary[f"{stage}_status"] = "RESUMED_SKIPPED"
            manifest_rows.append({"stage": stage, "status": "RESUMED_SKIPPED", "output": str(stage_output_marker)})
            continue

        if dry_run:
            logger.info("[dry-run] would run stage %s (not executed)", stage)
            manifest_rows.append({"stage": stage, "status": "DRY_RUN", "output": str(stage_output_marker)})
            continue

        logger.info("=== starting stage: %s ===", stage)
        try:
            if stage == "mclp":
                mod = _load_module("01_mclp_location_selection.py")
                mod.run(build_mclp_config(cfg, mod))
            elif stage == "pmedian":
                mod = _load_module("02_pmedian_location_selection.py")
                mod.run(build_pmedian_config(cfg, mod))
            elif stage == "merge":
                if not (out_dir / "mclp" / "mclp_selected_stops.csv").exists() or \
                   not (out_dir / "pmedian" / "pmedian_selected_stops.csv").exists():
                    raise FileNotFoundError("merge stage inputs (mclp/pmedian results) are missing. Run mclp/pmedian first.")
                mod = _load_module("03_merge_location_candidates.py")
                mod.run(build_merge_config(cfg, mod))
            elif stage == "greedy":
                if not (out_dir / "merged" / "combined_candidate_stops.csv").exists():
                    raise FileNotFoundError("greedy stage input (combined_candidate_stops.csv) is missing. Run merge first.")
                mod = _load_module("04_greedy_candidate_reconciliation.py")
                metrics = mod.run(build_greedy_config(cfg, mod))
                summary["final_selected_stop_count"] = metrics["final_selected_count"]
                summary["final_output_path"] = str(out_dir / "greedy" / "greedy_final_selected_stops.csv")

            summary[f"{stage}_status"] = "SUCCESS"
            manifest_rows.append({"stage": stage, "status": "SUCCESS", "output": str(stage_output_marker)})
        except Exception as e:
            summary[f"{stage}_status"] = f"FAILED: {e}"
            manifest_rows.append({"stage": stage, "status": f"FAILED: {e}", "output": str(stage_output_marker)})
            logger.error("stage %s failed: %s", stage, e)
            break

    summary["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    summary["total_runtime_sec"] = time.time() - t0

    with open(out_dir / "pipeline_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    import pandas as pd
    pd.DataFrame(manifest_rows).to_csv(out_dir / "pipeline_manifest.csv", index=False, encoding="utf-8-sig")

    logger.info("Pipeline finished. summary=%s", json.dumps(summary, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full MCLP/P-Median/Merge/Greedy pipeline runner")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--start-from", choices=STAGE_ORDER, default=None)
    parser.add_argument("--stop-after", choices=STAGE_ORDER, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_pipeline(args.config, args.resume, args.start_from, args.stop_after, args.dry_run)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
