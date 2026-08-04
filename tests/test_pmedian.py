import pandas as pd
from conftest import load_src_module

pmedian = load_src_module("02_pmedian_location_selection.py")


def _toy_data():
    # 3 candidates, 6 demand points (2 near each), p=2 -> the farthest candidate must be dropped
    candidate_df = pd.DataFrame({
        "candidate_id": ["C1", "C2", "C3"],
        "candidate_name": ["C1", "C2", "C3"],
        "latitude": [0, 1000, 2000],
        "longitude": [0, 0, 0],
        "district": ["A", "A", "B"],
    })
    demand_df = pd.DataFrame({
        "candidate_id": ["D1", "D2", "D3", "D4", "D5", "D6"],
        "latitude": [10, 20, 1010, 990, 2010, 1990],
        "longitude": [0, 0, 0, 0, 0, 0],
        "demand_weight": [1, 1, 1, 1, 1, 1],
    })
    return demand_df, candidate_df


def _base_config(tmp_path):
    return pmedian.PMedianConfig(
        demand_csv="unused", candidate_csv="unused",
        p=2, distance_method="euclidean", solver_msg=False, output_dir=str(tmp_path),
    )


def test_exactly_p_selected_and_all_assigned(tmp_path):
    demand_df, candidate_df = _toy_data()
    cfg = _base_config(tmp_path)
    dist = pmedian.calculate_distance_matrix(demand_df, candidate_df, cfg)
    allowed = pmedian.build_allowed_assignment_pairs(dist, cfg)
    prob, x, z = pmedian.build_pmedian_model(demand_df, candidate_df, dist, allowed, cfg)
    status = pmedian.solve_pmedian(prob, cfg)
    assert status == "Optimal"

    selected_df, assignment_df = pmedian.extract_pmedian_solution(demand_df, candidate_df, dist, x, z, cfg)
    assert selected_df["pmedian_selected"].sum() == 2
    assert assignment_df["assigned_stop_id"].notna().all()
    # C2 is central and should always be selected (minimizes total distance)
    assert "C2" in set(selected_df.loc[selected_df["pmedian_selected"], "candidate_id"])


def test_max_assignment_distance_infeasible_detected(tmp_path):
    demand_df, candidate_df = _toy_data()
    cfg = _base_config(tmp_path)
    cfg.max_assignment_distance_m = 5.0  # every demand point is at least 10m away -> infeasible
    dist = pmedian.calculate_distance_matrix(demand_df, candidate_df, cfg)
    allowed = pmedian.build_allowed_assignment_pairs(dist, cfg)
    try:
        pmedian.validate_assignment_feasibility(allowed, demand_df)
        assert False, "infeasible was not detected"
    except pmedian.DataValidationError:
        pass


def test_objective_matches_recomputed_weighted_distance(tmp_path):
    demand_df, candidate_df = _toy_data()
    cfg = _base_config(tmp_path)
    dist = pmedian.calculate_distance_matrix(demand_df, candidate_df, cfg)
    allowed = pmedian.build_allowed_assignment_pairs(dist, cfg)
    prob, x, z = pmedian.build_pmedian_model(demand_df, candidate_df, dist, allowed, cfg)
    pmedian.solve_pmedian(prob, cfg)
    selected_df, assignment_df = pmedian.extract_pmedian_solution(demand_df, candidate_df, dist, x, z, cfg)

    import pulp
    objective = float(pulp.value(prob.objective))
    recomputed = float(assignment_df["weighted_assignment_distance"].sum())
    assert abs(objective - recomputed) < 1e-6
