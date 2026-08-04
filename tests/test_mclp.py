import pandas as pd
from conftest import load_src_module

mclp = load_src_module("01_mclp_location_selection.py")


def _toy_data():
    # small hand-verifiable dataset using lat/lon columns as x/y (meters) with euclidean distance
    candidate_df = pd.DataFrame({
        "candidate_id": ["C1", "C2", "C3", "C4"],
        "candidate_name": ["C1", "C2", "C3", "C4"],
        "latitude": [0, 0, 1000, 1000],   # y
        "longitude": [0, 1000, 0, 1000],  # x
        "district": ["A", "A", "B", "B"],
    })
    demand_df = pd.DataFrame({
        "candidate_id": ["D1", "D2", "D3", "D4", "D5", "D6"],
        "latitude": [50, 50, 950, 950, 500, 10],
        "longitude": [50, 950, 50, 950, 500, 10],
        "demand_weight": [10, 20, 30, 40, 5, 15],
    })
    return demand_df, candidate_df


def _base_config(tmp_path):
    return mclp.MCLPConfig(
        demand_csv="unused", candidate_csv="unused",
        p=2, coverage_radius_m=500.0, distance_method="euclidean",
        solver_msg=False, output_dir=str(tmp_path),
    )


def test_distance_matrix_shape_and_coverage(tmp_path):
    demand_df, candidate_df = _toy_data()
    cfg = _base_config(tmp_path)
    coverage, dist = mclp.build_coverage_matrix(demand_df, candidate_df, cfg)
    assert dist.shape == (6, 4)
    # D5(500,500) is ~707m from every candidate, exceeding R=500 -> not coverable
    assert not coverage[4].any()
    # D1(50,50) is ~70.7m from C1(0,0) -> coverable
    assert coverage[0, 0]


def test_mclp_selects_best_two_candidates(tmp_path):
    demand_df, candidate_df = _toy_data()
    cfg = _base_config(tmp_path)
    coverage, dist = mclp.build_coverage_matrix(demand_df, candidate_df, cfg)
    prob, x, y = mclp.build_mclp_model(demand_df, candidate_df, coverage, cfg)
    status = mclp.solve_mclp(prob, cfg)
    assert status == "Optimal"

    selected_df = mclp.extract_mclp_solution(candidate_df, x, cfg)
    selected_ids = set(selected_df.loc[selected_df["mclp_selected"], "candidate_id"])
    # C4(40) and C3(30) have the largest covered demand, so both should be selected
    assert selected_ids == {"C3", "C4"}
    assert selected_df["mclp_selected"].sum() == 2

    import pulp
    assert abs(float(pulp.value(prob.objective)) - 70.0) < 1e-6


def test_forced_include_exclude(tmp_path):
    demand_df, candidate_df = _toy_data()
    cfg = _base_config(tmp_path)
    cfg.must_include_stop_ids = ["C1"]
    cfg.must_exclude_stop_ids = ["C4"]
    coverage, dist = mclp.build_coverage_matrix(demand_df, candidate_df, cfg)
    prob, x, y = mclp.build_mclp_model(demand_df, candidate_df, coverage, cfg)
    mclp.solve_mclp(prob, cfg)
    selected_df = mclp.extract_mclp_solution(candidate_df, x, cfg)
    selected_ids = set(selected_df.loc[selected_df["mclp_selected"], "candidate_id"])
    assert "C1" in selected_ids
    assert "C4" not in selected_ids
    assert len(selected_ids) == 2


def test_infeasible_district_constraint_detected(tmp_path):
    _, candidate_df = _toy_data()
    cfg = _base_config(tmp_path)
    cfg.district_minimums = {"A": 5}  # district A only has 2 candidates -> infeasible
    try:
        mclp.validate_location_constraints(candidate_df, cfg)
        assert False, "infeasible constraint was not detected"
    except mclp.DataValidationError:
        pass
