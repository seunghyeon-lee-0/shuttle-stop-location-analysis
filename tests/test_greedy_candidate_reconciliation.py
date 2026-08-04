import pandas as pd
from conftest import load_src_module

greedy = load_src_module("04_greedy_candidate_reconciliation.py")


def _toy_df():
    # 5 candidates: A,B within 100m (conflict), C,D,E all 400m+ apart
    return pd.DataFrame({
        "candidate_id": ["A", "B", "C", "D", "E"],
        "candidate_name": ["A", "B", "C", "D", "E"],
        "latitude": [0, 0, 0, 0, 0],
        "longitude": [0, 100, 1000, 2000, 3000],
        "district": ["d1", "d1", "d1", "d2", "d2"],
        "preliminary_priority_score": [0.9, 0.95, 0.5, 0.4, 0.3],
        "ahp_score": [0.9, 0.95, 0.5, 0.4, 0.3],
        "ahp_score_norm": [0.9, 0.95, 0.5, 0.4, 0.3],
        "mclp_coverage_norm": [0.5, 0.5, 0.5, 0.5, 0.5],
        "pmedian_distance_quality_norm": [0.5, 0.5, 0.5, 0.5, 0.5],
        "selected_by_both": [False, True, False, False, False],
    })


def _cfg(tmp_path, **kwargs):
    base = dict(
        combined_csv="unused", target_stop_count=3, min_stop_spacing_m=400.0,
        distance_method="euclidean", score_mode="existing_score",
        existing_score_col="preliminary_priority_score", output_dir=str(tmp_path),
    )
    base.update(kwargs)
    return greedy.GreedyConfig(**base)


def test_higher_score_wins_within_conflict_distance(tmp_path):
    df = _toy_df()
    cfg = _cfg(tmp_path)
    df = greedy.calculate_priority_score(greedy.normalize_priority_features(df), cfg)
    dist = greedy.calculate_candidate_distance_matrix(df, cfg)
    audit = []
    selected_idx = greedy.greedy_select_remaining(df, dist, [], [], cfg, audit)
    selected_ids = set(df.loc[selected_idx, "candidate_id"])
    # A and B conflict at 100m -> only the higher-scoring B should be selected
    assert "B" in selected_ids and "A" not in selected_ids
    assert selected_ids == {"B", "C", "D"}


def test_far_apart_candidates_all_selectable(tmp_path):
    df = _toy_df()
    cfg = _cfg(tmp_path, target_stop_count=3)
    df = greedy.calculate_priority_score(greedy.normalize_priority_features(df), cfg)
    dist = greedy.calculate_candidate_distance_matrix(df, cfg)
    # C, D, E are 1000m+ apart -> all should be selectable
    for a, b in [("C", "D"), ("D", "E"), ("C", "E")]:
        ia = df.index[df["candidate_id"] == a][0]
        ib = df.index[df["candidate_id"] == b][0]
        assert dist[ia, ib] >= cfg.min_stop_spacing_m


def test_deterministic_tie_break_by_candidate_id(tmp_path):
    df = _toy_df()
    df["preliminary_priority_score"] = [0.5, 0.5, 0.5, 0.5, 0.5]  # all tied
    df["selected_by_both"] = [False, False, False, False, False]  # tie on the next tie-break column too
    df["ahp_score"] = [0.5, 0.5, 0.5, 0.5, 0.5]
    df["mclp_coverage_norm"] = [0.5, 0.5, 0.5, 0.5, 0.5]
    df["pmedian_distance_quality_norm"] = [0.5, 0.5, 0.5, 0.5, 0.5]
    cfg = _cfg(tmp_path, target_stop_count=3)
    df = greedy.calculate_priority_score(greedy.normalize_priority_features(df), cfg)
    ranked1 = greedy._tie_break_sort(df, cfg.tie_break_columns)["candidate_id"].tolist()
    ranked2 = greedy._tie_break_sort(df, cfg.tie_break_columns)["candidate_id"].tolist()
    assert ranked1 == ranked2 == sorted(ranked1)  # tie-break by candidate_id ascending


def test_forced_include_conflict_raises_error(tmp_path):
    df = _toy_df()
    cfg = _cfg(tmp_path, must_include_stop_ids=["A", "B"], forced_conflict_policy="error")
    df = greedy.calculate_priority_score(greedy.normalize_priority_features(df), cfg)
    dist = greedy.calculate_candidate_distance_matrix(df, cfg)
    id_to_idx = {cid: i for i, cid in enumerate(df["candidate_id"])}
    try:
        greedy.validate_forced_constraints(df, dist, id_to_idx, cfg)
        assert False, "distance conflict between must_include stops was not detected"
    except greedy.DataValidationError:
        pass


def test_target_count_shortfall_raises_runtime_error(tmp_path):
    # A,B conflict so only 4 candidates (B,C,D,E) are usable; requesting target=5 should fail
    df = _toy_df()
    cfg = _cfg(tmp_path, target_stop_count=5)
    df = greedy.calculate_priority_score(greedy.normalize_priority_features(df), cfg)
    dist = greedy.calculate_candidate_distance_matrix(df, cfg)
    audit = []
    selected_idx = greedy.greedy_select_remaining(df, dist, [], [], cfg, audit)
    try:
        greedy.validate_final_selection(df, dist, selected_idx, [], cfg)
        assert False, "an error should have been raised for insufficient candidates"
    except RuntimeError:
        pass


def test_all_pairwise_selected_distances_meet_minimum(tmp_path):
    df = _toy_df()
    cfg = _cfg(tmp_path, target_stop_count=3)
    df = greedy.calculate_priority_score(greedy.normalize_priority_features(df), cfg)
    dist = greedy.calculate_candidate_distance_matrix(df, cfg)
    audit = []
    selected_idx = greedy.greedy_select_remaining(df, dist, [], [], cfg, audit)
    sub = dist[selected_idx][:, selected_idx]
    import numpy as np
    for i in range(len(selected_idx)):
        for j in range(len(selected_idx)):
            if i != j:
                assert sub[i, j] >= cfg.min_stop_spacing_m
