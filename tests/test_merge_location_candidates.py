import pandas as pd
from conftest import load_src_module

merge_mod = load_src_module("03_merge_location_candidates.py")


def _mclp_df():
    return pd.DataFrame({
        "candidate_id": ["A", "B", "C"],
        "candidate_name": ["A", "B", "C"],
        "latitude": [0, 1, 2], "longitude": [0, 1, 2],
        "district": ["d1", "d1", "d2"],
        "mclp_selected": [True, True, False],
        "covered_demand_weight": [10.0, 5.0, 0.0],
        "covered_demand_count": [3, 2, 0],
    })


def _pmedian_df():
    return pd.DataFrame({
        "candidate_id": ["A", "D"],
        "candidate_name": ["A", "D"],
        "latitude": [0, 3], "longitude": [0, 3],
        "district": ["d1", "d2"],
        "pmedian_selected": [True, True],
        "pmp_selected": [True, True],
        "assigned_demand_weight": [8.0, 4.0],
        "mean_assignment_distance_m": [50.0, 120.0],
    })


def _cfg(tmp_path):
    return merge_mod.MergeConfig(mclp_csv="unused", pmedian_csv="unused", output_dir=str(tmp_path))


def test_exact_id_merge_preserves_selected_counts(tmp_path):
    mclp, pmedian = _mclp_df(), _pmedian_df()
    cfg = _cfg(tmp_path)
    merged = merge_mod.exact_id_merge(mclp, pmedian, cfg)
    assert merged["mclp_selected"].sum() == 2  # A, B
    assert merged["pmedian_selected"].sum() == 2  # A, D
    assert set(merged["candidate_id"]) == {"A", "B", "C", "D"}


def test_selected_by_both_and_only_counts(tmp_path):
    mclp, pmedian = _mclp_df(), _pmedian_df()
    cfg = _cfg(tmp_path)
    merged = merge_mod.exact_id_merge(mclp, pmedian, cfg)
    merged = merge_mod.calculate_normalized_scores(merge_mod.attach_ahp_features(merged, cfg))
    row_a = merged.loc[merged["candidate_id"] == "A"].iloc[0]
    assert row_a["selected_by_both"] == True
    mclp_only = merged.loc[(merged["mclp_selected"]) & (~merged["pmedian_selected"])]
    assert set(mclp_only["candidate_id"]) == {"B"}
    pmedian_only = merged.loc[(~merged["mclp_selected"]) & (merged["pmedian_selected"])]
    assert set(pmedian_only["candidate_id"]) == {"D"}


def test_ahp_missing_kept_as_nan_not_fabricated(tmp_path):
    mclp, pmedian = _mclp_df(), _pmedian_df()
    cfg = _cfg(tmp_path)
    merged = merge_mod.exact_id_merge(mclp, pmedian, cfg)
    merged = merge_mod.attach_ahp_features(merged, cfg)
    assert merged["ahp_score"].isna().all()


def test_normalized_scores_within_0_1(tmp_path):
    mclp, pmedian = _mclp_df(), _pmedian_df()
    cfg = _cfg(tmp_path)
    merged = merge_mod.exact_id_merge(mclp, pmedian, cfg)
    merged = merge_mod.calculate_normalized_scores(merge_mod.attach_ahp_features(merged, cfg))
    for c in ["mclp_coverage_norm", "pmedian_distance_quality_norm"]:
        vals = merged[c].dropna()
        assert vals.between(0, 1).all()
