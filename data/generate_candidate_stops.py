from __future__ import annotations

import numpy as np
import pandas as pd

RNG = np.random.default_rng(7)

DISTRICT_CENTERS = {
    "단구동": (37.330, 127.960),
    "개운동": (37.345, 127.955),
    "명륜동": (37.343, 127.950),
    "단계동": (37.348, 127.945),
    "무실동": (37.320, 127.935),
    "반곡관설동": (37.345, 127.990),
    "지정면": (37.400, 127.930),
}

AHP_WEIGHTS = {
    "인구밀집도": 0.5993,
    "총_유동인구수": 0.1774,
    "관광객방문수": 0.1031,
    "승하차수": 0.1203,
}

N_PER_DISTRICT = {
    "단구동": 7, "개운동": 5, "명륜동": 6, "단계동": 6,
    "무실동": 7, "반곡관설동": 7, "지정면": 6,
}


def minmax_norm(s: pd.Series) -> pd.Series:
    if s.max() == s.min():
        return pd.Series(0.5, index=s.index)
    return (s - s.min()) / (s.max() - s.min())


def generate() -> pd.DataFrame:
    rows = []
    cid = 1
    for district, n in N_PER_DISTRICT.items():
        lat0, lon0 = DISTRICT_CENTERS[district]
        is_festival_area = district == "명륜동"
        for i in range(n):
            lat = lat0 + RNG.normal(0, 0.007)
            lon = lon0 + RNG.normal(0, 0.007)
            base_pop = RNG.uniform(2000, 8000)
            base_float = RNG.uniform(2000, 20000)
            base_tour = RNG.uniform(50, 800)
            base_board = RNG.integers(5, 500)
            if is_festival_area:
                base_pop *= RNG.uniform(1.8, 2.5)
                base_float *= RNG.uniform(1.5, 2.2)
                base_tour *= RNG.uniform(2.0, 3.0)
            rows.append({
                "candidate_id": f"STOP_{cid:03d}",
                "candidate_name": f"{district}_{i+1}번_임시정류장",
                "latitude": round(lat, 6),
                "longitude": round(lon, 6),
                "district": district,
                "인구밀집도": round(base_pop, 2),
                "총_유동인구수": round(base_float, 2),
                "관광객방문수": round(base_tour, 2),
                "승하차수": int(base_board),
            })
            cid += 1

    df = pd.DataFrame(rows)

    for col in AHP_WEIGHTS:
        df[f"{col}_norm"] = minmax_norm(df[col])

    df["ahp_score"] = sum(
        df[f"{col}_norm"] * w for col, w in AHP_WEIGHTS.items()
    )
    df["demand_weight"] = df["ahp_score"]
    return df


if __name__ == "__main__":
    df = generate()
    out_path = "candidate_stops_sample.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Candidate/demand data generated: {df.shape} -> {out_path}")
    print(df.groupby("district").size())
    print(df.sort_values("ahp_score", ascending=False).head(5)[
        ["candidate_id", "candidate_name", "district", "ahp_score"]
    ])
