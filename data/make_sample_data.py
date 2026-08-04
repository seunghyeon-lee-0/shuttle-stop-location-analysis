import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

DONG_TABLE = pd.DataFrame([
    {"행정동코드": "51130310", "시도명": "강원특별자치도", "시군구명": "원주시", "읍면동명": "단구동"},
    {"행정동코드": "51130320", "시도명": "강원특별자치도", "시군구명": "원주시", "읍면동명": "개운동"},
    {"행정동코드": "51130330", "시도명": "강원특별자치도", "시군구명": "원주시", "읍면동명": "명륜동"},
    {"행정동코드": "51130340", "시도명": "강원특별자치도", "시군구명": "원주시", "읍면동명": "단계동"},
    {"행정동코드": "51130350", "시도명": "강원특별자치도", "시군구명": "원주시", "읍면동명": "무실동"},
    {"행정동코드": "51130360", "시도명": "강원특별자치도", "시군구명": "원주시", "읍면동명": "반곡관설동"},
    {"행정동코드": "51130370", "시도명": "강원특별자치도", "시군구명": "원주시", "읍면동명": "지정면"},
    {"행정동코드": "11110515", "시도명": "서울특별시",     "시군구명": "종로구", "읍면동명": "청운효자동"},
])

N = 500

def make_raw_od_sample():
    hdongs = DONG_TABLE["행정동코드"].tolist()
    dates = pd.date_range("2023-09-01", "2023-10-15", freq="D")
    festival_days = {"2023-10-06", "2023-10-07", "2023-10-08", "2023-10-09",
                      "2023-10-13", "2023-10-14", "2023-10-15"}

    rows = []
    for _ in range(N):
        d = pd.Timestamp(RNG.choice(dates))
        date_str = d.strftime("%Y-%m-%d")
        start_h = RNG.integers(6, 23)
        end_h = min(start_h + RNG.integers(0, 2), 23)
        row = {
            "origin_hdong_cd": RNG.choice(hdongs),
            "dest_hdong_cd": RNG.choice(hdongs),
            "date": date_str,
            "start_time": f"{start_h:02d}:00",
            "end_time": f"{end_h:02d}:00",
            "gender": RNG.choice(["M", "F"]),
            "age": RNG.choice([10, 20, 30, 40, 50, 60, 70]),
            "modal": RNG.choice([0, 1, 2, 3, 5, 7], p=[0.45, 0.2, 0.05, 0.15, 0.1, 0.05]),
            "origin_purpose": RNG.choice([0, 1, 2, 3, 5], p=[0.35, 0.15, 0.1, 0.2, 0.2]),
            "dest_purpose": RNG.choice([0, 1, 2, 3, 5], p=[0.35, 0.15, 0.1, 0.2, 0.2]),
            "od_dist_avg": round(RNG.uniform(0.5, 30), 2),
            "od_duration_avg": round(RNG.uniform(3, 90), 1),
            "od_cnts": int(RNG.integers(1, 50) * (3 if date_str in festival_days else 1)),
        }
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    raw = make_raw_od_sample()
    raw.to_csv("od_20230901_sample.csv", index=False, encoding="utf-8-sig")
    DONG_TABLE.to_csv("dong_code_sample.csv", index=False, encoding="utf-8-sig")
    print("Sample generation done:")
    print(" - od_20230901_sample.csv:", raw.shape)
    print(" - dong_code_sample.csv:", DONG_TABLE.shape)
