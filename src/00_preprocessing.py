import glob
import os

import numpy as np
import pandas as pd

RAW_DIR = "../data"
OUT_DIR = "../results"
os.makedirs(OUT_DIR, exist_ok=True)

ANALYSIS_START = "2023-09-01"
ANALYSIS_END = "2023-10-15"
CHUSEOK_EXCLUDE = {"2023-09-28", "2023-09-29", "2023-09-30"}

MODAL_NAME_MAP = {
    0: "modal_차량", 1: "modal_시내버스", 2: "modal_지하철",
    3: "modal_도보", 5: "modal_철도", 7: "modal_항공기",
}
PURPOSE_NAME_MAP = {0: "귀가", 3: "쇼핑여가", 5: "여행"}
EXCLUDED_PURPOSE_CODES = {1, 2}

MAX_SPEED_KMH = {
    0: 120,
    1: 80,
    2: 90,
    3: 10,
    5: 150,
    7: 900,
}


def load_raw_od(raw_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(raw_dir, "od_*.csv")))
    if not paths:
        raise FileNotFoundError(
            f"No od_*.csv files found in {raw_dir}. Run data/make_sample_data.py first if no real data is available."
        )

    dfs = []
    for p in paths:
        df = None
        for enc in ("cp949", "utf-8-sig"):
            try:
                candidate = pd.read_csv(p, encoding=enc)
            except UnicodeDecodeError:
                continue
            if "origin_hdong_cd" in candidate.columns:
                df = candidate
                break
        if df is None:
            raise ValueError(f"Failed to read {p} with cp949/utf-8-sig.")
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def load_dong_code(raw_dir: str) -> pd.DataFrame:
    xlsx_path = os.path.join(raw_dir, "KIKcd_H.20240801.xlsx")
    if os.path.exists(xlsx_path):
        dong_cd = pd.read_excel(xlsx_path)
    else:
        csv_path = os.path.join(raw_dir, "dong_code_sample.csv")
        dong_cd = pd.read_csv(csv_path, dtype={"행정동코드": str})
    return dong_cd[["행정동코드", "시도명", "시군구명", "읍면동명"]]


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["modal", "origin_purpose"]).copy()
    df["origin_purpose"] = df["origin_purpose"].astype(int)
    df["dest_purpose"] = df["dest_purpose"].astype(int)
    df["modal"] = df["modal"].astype(int)
    df["origin_hdong_cd"] = df["origin_hdong_cd"].astype(str)
    df["dest_hdong_cd"] = df["dest_hdong_cd"].astype(str)
    df["start_time"] = df["start_time"].astype(str).str.split(":").str[0].astype(int)
    df["end_time"] = df["end_time"].astype(str).str.split(":").str[0].astype(int)
    return df


def filter_purpose(df: pd.DataFrame) -> pd.DataFrame:
    home_to_home = (df["origin_purpose"] == 0) & (df["dest_purpose"] == 0)
    commute_or_school = (
        df["origin_purpose"].isin(EXCLUDED_PURPOSE_CODES)
        | df["dest_purpose"].isin(EXCLUDED_PURPOSE_CODES)
    )
    return df[~(home_to_home | commute_or_school)].copy()


def merge_dong_code(df: pd.DataFrame, dong_cd: pd.DataFrame) -> pd.DataFrame:
    dong_cd = dong_cd.copy()
    dong_cd["행정동코드"] = dong_cd["행정동코드"].astype(str)

    df = df.merge(
        dong_cd.rename(columns={
            "시도명": "출발지_시도명", "시군구명": "출발지_시군구명", "읍면동명": "출발지_읍면동명",
        }),
        left_on="origin_hdong_cd", right_on="행정동코드", how="left",
    ).drop(columns=["행정동코드"])

    df = df.merge(
        dong_cd.rename(columns={
            "시도명": "도착지_시도명", "시군구명": "도착지_시군구명", "읍면동명": "도착지_읍면동명",
        }),
        left_on="dest_hdong_cd", right_on="행정동코드", how="left",
    ).drop(columns=["행정동코드"])

    return df.drop(columns=["origin_hdong_cd", "dest_hdong_cd"])


def add_speed_and_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    hours = df["od_duration_avg"] / 60.0
    df["speed_kmh"] = np.where(hours > 0, df["od_dist_avg"] / hours, np.nan)

    df["요일"] = df["date"].dt.day_name()
    df["weekend"] = (df["date"].dt.dayofweek >= 5).astype(int)
    df["외부인"] = (df["출발지_시군구명"] != df["도착지_시군구명"]).astype(int)
    return df


def remove_speed_outliers(df: pd.DataFrame) -> pd.DataFrame:
    max_allowed = df["modal"].map(MAX_SPEED_KMH)
    keep = df["speed_kmh"].isna() | (df["speed_kmh"] <= max_allowed)
    return df[keep].copy()


def exclude_chuseok_and_out_of_range(df: pd.DataFrame) -> pd.DataFrame:
    date_str = df["date"].dt.strftime("%Y-%m-%d")
    in_range = (df["date"] >= ANALYSIS_START) & (df["date"] <= ANALYSIS_END)
    not_chuseok = ~date_str.isin(CHUSEOK_EXCLUDE)
    return df[in_range & not_chuseok].copy()


def filter_wonju(df: pd.DataFrame) -> pd.DataFrame:
    origin_wonju = (df["출발지_시도명"] == "강원특별자치도") & (df["출발지_시군구명"] == "원주시")
    dest_wonju = (df["도착지_시도명"] == "강원특별자치도") & (df["도착지_시군구명"] == "원주시")
    return df[origin_wonju | dest_wonju].copy()


def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    df = pd.get_dummies(df, columns=["modal", "origin_purpose", "dest_purpose"],
                         prefix=["modal", "origin_purpose", "dest_purpose"])

    rename_map = {f"modal_{k}": v for k, v in MODAL_NAME_MAP.items()}
    rename_map.update({f"origin_purpose_{k}": f"origin_purpose_{v}" for k, v in PURPOSE_NAME_MAP.items()})
    rename_map.update({f"dest_purpose_{k}": f"dest_purpose_{v}" for k, v in PURPOSE_NAME_MAP.items()})
    df = df.rename(columns=rename_map)
    return df


def run_pipeline(raw_dir: str = RAW_DIR, out_dir: str = OUT_DIR) -> pd.DataFrame:
    df = load_raw_od(raw_dir)
    dong_cd = load_dong_code(raw_dir)

    df = basic_cleaning(df)
    df = filter_purpose(df)
    df = merge_dong_code(df, dong_cd)
    df = add_speed_and_date_features(df)
    df = remove_speed_outliers(df)
    df = exclude_chuseok_and_out_of_range(df)
    df = filter_wonju(df)

    df = df.sort_values(by=["출발지_시도명", "출발지_시군구명", "출발지_읍면동명"])

    out_path = os.path.join(out_dir, "preprocessed_wonju_od.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Preprocessing done: {df.shape[0]} rows -> {out_path}")
    return df


if __name__ == "__main__":
    run_pipeline()
