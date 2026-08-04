from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    pass


def load_csv_robust(path: str) -> pd.DataFrame:
    last_err: Exception | None = None
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as e:
            last_err = e
            continue
    raise DataValidationError(f"Could not detect encoding for {path}: {last_err}")


def validate_columns(df: pd.DataFrame, required_cols: list[str], df_name: str) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise DataValidationError(f"[{df_name}] missing required columns: {missing} (found: {list(df.columns)})")


def validate_unique_ids(df: pd.DataFrame, id_col: str, df_name: str) -> None:
    dup = df[df.duplicated(subset=[id_col], keep=False)]
    if not dup.empty:
        raise DataValidationError(
            f"[{df_name}] duplicate {id_col} found ({dup[id_col].nunique()}): "
            f"{dup[id_col].unique().tolist()[:10]}"
        )


def validate_coordinates(df: pd.DataFrame, lat_col: str, lon_col: str, df_name: str) -> pd.Series:
    lat_invalid = df[lat_col].isna() | ~df[lat_col].between(-90, 90)
    lon_invalid = df[lon_col].isna() | ~df[lon_col].between(-180, 180)
    invalid = lat_invalid | lon_invalid
    n_invalid = int(invalid.sum())
    if n_invalid:
        logger.warning("[%s] %d rows with invalid coordinates (marked INVALID_COORDINATE)", df_name, n_invalid)
    return invalid


def validate_no_negative(df: pd.DataFrame, col: str, df_name: str) -> None:
    if (df[col] < 0).any():
        n = int((df[col] < 0).sum())
        raise DataValidationError(f"[{df_name}] {col} has {n} negative values")


def validate_no_missing(df: pd.DataFrame, cols: list[str], df_name: str) -> None:
    for c in cols:
        n_na = int(df[c].isna().sum())
        if n_na:
            raise DataValidationError(f"[{df_name}] {c} has {n_na} missing values")
