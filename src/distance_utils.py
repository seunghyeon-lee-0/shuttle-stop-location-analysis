from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

EARTH_RADIUS_M = 6_371_000.0


def haversine_distance_matrix(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    lat1_r = np.radians(lat1)[:, None]
    lon1_r = np.radians(lon1)[:, None]
    lat2_r = np.radians(lat2)[None, :]
    lon2_r = np.radians(lon2)[None, :]

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_M * c


def euclidean_distance_matrix(
    x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray
) -> np.ndarray:
    dx = x1[:, None] - x2[None, :]
    dy = y1[:, None] - y2[None, :]
    return np.sqrt(dx ** 2 + dy ** 2)


def compute_distance_matrix(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
    method: str = "haversine",
) -> np.ndarray:
    if method == "haversine":
        return haversine_distance_matrix(lat1, lon1, lat2, lon2)
    if method == "euclidean":
        return euclidean_distance_matrix(lat1, lon1, lat2, lon2)
    raise ValueError(f"Unknown distance_method: {method} (supported: haversine, euclidean)")
