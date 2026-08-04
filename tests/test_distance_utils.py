import numpy as np
from distance_utils import haversine_distance_matrix, compute_distance_matrix


def test_haversine_matrix_shape():
    lat1, lon1 = np.array([37.0, 37.1]), np.array([127.0, 127.1])
    lat2, lon2 = np.array([37.0, 37.05, 37.1]), np.array([127.0, 127.05, 127.1])
    d = haversine_distance_matrix(lat1, lon1, lat2, lon2)
    assert d.shape == (2, 3)


def test_haversine_known_distance_seoul_busan():
    # Seoul City Hall to Busan City Hall is roughly 325km (allow 5% tolerance)
    lat1, lon1 = np.array([37.5665]), np.array([126.9780])
    lat2, lon2 = np.array([35.1796]), np.array([129.0756])
    d = haversine_distance_matrix(lat1, lon1, lat2, lon2)[0, 0]
    assert 300_000 < d < 350_000


def test_zero_distance_same_point():
    lat, lon = np.array([37.0]), np.array([127.0])
    d = haversine_distance_matrix(lat, lon, lat, lon)
    assert d[0, 0] == 0.0


def test_compute_distance_matrix_invalid_method():
    lat, lon = np.array([37.0]), np.array([127.0])
    try:
        compute_distance_matrix(lat, lon, lat, lon, method="invalid")
        assert False, "an exception should have been raised"
    except ValueError:
        pass
