import numpy as np

from xarray_subset_grid.utils import normalize_polygon_x_coords


def test_normalize_x_coords():
    polygon = np.array(
        [
            [-73, 41],
            [-70, 41],
            [-73, 39],
            [-73, 41],
        ]
    )

    x1 = np.array([-85, -84, -83, 10])
    normalized_polygon = normalize_polygon_x_coords(x1, polygon)
    assert np.allclose(
        normalized_polygon,
        np.array(
            [
                [-73, 41],
                [-70, 41],
                [-73, 39],
                [-73, 41],
            ]
        ),
    )

    x2 = np.array([60, 45, 85, 70])
    normalized_polygon = normalize_polygon_x_coords(x2, polygon)
    assert np.allclose(
        normalized_polygon,
        np.array(
            [
                [-73, 41],
                [-70, 41],
                [-73, 39],
                [-73, 41],
            ]
        ),
    )

    x3 = np.array([190, 200, 220, 250, 260])
    normalized_polygon = normalize_polygon_x_coords(x3, polygon)
    assert np.allclose(
        normalized_polygon,
        np.array(
            [
                [287, 41],
                [290, 41],
                [287, 39],
                [287, 41],
            ]
        ),
    )

    polygon = np.array(
        [
            [234, 41],
            [234, 41],
            [250, 39],
            [290, 41],
        ]
    )

    normalized_polygon = normalize_polygon_x_coords(x1, polygon)
    assert np.allclose(
        normalized_polygon,
        np.array(
            [
                [-126, 41],
                [-126, 41],
                [-110, 39],
                [-70, 41],
            ]
        ),
    )

    normalized_polygon = normalize_polygon_x_coords(x2, polygon)
    assert np.allclose(
        normalized_polygon,
        np.array(
            [
                [-126, 41],
                [-126, 41],
                [-110, 39],
                [-70, 41],
            ]
        ),
    )

    normalized_polygon = normalize_polygon_x_coords(x3, polygon)
    assert np.allclose(
        normalized_polygon,
        np.array(
            [
                [234, 41],
                [234, 41],
                [250, 39],
                [290, 41],
            ]
        ),
    )
