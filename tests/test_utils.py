import numpy as np
import pytest

from xarray_subset_grid.utils import normalize_polygon_x_coords

poly1_180 = np.array([
    [-73, 41],
    [-70, 41],
    [-73, 39],
    [-73, 41],
])
poly1_360 = np.array([
    [287, 41],
    [290, 41],
    [287, 39],
    [287, 41],
])

poly2_360 = np.array([
    [234, 41],
    [234, 41],
    [250, 39],
    [290, 41],
])

poly2_180 = np.array([
    [-126, 41],
    [-126, 41],
    [-110, 39],
    [-70, 41],
])
@pytest.mark.parametrize("lons, poly, norm_poly",
                         [
                          ([-85, -84, -83, 10], poly1_180, poly1_180), # x1
                          ([60, 45, 85, 70], poly1_180, poly1_180),  # x2
                          ([190, 200, 220, 250, 260], poly1_180, poly1_360),  # x3
                          ([-85, -84, -83, 10], poly2_360, poly2_180), # x1
                          ([60, 45, 85, 70], poly2_360, poly2_360),  # x2
                          ([190, 200, 220, 250, 260], poly2_360, poly2_360),  # x3
                          ]
                         )
def test_normalize_x_coords(lons, poly, norm_poly):
    lons = np.array(lons)
    normalized_polygon = normalize_polygon_x_coords(lons, np.array(poly))
    print(f"{lons=}")
    print(f"{poly=}")
    print(f"{norm_poly=}")
    print(f"{normalized_polygon=}")

    assert np.allclose(normalized_polygon, norm_poly)

    # normalized_polygon = normalize_polygon_x_coords(x2, polygon)
    # assert np.allclose(
    #     normalized_polygon,
    #     np.array(
    #         [
    #             [-126, 41],
    #             [-126, 41],
    #             [-110, 39],
    #             [-70, 41],
    #         ]
    #     ),
    # )

    # normalized_polygon = normalize_polygon_x_coords(x3, polygon)
    # assert np.allclose(
    #     normalized_polygon,
    #     np.array(
    #         [
    #             [234, 41],
    #             [234, 41],
    #             [250, 39],
    #             [290, 41],
    #         ]
    #     ),
    # )
