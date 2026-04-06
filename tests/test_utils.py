import os
from datetime import datetime

import cftime
import numpy as np
import pytest
import xarray as xr

import xarray_subset_grid.utils as xsg_utils

# normalize_polygon_x_coords tests.


def get_test_file_dir():
    """
    returns the test file dir path
    """
    test_file_dir = os.path.join(os.path.dirname(__file__), "example_data")
    return test_file_dir


poly1_180 = np.array(
    [
        [-73, 41],
        [-70, 41],
        [-73, 39],
        [-73, 41],
    ]
)
poly1_360 = np.array(
    [
        [287, 41],
        [290, 41],
        [287, 39],
        [287, 41],
    ]
)

poly2_360 = np.array(
    [
        [234, 41],
        [234, 41],
        [250, 39],
        [290, 41],
    ]
)

poly2_180 = np.array(
    [
        [-126, 41],
        [-126, 41],
        [-110, 39],
        [-70, 41],
    ]
)


@pytest.mark.parametrize(
    "lons, poly, norm_poly",
    [
        ([-85, -84, -83, 10], poly1_180, poly1_180),  # x1
        ([60, 45, 85, 70], poly1_180, poly1_180),  # x2
        ([190, 200, 220, 250, 260], poly1_180, poly1_360),  # x3
        ([-85, -84, -83, 10], poly2_360, poly2_180),  # x1
        ([60, 45, 85, 70], poly2_360, poly2_360),  # x2
        ([190, 200, 220, 250, 260], poly2_360, poly2_360),  # x3
    ],
)
def test_normalize_x_coords(lons, poly, norm_poly):
    lons = np.array(lons)
    normalized_polygon = xsg_utils.normalize_polygon_x_coords(lons, np.array(poly))
    assert np.allclose(normalized_polygon, norm_poly)


bbox1_180 = [-73, 39, -70, 41]
bbox1_360 = [287, 39, 290, 41]
bbox2_360 = [234, 39, 290, 41]
bbox2_180 = [-126, 39, -70, 41]


@pytest.mark.parametrize(
    "lons, bbox, norm_bbox",
    [
        ([-85, -84, -83, 10], bbox1_180, bbox1_180),  # x1
        ([60, 45, 85, 70], bbox1_180, bbox1_180),  # x2
        ([190, 200, 220, 250, 260], bbox1_180, bbox1_360),  # x3
        ([-85, -84, -83, 10], bbox2_360, bbox2_180),  # x1
        ([60, 45, 85, 70], bbox2_360, bbox2_360),  # x2
        ([190, 200, 220, 250, 260], bbox2_360, bbox2_360),  # x3
    ],
)
def test_normalize_x_coords_bbox(lons, bbox, norm_bbox):
    lons = np.array(lons)
    normalized_polygon = xsg_utils.normalize_bbox_x_coords(lons, bbox)
    assert np.allclose(normalized_polygon, norm_bbox)


def test_ray_tracing_numpy():
    """
    minimal test, but at least it'll show it's not totally broken

    NOTE: this function was compared to shapely and a Cython implementation
    """
    poly = [
        (3.0, 3.0),
        (5.0, 8.0),
        (10.0, 5.0),
        (7.0, 1.0),
    ]

    points = np.array(
        [
            (3.0, 6.0),  # outside
            (6.0, 4.0),  # inside
            (9.0, 7.0),  # outside
        ]
    )

    result = xsg_utils.ray_tracing_numpy(points[:, 0], points[:, 1], poly)

    assert np.array_equal(result, [False, True, False])


@pytest.mark.parametrize(
    "num, unit",
    [
        (512, "bytes"),
        (2048, "KB"),
        (3 * 1024**2, "MB"),
    ],
)
def test_format_bytes(num, unit):
    assert unit in xsg_utils.format_bytes(num)


def test_asdatetime_none():
    assert xsg_utils.asdatetime(None) is None


def test_asdatetime_datetime_passthrough():
    dt = datetime(2020, 6, 15, 12, 30, 0)
    assert xsg_utils.asdatetime(dt) is dt


def test_asdatetime_cftime_passthrough():
    dt = cftime.datetime(2020, 6, 15, 12)
    assert xsg_utils.asdatetime(dt) is dt


def test_asdatetime_parse_string():
    dt = xsg_utils.asdatetime("2020-06-15T12:30:00")
    assert dt.year == 2020 and dt.month == 6 and dt.day == 15


def test_compute_2d_subset_mask_all_inside():
    ny, nx = 5, 5
    lat = np.linspace(40.0, 44.0, ny)
    lon = np.linspace(-74.0, -70.0, nx)
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
    lat_da = xr.DataArray(lat2d, dims=("y", "x"))
    lon_da = xr.DataArray(lon2d, dims=("y", "x"))
    poly = np.array([(-75.0, 39.0), (-69.0, 39.0), (-69.0, 45.0), (-75.0, 45.0)])
    mask = xsg_utils.compute_2d_subset_mask(lat_da, lon_da, poly)
    assert mask.dims == ("y", "x")
    assert mask.all()


def test_compute_2d_subset_mask_partial():
    # Include explicit lon/lat nodes inside the polygon so the mask can be checked at a
    # non-boundary grid point (ray-casting is ambiguous on polygon edges).
    lat = np.array([40.0, 40.5, 41.0, 43.0, 46.0])
    lon = np.array([-74.5, -73.75, -73.0, -71.0, -68.0])
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
    lat_da = xr.DataArray(lat2d, dims=("y", "x"))
    lon_da = xr.DataArray(lon2d, dims=("y", "x"))
    poly = np.array([(-74.5, 40.0), (-73.0, 40.0), (-73.0, 41.0), (-74.5, 41.0)])
    mask = xsg_utils.compute_2d_subset_mask(lat_da, lon_da, poly)
    assert mask.dims == ("y", "x")
    assert mask.any()
    assert not mask.all()
    i_inside = int(np.where(lat == 40.5)[0][0])
    j_inside = int(np.where(lon == -73.75)[0][0])
    assert mask.values[i_inside, j_inside]
    assert not mask.values[-1, -1]


def test_compute_2d_subset_mask_list_polygon_coerced():
    """list/tuple polygon vertices are accepted (coerced via normalize_polygon_x_coords)."""
    ny, nx = 5, 5
    lat = np.linspace(40.0, 44.0, ny)
    lon = np.linspace(-74.0, -70.0, nx)
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
    lat_da = xr.DataArray(lat2d, dims=("y", "x"))
    lon_da = xr.DataArray(lon2d, dims=("y", "x"))
    poly = [(-75.0, 39.0), (-69.0, 39.0), (-69.0, 45.0), (-75.0, 45.0)]
    mask = xsg_utils.compute_2d_subset_mask(lat_da, lon_da, poly)
    assert mask.all()
