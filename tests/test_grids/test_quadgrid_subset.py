"""
Tests that subsetting irregular (curvilinear) quad grids keeps the result
logically rectangular, and that it matches an index-slice reference.

Reference: find the (j, i) nodes inside the bounding box, take the min/max
of each index, and slice with isel. The result of the subset should be
exactly that slice, with data outside the box masked.
"""

import numpy as np
import pytest
import xarray as xr

from xarray_subset_grid.grids.node_grids import QuadGrid

NY, NX = 30, 40


def make_quad_dataset(lon2d, lat2d):
    """Small synthetic curvilinear dataset with 2D lat/lon and a unique value per node."""
    ny, nx = lon2d.shape
    temp = np.arange(ny * nx, dtype="float64").reshape(ny, nx)
    ds = xr.Dataset(
        data_vars={
            "temp": (("y", "x"), temp),
            "salt": (("time", "y", "x"), np.stack([temp, temp + 0.5])),
        },
        coords={
            "lon": (("y", "x"), lon2d),
            "lat": (("y", "x"), lat2d),
            "time": [0, 1],
        },
    )
    ds.lat.attrs = {"standard_name": "latitude", "units": "degrees_north"}
    ds.lon.attrs = {"standard_name": "longitude", "units": "degrees_east"}
    ds.temp.attrs = {"standard_name": "sea_water_temperature"}
    ds.salt.attrs = {"standard_name": "sea_water_salinity"}
    return ds


def rotated_grid(angle_deg=35.0):
    """A regular grid rotated so it is not north-east aligned."""
    x, y = np.meshgrid(np.linspace(-10, 10, NX), np.linspace(-6, 6, NY))
    a = np.deg2rad(angle_deg)
    lon = x * np.cos(a) - y * np.sin(a)
    lat = 40.0 + x * np.sin(a) + y * np.cos(a)
    return make_quad_dataset(lon, lat)


def curved_grid():
    """A strongly bent grid: rows follow an arc, spacing varies along the arc."""
    r = np.linspace(8, 14, NY)[:, None]
    theta = np.deg2rad(np.linspace(-80, 80, NX) ** 3 / 80**2)[None, :]
    lon = r * np.sin(theta)
    lat = 30.0 + r * np.cos(theta)
    return make_quad_dataset(lon, lat)


GRIDS = {"rotated": rotated_grid, "curved": curved_grid}

# (name, grid, bbox as xmin, ymin, xmax, ymax)
BBOXES = [
    ("rotated-center", "rotated", (-4, 37, 4, 43)),
    ("rotated-corner", "rotated", (3, 42, 9, 48)),
    ("curved-center", "curved", (-3, 36, 3, 41)),
    ("curved-edge", "curved", (4, 33, 10, 39)),
]


def reference_slices(ds, bbox):
    """Independent reference: index slices spanning all nodes inside the bbox."""
    xmin, ymin, xmax, ymax = bbox
    lon = ds.lon.values
    lat = ds.lat.values
    inside = (lon >= xmin) & (lon <= xmax) & (lat >= ymin) & (lat <= ymax)
    jj, ii = np.nonzero(inside)
    assert jj.size > 0, "test bbox must contain some nodes"
    return inside, slice(jj.min(), jj.max() + 1), slice(ii.min(), ii.max() + 1)


def check_against_reference(ds, result, bbox):
    inside, jslice, islice = reference_slices(ds, bbox)
    ref = ds.isel(y=jslice, x=islice)
    ref_inside = inside[jslice, islice]

    # still logically rectangular: 2D coords, same dims and shape as the data
    assert result.lat.dims == ("y", "x")
    assert result.lon.dims == ("y", "x")
    assert result.lat.shape == result.lon.shape == ref.lat.shape
    for name in ("temp", "salt"):
        assert result[name].dims[-2:] == ("y", "x")
        assert result[name].shape[-2:] == result.lat.shape
    assert result.sizes["y"] == jslice.stop - jslice.start
    assert result.sizes["x"] == islice.stop - islice.start

    # the coordinates are exactly the index-slice reference
    np.testing.assert_array_equal(result.lat.values, ref.lat.values)
    np.testing.assert_array_equal(result.lon.values, ref.lon.values)

    # data is kept inside the bbox, and is NaN only outside it
    for name in ("temp", "salt"):
        got = result[name].values
        want = ref[name].values
        mask = np.broadcast_to(ref_inside, got.shape)
        np.testing.assert_array_equal(got[mask], want[mask])
        assert np.isnan(got[~mask]).all()
        assert not np.isnan(got[mask]).any()


@pytest.mark.parametrize("gridname", GRIDS)
def test_recognized_as_quadgrid(gridname):
    assert QuadGrid.recognize(GRIDS[gridname]())


@pytest.mark.parametrize(("case", "gridname", "bbox"), BBOXES, ids=[b[0] for b in BBOXES])
def test_bbox_subset_matches_index_slice(case, gridname, bbox):
    ds = GRIDS[gridname]()
    result = ds.xsg.subset_bbox(bbox)
    check_against_reference(ds, result, bbox)


@pytest.mark.parametrize(("case", "gridname", "bbox"), BBOXES, ids=[b[0] for b in BBOXES])
def test_polygon_subset_matches_index_slice(case, gridname, bbox):
    """A polygon is subset by its bounding box on these grids; use a triangle."""
    ds = GRIDS[gridname]()
    xmin, ymin, xmax, ymax = bbox
    poly = [(xmin, ymin), (xmax, ymin), (xmax, ymax)]
    result = ds.xsg.subset_polygon(poly)
    check_against_reference(ds, result, bbox)


def test_rotated_grid_result_has_masked_corners():
    """A rotated grid cannot be cut exactly by a box: the slice has NaN corners."""
    ds = rotated_grid()
    result = ds.xsg.subset_bbox((-4, 37, 4, 43))
    assert np.isnan(result.temp.values).any()
    assert not np.isnan(result.temp.values).all()
    # the dropped rows/columns are entirely outside the box
    assert result.sizes["y"] < ds.sizes["y"]
    assert result.sizes["x"] < ds.sizes["x"]


def test_bbox_covering_whole_grid_is_unchanged():
    ds = curved_grid()
    result = ds.xsg.subset_bbox((-100, -100, 100, 100))
    assert result.temp.shape == ds.temp.shape
    np.testing.assert_array_equal(result.temp.values, ds.temp.values)


def test_bbox_with_no_nodes():
    """Nothing inside the box: should give an empty result, not an error with holes."""
    ds = rotated_grid()
    result = ds.xsg.subset_bbox((100, 0, 101, 1))
    assert result.sizes["y"] == 0 or result.sizes["x"] == 0


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Dataset.where(drop=True) drops every all-False row/column, including "
        "interior ones, so nodes that were not neighbors become adjacent. The "
        "result is not a contiguous index slice (see ioos/xarray-subset-grid#185)."
    ),
)
def test_bbox_two_arms_of_a_bent_grid_stays_rectangular():
    """
    A U-shaped grid cut by a box that catches both arms but not the bend:
    the nodes inside are not connected, so the only logically rectangular
    result that keeps neighbors adjacent is the index box spanning both arms.
    """
    # rows fold back on themselves: x runs along the first half of columns,
    # then returns along the second half, offset in latitude
    nx = 40
    half = nx // 2
    s = np.linspace(0, 10, half)
    lon = np.concatenate([s, s[::-1]])
    base = np.concatenate([np.zeros(half), np.full(half, 4.0)])
    lon2d = np.tile(lon, (NY, 1))
    lat2d = 30.0 + np.linspace(0, 2, NY)[:, None] + base[None, :]
    ds = make_quad_dataset(lon2d, lat2d)

    bbox = (3, 30.5, 6, 37)
    result = ds.xsg.subset_bbox(bbox)
    check_against_reference(ds, result, bbox)
