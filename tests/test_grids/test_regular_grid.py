"""
Tests for rectangular grid code.
"""

from pathlib import Path

import numpy as np
import pytest

# only needed if you want to hit AWS servers.
# try:
#     import fsspec
# except ImportError:
#     fsspec = None
import xarray as xr

from tests.conftest import RGRID_FILES, SGRID_FILES, UGRID_FILES
from xarray_subset_grid.grids.regular_grid import RegularGrid

EXAMPLE_DATA = Path(__file__).parent.parent / "example_data"


# NGOFS2_RGRID.nc is a small subset of the regridded NGOFS2 model.

# It was created by the "OFS subsetter"


@pytest.mark.parametrize("test_file", RGRID_FILES)
def test_recognize(test_file):
    """
    works for at least one file ...
    """
    ds = xr.open_dataset(test_file)

    assert RegularGrid.recognize(ds)


@pytest.mark.parametrize("test_file", UGRID_FILES + SGRID_FILES)
def test_recognize_not(test_file):
    """
    should not recognize an SGrid
    """
    ds = xr.open_dataset(test_file)

    assert not RegularGrid.recognize(ds)


def create_synthetic_rectangular_grid_dataset(decreasing=False):
    """
    Create a synthetic dataset with regular grid.

    Can be either decreasing or increasing in latitude
    """

    lon = np.linspace(-100, -80, 21)
    if decreasing:
        lat = np.linspace(50, 30, 21)
    else:
        lat = np.linspace(30, 50, 21)

    data = np.random.rand(21, 21)

    ds = xr.Dataset(
        data_vars={
            "temp": (("lat", "lon"), data),
            "salt": (("lat", "lon"), data),
        },
        coords={
            "lat": lat,
            "lon": lon,
        },
    )
    # Add cf attributes
    ds.lat.attrs = {"standard_name": "latitude", "units": "degrees_north"}
    ds.lon.attrs = {"standard_name": "longitude", "units": "degrees_east"}
    ds.temp.attrs = {"standard_name": "sea_water_temperature"}

    return ds


def create_synthetic_global_rectangular_grid_dataset(*, use_360=True, decreasing_lon=False):
    """Create a synthetic global regular-grid dataset for longitude wrap tests."""
    lat = np.linspace(-10, 10, 21)
    if use_360:
        lon = np.arange(0, 360)
    else:
        lon = np.arange(-180, 180)

    if decreasing_lon:
        lon = lon[::-1]

    data = np.random.rand(lat.size, lon.size)

    ds = xr.Dataset(
        data_vars={
            "temp": (("lat", "lon"), data),
        },
        coords={
            "lat": lat,
            "lon": lon,
        },
    )

    ds.lat.attrs = {"standard_name": "latitude", "units": "degrees_north"}
    ds.lon.attrs = {"standard_name": "longitude", "units": "degrees_east"}
    ds.temp.attrs = {"standard_name": "sea_water_temperature"}

    return ds


def test_grid_vars():
    """
    Check if the grid vars are defined properly
    """
    ds = xr.open_dataset(EXAMPLE_DATA / "AMSEAS-subset.nc")

    grid_vars = ds.xsg.grid_vars

    # ['mesh', 'nv', 'lon', 'lat', 'lonc', 'latc']
    assert grid_vars == {"lat", "lon"}


def test_data_vars():
    """
    Check if the data vars are defined properly

    This is not currently working correctly!

    it finds extra stuff
    """
    ds = xr.open_dataset(EXAMPLE_DATA / "AMSEAS-subset.nc")

    data_vars = ds.xsg.data_vars

    # the extra "time" variables are not using the grid
    # so they should not be listed as data_vars
    assert data_vars == {
        "water_w",
        "salinity",
        "surf_roughness",
        "surf_temp_flux",
        "water_v",
        # 'time_offset',
        "water_temp",
        "water_baro_v",
        "surf_atm_press",
        "surf_el",
        "surf_salt_flux",
        "water_u",
        "surf_wnd_stress_gridy",
        "water_baro_u",
        "watdep",
        "surf_solar_flux",
        # 'time1_run',
        "surf_wnd_stress_gridx",
        # 'time1_offset'
    }


# might not be needed if tested elsewhere.
def test_data_vars2():
    """
    redundant with above, by already written ...
    """
    print("Testing data_vars error...")
    ds = create_synthetic_rectangular_grid_dataset()
    # Ensure it is recognized as a RegularGrid
    assert RegularGrid.recognize(ds)

    # Access xsg accessor
    data_vars = ds.xsg.data_vars
    print(f"data_vars: {data_vars}")

    assert data_vars == {"salt", "temp"}


def test_extra_vars():
    """
    Check if the extra vars are defined properly
    """
    ds = xr.open_dataset(EXAMPLE_DATA / "AMSEAS-subset.nc")

    extra_vars = ds.xsg.extra_vars

    # the extra "time" variables are not using the grid
    # so they should be listed as extra_vars
    assert extra_vars == {"time_offset", "time1_run", "time1_offset"}


def test_subset_to_bb():
    """
    Not a complete test by any means, but the basics are there.

    NOTE: it doesn't test if the variables got subset correctly ...

    """
    ds = xr.open_dataset(EXAMPLE_DATA / "2D-rectangular_grid_wind.nc")

    print(
        "initial bounds:",
        ds["lon"].data.min(),
        ds["lat"].data.min(),
        ds["lon"].data.max(),
        ds["lat"].data.max(),
    )

    bbox = (-0.5, 0, 0.5, 0.5)

    ds2 = ds.xsg.subset_bbox(bbox)

    assert ds2["lat"].size == 15
    assert ds2["lon"].size == 29

    new_bounds = (
        ds2["lon"].data.min(),
        ds2["lat"].data.min(),
        ds2["lon"].data.max(),
        ds2["lat"].data.max(),
    )
    print("new bounds:", new_bounds)
    assert new_bounds == bbox


def test_decreasing_latitude():
    """
    Some datasets have the latitude or longitude decreasing: 10, 9, 8 etc.
    e.g the NOAA GFS met model

    subsetting should still work

    """
    ds = xr.open_dataset(EXAMPLE_DATA / "rectangular_grid_decreasing.nc")

    print(
        "initial bounds:",
        ds["lon"].data.min(),
        ds["lat"].data.min(),
        ds["lon"].data.max(),
        ds["lat"].data.max(),
    )

    bbox = (-0.5, 0, 0.5, 0.5)

    ds2 = ds.xsg.subset_bbox(bbox)

    assert ds2["lat"].size == 15
    assert ds2["lon"].size == 29

    new_bounds = (
        ds2["lon"].data.min(),
        ds2["lat"].data.min(),
        ds2["lon"].data.max(),
        ds2["lat"].data.max(),
    )
    print("new bounds:", new_bounds)
    assert new_bounds == bbox


def test_decreasing_coords():
    """
    Redundant with above, but already written ...
    """
    print("\nTesting decreasing coordinates support...")
    ds = create_synthetic_rectangular_grid_dataset(decreasing=True)
    # assert RegularGrid.recognize(ds)

    # bbox: (min_lon, min_lat, max_lon, max_lat)
    bbox = (-95, 35, -85, 45)

    subset = ds.xsg.subset_bbox(bbox)
    print(f"Subset size: {subset.sizes}")

    # Check if subset has data
    assert subset.sizes["lat"] > 0
    assert subset.sizes["lon"] > 0


def test_subset_polygon():
    """
    Not a complete test by any means, but the basics are there.

    NOTE: it doesn't test if the variables got subset correctly ...

    """
    ds = xr.open_dataset(EXAMPLE_DATA / "2D-rectangular_grid_wind.nc")

    print(
        "initial bounds:",
        ds["lon"].data.min(),
        ds["lat"].data.min(),
        ds["lon"].data.max(),
        ds["lat"].data.max(),
    )

    poly = [(-0.5, 0.0), (0.0, 0.5), (0.5, 0.5), (0.5, 0.0), (0, 0.0)]
    # this poly has this bounding box:
    # bbox = (-0.5, 0, 0.5, 0.5)
    # so results should be the same as the bbox tests

    ds2 = ds.xsg.subset_polygon(poly)

    assert ds2["lat"].size == 15
    assert ds2["lon"].size == 29

    new_bounds = (
        ds2["lon"].data.min(),
        ds2["lat"].data.min(),
        ds2["lon"].data.max(),
        ds2["lat"].data.max(),
    )
    print("new bounds:", new_bounds)
    assert new_bounds == (-0.5, 0, 0.5, 0.5)


def test_subset_bbox_wrap_prime_meridian_on_360_grid():
    ds = create_synthetic_global_rectangular_grid_dataset(use_360=True)

    ds_subset = ds.xsg.subset_bbox((-10, -5, 10, 5))

    assert ds_subset["lat"].size == 11
    assert ds_subset["lon"].size == 21
    lon_values = ds_subset["lon"].values
    assert lon_values.min() == 0
    assert lon_values.max() == 359
    assert set(range(0, 11)).issubset(set(lon_values.tolist()))
    assert set(range(350, 360)).issubset(set(lon_values.tolist()))


def test_subset_bbox_wrap_dateline_on_180_grid():
    ds = create_synthetic_global_rectangular_grid_dataset(use_360=False)

    ds_subset = ds.xsg.subset_bbox((170, -5, -170, 5))

    assert ds_subset["lat"].size == 11
    assert ds_subset["lon"].size == 21
    lon_values = ds_subset["lon"].values
    assert lon_values.min() == -180
    assert lon_values.max() == 179
    assert set(range(170, 180)).issubset(set(lon_values.tolist()))
    assert set(range(-180, -169)).issubset(set(lon_values.tolist()))


def test_subset_bbox_wrap_prime_meridian_descending_lon():
    ds = create_synthetic_global_rectangular_grid_dataset(use_360=True, decreasing_lon=True)

    ds_subset = ds.xsg.subset_bbox((-10, -5, 10, 5))

    assert ds_subset["lat"].size == 11
    assert ds_subset["lon"].size == 21
    lon_values = ds_subset["lon"].values
    assert set(range(0, 11)).issubset(set(lon_values.tolist()))
    assert set(range(350, 360)).issubset(set(lon_values.tolist()))


def test_subset_bbox_raises_for_span_ge_half_earth():
    ds = create_synthetic_global_rectangular_grid_dataset(use_360=False)

    with pytest.raises(ValueError, match="less than half-way around the earth"):
        ds.xsg.subset_bbox((-170, -5, 170, 5))


# def test_vertical_levels():
#     ds = xr.open_dataset(EXAMPLE_DATA / "SFBOFS_subset1.nc")
#     ds = ugrid.assign_ugrid_topology(ds, **grid_topology)

#     assert ds.xsg.has_vertical_levels is True

#     ds_subset = ds.xsg.subset_vertical_level(0.0)
#     assert ds_subset["siglay"].dims == ("node",)
#     assert np.isclose(ds_subset["siglay"].isel(node=0).values, -0.025)

#     ds_surface = ds.xsg.subset_surface_level(method="nearest")
#     assert ds_surface["siglay"].dims == ("node",)
#     assert np.isclose(ds_surface["siglay"].isel(node=0).values, -0.025)

#     ds_bottom = ds.xsg.subset_bottom_level()
#     assert ds_bottom["siglay"].dims == ("node",)
#     assert np.isclose(ds_bottom["siglay"].isel(node=0).values, -0.975)

#     ds_top = ds.xsg.subset_top_level()
#     assert ds_top["siglay"].dims == ("node",)
#     assert np.isclose(ds_top["siglay"].isel(node=0).values, -0.025)

#     ds_subset2 = ds.xsg.subset_vertical_levels((0, 0.2), method="nearest")
#     assert ds_subset2["siglay"].dims == (
#         "siglay",
#         "node",
#     )

# @pytest.mark.online
# def test_3d_selector():
#     if fsspec is None:
#         raise ImportError("Must have fsspec installed to run --online tests")
#     bbox = (-70, 40, -60, 55)
#     name = "northeastUSA3d"

#     fs = fsspec.filesystem("s3", anon=True)
#     ds = xr.open_dataset(
#         fs.open("s3://noaa-nos-stofs3d-pds/STOFS-3D-Atl-shadow-VIMS/20240716/out2d_20240717.nc"),
#         chunks={},
#         engine="h5netcdf",
#         drop_variables=["nvel"],
#     )
#     ds = ugrid.assign_ugrid_topology(ds)

#     bbox_selector = ds.xsg.grid.compute_bbox_subset_selector(ds, bbox, name)

#     filepath = EXAMPLE_DATA / "northeastUSA3d_076e4d62.pkl"
#     selector_bytes = open(filepath, "rb").read()
#     loaded_selector = Selector(selector_bytes)

#     assert bbox_selector == loaded_selector


# @pytest.mark.online
# def test_2d_selector():
#     if fsspec is None:
#         raise ImportError("Must have fsspec installed to run --online tests")
#     bbox = (-70, 40, -60, 50)
#     name = "northeastUSA2d"

#     fs = fsspec.filesystem("s3", anon=True)
#     ds = xr.open_dataset(
#         fs.open("s3://noaa-gestofs-pds/stofs_2d_glo.20240807/stofs_2d_glo.t06z.fields.cwl.nc"),
#         chunks={},
#         drop_variables=["nvel"],
#     )
#     ds = ugrid.assign_ugrid_topology(ds)

#     bbox_selector = ds.xsg.grid.compute_bbox_subset_selector(ds, bbox, name)

#     filepath = EXAMPLE_DATA / "northeastUSA2d_bb3d126e.pkl"
#     selector_bytes = open(filepath, "rb").read()
#     loaded_selector = Selector(selector_bytes)

#     assert bbox_selector == loaded_selector
