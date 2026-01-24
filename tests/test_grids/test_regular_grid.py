"""
Tests for rectangular grid code.
"""

from pathlib import Path

try:
    import fsspec
except ImportError:
    fsspec = None
import xarray as xr

from xarray_subset_grid.grids.regular_grid import RegularGrid

EXAMPLE_DATA = Path(__file__).parent.parent / "example_data"

TEST_FILE1 = EXAMPLE_DATA / "AMSEAS-subset.nc"

# NGOFS2_RGRID.nc is a small subset of the regridded NGOFS2 model.

# It was created by the "OFS subsetter"


def test_recognise():
    """
    works for at least one file ...
    """
    ds = xr.open_dataset(TEST_FILE1)

    assert RegularGrid.recognize(ds)


def test_recognise_not():
    """
    should not recognise an SGrid
    """
    ds = xr.open_dataset(EXAMPLE_DATA / "arakawa_c_test_grid.nc")

    assert not RegularGrid.recognize(ds)


#######
# These from the ugrid tests -- need to be adapted
#######

def test_grid_vars():
    """
    Check if the grid vars are defined properly
    """
    ds = xr.open_dataset(EXAMPLE_DATA / "AMSEAS-subset.nc")

    grid_vars = ds.xsg.grid_vars

    # ['mesh', 'nv', 'lon', 'lat', 'lonc', 'latc']
    assert grid_vars == {'lat', 'lon'}


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
        'water_w',
        'salinity',
        'surf_roughness',
        'surf_temp_flux',
        'water_v',
        # 'time_offset',
        'water_temp',
        'water_baro_v',
        'surf_atm_press',
        'surf_el',
        'surf_salt_flux',
        'water_u',
        'surf_wnd_stress_gridy',
        'water_baro_u',
        'watdep',
        'surf_solar_flux',
        # 'time1_run',
        'surf_wnd_stress_gridx',
        # 'time1_offset'
    }

def test_extra_vars():
    """
    Check if the extra vars are defined properly

    """
    ds = xr.open_dataset(EXAMPLE_DATA / "AMSEAS-subset.nc")

    extra_vars = ds.xsg.extra_vars

    # the extra "time" variables are not using the grid
    # so they should be listed as extra_vars
    assert extra_vars == {
        'time_offset',
        'time1_run',
        'time1_offset'
    }


# def test_coords():
#     ds = xr.open_dataset(EXAMPLE_DATA / "SFBOFS_subset1.nc")
#     ds = ugrid.assign_ugrid_topology(ds, **grid_topology)

#     coords = ds.xsg.coords

#     print(f"{coords=}")
#     print(f"{ds.coords=}")

#     assert set(coords) == set(
#         [
#             "lon",
#             "lat",
#             "lonc",
#             "latc",
#             "time",
#             "siglay",
#             "siglev",
#         ]
#     )


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
