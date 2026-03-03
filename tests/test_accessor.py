"""
Tests for the GridDatasetAccessor (xsg accessor).

These tests verify that the accessor is correctly exposed on xr.Dataset
objects and that all its methods behave correctly for both recognized
and unrecognized grid datasets.
"""
import numpy as np
import pytest
import xarray as xr
from pathlib import Path

EXAMPLE_DATA = Path(__file__).parent / "example_data"


# ------------------------------------------------------------------ #
#  Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture
def rgrid_ds():
    """Load a real regular-grid NetCDF file for accessor tests."""
    path = EXAMPLE_DATA / "AMSEAS-subset.nc"
    return xr.open_dataset(path)


@pytest.fixture
def unknown_ds():
    """A minimal dataset that no grid implementation will recognize."""
    return xr.Dataset({"foo": (["x"], [1.0, 2.0, 3.0])})


# ------------------------------------------------------------------ #
#  Accessor existence tests
# ------------------------------------------------------------------ #

class TestAccessorPresence:
    def test_xsg_accessor_is_attached(self, rgrid_ds):
        """The xsg accessor must be available on any xr.Dataset."""
        assert hasattr(rgrid_ds, "xsg")

    def test_xsg_accessor_on_unknown_ds(self, unknown_ds):
        """The accessor should still be attached even if no grid is recognised."""
        assert hasattr(unknown_ds, "xsg")


# ------------------------------------------------------------------ #
#  Grid recognition tests
# ------------------------------------------------------------------ #

class TestGridRecognition:
    def test_known_grid_is_not_none(self, rgrid_ds):
        """A CF-compliant regular-grid dataset should be recognised."""
        assert rgrid_ds.xsg.grid is not None

    def test_unknown_grid_is_none(self, unknown_ds):
        """An unstructured / unknown dataset should return None for grid."""
        assert unknown_ds.xsg.grid is None


# ------------------------------------------------------------------ #
#  Property tests (including the data_vars bug regression)
# ------------------------------------------------------------------ #

class TestAccessorProperties:
    def test_grid_vars_returns_set(self, rgrid_ds):
        gvars = rgrid_ds.xsg.grid_vars
        assert isinstance(gvars, set)

    def test_data_vars_returns_set(self, rgrid_ds):
        dvars = rgrid_ds.xsg.data_vars
        assert isinstance(dvars, set)

    def test_data_vars_no_grid_does_not_raise(self, unknown_ds):
        """
        Regression test for the data_vars bug.
        Before the fix, this raised AttributeError because self._ds was
        checked instead of self._grid, causing None.data_vars() to be called.
        """
        result = unknown_ds.xsg.data_vars
        assert result == set()

    def test_grid_vars_no_grid_returns_empty(self, unknown_ds):
        assert unknown_ds.xsg.grid_vars == set()

    def test_extra_vars_no_grid_returns_empty(self, unknown_ds):
        assert unknown_ds.xsg.extra_vars == set()

    def test_has_vertical_levels_returns_bool(self, rgrid_ds):
        result = rgrid_ds.xsg.has_vertical_levels
        assert isinstance(result, bool)

    def test_has_vertical_levels_false_on_unknown(self, unknown_ds):
        assert unknown_ds.xsg.has_vertical_levels is False


# ------------------------------------------------------------------ #
#  Subsetting tests
# ------------------------------------------------------------------ #

class TestSubsetting:
    def test_subset_bbox_returns_dataset(self, rgrid_ds):
        """subset_bbox should return an xr.Dataset for a recognised grid."""
        # Build bbox dynamically from actual coordinate range
        lats = rgrid_ds.cf["latitude"].values
        lons = rgrid_ds.cf["longitude"].values
        lat_min, lat_max = float(lats.min()), float(lats.max())
        lon_min, lon_max = float(lons.min()), float(lons.max())
        # Use the centre quarter of the domain
        lat_mid = (lat_min + lat_max) / 2
        lon_mid = (lon_min + lon_max) / 2
        bbox = (lon_min, lat_min, lon_mid, lat_mid)
        ds_sub = rgrid_ds.xsg.subset_bbox(bbox)
        assert ds_sub is not None
        assert isinstance(ds_sub, xr.Dataset)

    def test_subset_polygon_returns_dataset(self, rgrid_ds):
        """subset_polygon should return an xr.Dataset for a recognised grid."""
        lats = rgrid_ds.cf["latitude"].values
        lons = rgrid_ds.cf["longitude"].values
        lat_min, lat_max = float(lats.min()), float(lats.max())
        lon_min, lon_max = float(lons.min()), float(lons.max())
        lat_mid = (lat_min + lat_max) / 2
        lon_mid = (lon_min + lon_max) / 2
        poly = np.array([
            [lon_min, lat_min],
            [lon_mid, lat_min],
            [lon_mid, lat_mid],
            [lon_min, lat_mid],
            [lon_min, lat_min],
        ])
        ds_sub = rgrid_ds.xsg.subset_polygon(poly)
        assert ds_sub is not None
        assert isinstance(ds_sub, xr.Dataset)

    def test_subset_bbox_no_grid_returns_none(self, unknown_ds):
        result = unknown_ds.xsg.subset_bbox((-80, 30, -70, 40))
        assert result is None

    def test_subset_polygon_no_grid_returns_none(self, unknown_ds):
        poly = np.array([[-80, 30], [-70, 30], [-70, 40], [-80, 40], [-80, 30]])
        result = unknown_ds.xsg.subset_polygon(poly)
        assert result is None

    def test_subset_vars_keeps_grid_vars(self, rgrid_ds):
        """Subsetting to a variable should always retain grid variables."""
        data_vars = list(rgrid_ds.xsg.data_vars)
        if not data_vars:
            pytest.skip("No data vars found in this dataset")
        ds_sub = rgrid_ds.xsg.subset_vars([data_vars[0]])
        grid_vars = rgrid_ds.xsg.grid_vars
        for gvar in grid_vars:
            assert gvar in ds_sub

    def test_subset_surface_level_no_vertical(self, unknown_ds):
        """Subsetting surface level on dataset with no verticals returns dataset unchanged."""
        result = unknown_ds.xsg.subset_surface_level(method="nearest")
        assert result is unknown_ds