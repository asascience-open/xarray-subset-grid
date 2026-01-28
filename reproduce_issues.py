import xarray as xr
import numpy as np
from xarray_subset_grid.grids.regular_grid import RegularGrid
import pytest

def create_synthetic_dataset(decreasing=False):
    """Create a synthetic dataset with regular grid."""
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

def test_data_vars_error():
    print("Testing data_vars error...")
    ds = create_synthetic_dataset()
    # Ensure it is recognized as a RegularGrid
    assert RegularGrid.recognize(ds)
    
    # Access xsg accessor
    try:
        data_vars = ds.xsg.data_vars
        print(f"data_vars: {data_vars}")
    except Exception as e:
        print(f"Caught expected error in data_vars: {e}")
        # import traceback
        # traceback.print_exc()

def test_decreasing_coords():
    print("\nTesting decreasing coordinates support...")
    ds = create_synthetic_dataset(decreasing=True)
    assert RegularGrid.recognize(ds)
    
    # bbox: (min_lon, min_lat, max_lon, max_lat)
    bbox = (-95, 35, -85, 45)
    
    try:
        subset = ds.xsg.subset_bbox(bbox)
        print(f"Subset size: {subset.sizes}")
        
        # Check if subset has data
        if subset.sizes["lat"] == 0 or subset.sizes["lon"] == 0:
            print("FAILURE: Subset has dimension size 0")
        else:
            print("SUCCESS: Subset has data")
            
    except Exception as e:
        print(f"Caught unexpected error in decreasing coords subsetting: {e}")

if __name__ == "__main__":
    test_data_vars_error()
    test_decreasing_coords()
