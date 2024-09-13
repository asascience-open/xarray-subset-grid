import os

import fsspec
import numpy as np
import xarray as xr

import xarray_subset_grid.accessor  # noqa: F401
from tests.test_utils import get_test_file_dir

# open dataset as zarr object using fsspec reference file system and xarray


test_dir = get_test_file_dir()
sample_sgrid_file = os.path.join(test_dir, 'arakawa_c_test_grid.nc')

def test_polygon_subset():
    '''
    This is a basic integration test for the subsetting of a ROMS sgrid dataset using a polygon.
    '''
    fs = fsspec.filesystem(
        "reference",
        fo="s3://nextgen-dmac-cloud-ingest/nos/wcofs/nos.wcofs.2ds.best.nc.zarr",
        remote_protocol="s3",
        remote_options={"anon": True},
        target_protocol="s3",
        target_options={"anon": True},
    )
    m = fs.get_mapper("")

    ds = xr.open_dataset(
        m, engine="zarr", backend_kwargs=dict(consolidated=False), chunks={}
    )

    polygon = np.array(
        [
            [-122.38488806417945, 34.98888604471138],
            [-122.02425311530737, 33.300351211467074],
            [-120.60402628930146, 32.723214427630836],
            [-116.63789131284673, 32.54346959375448],
            [-116.39346090873218, 33.8541384965596],
            [-118.83845767505964, 35.257586401855164],
            [-121.34541503969862, 35.50073821008141],
            [-122.38488806417945, 34.98888604471138],
        ]
    )
    ds_temp = ds.xsg.subset_vars(['temp_sur'])
    ds_subset = ds_temp.xsg.subset_polygon(polygon)

    #Check that the subset dataset has the correct dimensions given the original padding
    assert ds_subset.sizes['eta_rho'] == ds_subset.sizes['eta_psi'] + 1
    assert ds_subset.sizes['eta_u'] == ds_subset.sizes['eta_psi'] + 1
    assert ds_subset.sizes['eta_v'] == ds_subset.sizes['eta_psi']
    assert ds_subset.sizes['xi_rho'] == ds_subset.sizes['xi_psi'] + 1
    assert ds_subset.sizes['xi_u'] == ds_subset.sizes['xi_psi']
    assert ds_subset.sizes['xi_v'] == ds_subset.sizes['xi_psi'] + 1

    #Check that the subset rho/psi/u/v positional relationsip makes sense aka psi point is
    #'between' it's neighbor rho points
    #Note that this needs to be better generalized; it's not trivial to write a test that
    #works in all potential cases.
    assert (ds_subset['lon_rho'][0,0] < ds_subset['lon_psi'][0,0]
            and ds_subset['lon_rho'][0,1] > ds_subset['lon_psi'][0,0])

    #ds_subset.temp_sur.isel(ocean_time=0).plot(x="lon_rho", y="lat_rho")

def test_polygon_subset_2():
    ds = xr.open_dataset(sample_sgrid_file, decode_times=False)
    polygon = np.array([
        [6.5, 37.5],
        [6.5, 39.5],
        [9.5, 40.5],
        [8.5, 37.5],
        [6.5, 37.5]
    ])
    ds_subset = ds.xsg.subset_polygon(polygon)

    #Check that the subset dataset has the correct dimensions given the original padding
    assert ds_subset.sizes['eta_rho'] == ds_subset.sizes['eta_psi'] + 1
    assert ds_subset.sizes['eta_u'] == ds_subset.sizes['eta_psi'] + 1
    assert ds_subset.sizes['eta_v'] == ds_subset.sizes['eta_psi']
    assert ds_subset.sizes['xi_rho'] == ds_subset.sizes['xi_psi'] + 1
    assert ds_subset.sizes['xi_u'] == ds_subset.sizes['xi_psi']
    assert ds_subset.sizes['xi_v'] == ds_subset.sizes['xi_psi'] + 1

    assert ds_subset.lon_psi.min() <= 6.5 and ds_subset.lon_psi.max() >= 9.5
    assert ds_subset.lat_psi.min() <= 37.5 and ds_subset.lat_psi.max() >= 40.5
