import time

import fsspec
import shapely
import thalassa
import xarray as xr
import xugrid as xu

bucket_name = 'noaa-gestofs-pds'
key = '_para2/stofs_2d_glo.20230819/stofs_2d_glo.t00z.fields.cwl.nc'
url = f"s3://{bucket_name}/{key}"

fs = fsspec.filesystem("s3", anon=True)

ds = xr.open_dataset(
    fs.open(url),
    chunks={},
    drop_variables=["neta", "nvel", "max_nvdll", "max_nvell"],
    engine='h5netcdf'
)

bbox = (-70, 40, -60, 50)

def info(ds):
    print(f'Data Variables: {ds.data_vars}')
    print(f'Coordinates: {ds.coords}')
    print(f'Dimensions: {ds.dims}')
    print(f'Attributes: {ds.attrs}\n')


print("Subsetting methods comparison")
start = time.time()
ads = ds.xsg.grid.subset_bbox(ds, bbox)
print(f"Xarray-subset-grid - {time.time-start} sec")

start = time.time()
tds = thalassa.normalize(ds)
tds = thalassa.crop(tds, shapely.box(*bbox))
print(f"Thalassa - {time.time-start} sec")

# Checking time only for subsetting operation
uds = xu.UgridDataset(ds)
start = time.time()
uds = uds.ugrid.sel(y=slice(bbox[2], bbox[3]), x=slice(bbox[0], bbox[1]))
print(f"UGrid - {time.time-start} sec")
