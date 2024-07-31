import cf_xarray  # noqa
import numpy as np
import xarray as xr

from xarray_subset_grid.grids import ugrid
from xarray_subset_grid.utils import format_bytes
from xarray_subset_grid.visualization.mpl_plotting import plot_ugrid

import matplotlib.pyplot as plt

FILENAME = "example_data/SFBOFS_subset1.nc"

# polygon to subset to

polygon = np.array(
    [
        [-122.49008935972195, 37.85535293037749],
        [-122.46663678349302, 37.872023874021735],
        [-122.45807326471191, 37.87939868575954],
        [-122.48078332775191, 37.89508822711183],
        [-122.50107641768733, 37.90437518996049],
        [-122.51257315724197, 37.89737526062751],
        [-122.50081931329709, 37.88165893138216],
        [-122.51039590603393, 37.870731699265534],
        [-122.49008935972195, 37.85535293037749],
    ]
)

# the data are in 0--360 -- so converted for plotting
polygon360 = np.array(
    [
        [237.50991064, 37.85535293],
        [237.53336322, 37.87202387],
        [237.54192674, 37.87939869],
        [237.51921667, 37.89508823],
        [237.49892358, 37.90437519],
        [237.48742684, 37.89737526],
        [237.49918069, 37.88165893],
        [237.48960409, 37.8707317],
        [237.50991064, 37.85535293],
    ]
)

# Open the netcdf file as a dataset.
print("opening the dataset")
ds = xr.open_dataset(FILENAME)

# NOTE: This file is not UGRID compliant
#       we need to add UGRID specifications
ds = ugrid.assign_ugrid_topology(
    ds, face_node_connectivity="nv", face_face_connectivity="nbe"
)

# Plot the original grid
fig, ax = plt.subplots()
fig.suptitle("Original Grid w/ Polygon")
plot_ugrid(ax, ds)
ax.plot(polygon360[:, 0], polygon360[:, 1])

# now it should work.

bb = (
    ds["lon"].data.min(),
    ds["lat"].data.min(),
    ds["lon"].data.max(),
    ds["lat"].data.max(),
)

print(f"Bounding box is: {bb}")

print(f"Dataset size: {format_bytes(ds.nbytes)}")

print("about to subset the vars")

print("The data variables are:")
print(ds.xsg.data_vars)

ds = ds.xsg.subset_vars(["zeta", "u", "v"])

print("Now the data variables are:")
print(ds.xsg.data_vars)
print(f"Dataset size: {format_bytes(ds.nbytes)}")

# now to do the spatial subset:
ds_sub = ds.xsg.subset_polygon(polygon)

bb = (
    ds_sub["lon"].data.min(),
    ds_sub["lat"].data.min(),
    ds_sub["lon"].data.max(),
    ds_sub["lat"].data.max(),
)

print(f"Bounding box is: {bb}")
print(f"Dataset size: {format_bytes(ds_sub.nbytes)}")
print("Now the data variables are:")
print(ds_sub.xsg.data_vars)
print("And the grid variables are:")
print(ds_sub.xsg.grid_vars)

# save out the subset
ds_sub.to_netcdf("small_subset.nc")

# Plot the subset grid
fig2, ax2 = plt.subplots()
fig2.suptitle("Subset Grid")
plot_ugrid(ax2, ds_sub)

fig.show()
fig2.show()

input("Hit <enter> to quit")
