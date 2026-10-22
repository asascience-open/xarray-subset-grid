"""
make a simple curvilinear grid from a ROMS grid

This was tested, and use with output from the
COOPS OFS subsetter with a small subset from GOMOFS
"""

import numpy as np
import xarray as xr

infile = "6ad12b55-abf1-94640.nc"

print("opening input dataset")
dsin = xr.open_dataset(infile)

print("making u and v")

# averaging to rho points
u = dsin["u_sur"]
u2 = (u[:, 2:, 1:] + u[:, :-2, :-1]) / 2
v = dsin["v_sur"]
v2 = (v[:, 1:, 2:] + v[:, :-1, :-2]) / 2

u2 = u2.swap_dims({"eta_u": "lat", "xi_u": "lon"})
v2 = v2.swap_dims({"eta_v": "lat", "xi_v": "lon"})

new_dims = {"eta_rho": "lat", "xi_rho": "lon"}
lat = dsin["lat_rho"][1:-1, 1:-1]
lat2 = lat.swap_dims(new_dims)
lon = dsin["lon_rho"][1:-1, 1:-1]
lon2 = lon.swap_dims(new_dims)

# rotate the u an v
angle = dsin["angle"][1:-1, 1:-1]
angle = angle.swap_dims(new_dims)
cos = np.cos(angle)
sin = np.sin(angle)

print("computing rotation")
u3 = cos * u2 + sin * v2
v3 = cos * v2 + sin * u2

print("swapping dimensions")

new_dims = {"eta_rho": "lat", "xi_rho": "lon"}
lat = dsin["lat_rho"][1:-1, 1:-1]
lat2 = lat.swap_dims(new_dims)
lon = dsin["lon_rho"][1:-1, 1:-1]
lon2 = lon.swap_dims(new_dims)

lat2.attrs["standard_name"] = "latitude"
lon2.attrs["standard_name"] = "longitude"

print("creating new dataset")

dsout = xr.Dataset(
    data_vars={
        "lat": lat2,
        "lon": lon2,
        "ocean_time": dsin["ocean_time"],
        "u": u3,
        "v": v3,
        # just so there's some non-data vars in there
        "poly_x": dsin["x"],
        "poly_y": dsin["y"],
    },
)
dsout = dsout.drop_vars(["lon_rho", "lat_rho"])

dsout.attrs.update(
    {
        "note": "Converted from ROMS output from CO-OPS subsetting/Aggregation tool",
        "Conventions": "CF-1.4, SGRID-0.3",
        "type": "CF compliant curvilnear grid",
    }
)

print("writting out new dataset")

dsout.to_netcdf("curve_grid_example.nc")
