"""
make a simple curvilinear grid from a ROMS grid

['x',
 'y',
 'requested_times',
 'ntimes',
 'ndtfast',
 'dt',
 'dtfast',
 'dstart',
 'nHIS',
 'ndefHIS',
 'nRST',
 'nSTA',
 'Falpha',
 'Fbeta',
 'Fgamma',
 'Akt_bak',
 'Akv_bak',
 'Akk_bak',
 'Akp_bak',
 'rdrg',
 'rdrg2',
 'Zob',
 'Zos',
 'Znudg',
 'M2nudg',
 'M3nudg',
 'Tnudg',
 'FSobc_in',
 'FSobc_out',
 'M2obc_in',
 'M2obc_out',
 'Tobc_in',
 'Tobc_out',
 'M3obc_in',
 'M3obc_out',
 'rho0',
 'gamma2',
 'spherical',
 'xl',
 'el',
 'h',
 'f',
 'pm',
 'pn',
 'angle',
 'mask_rho',
 'mask_u',
 'mask_v',
 'mask_psi',
 'nf_type',
 'Times',
 'temp_sur',
 'salt_sur',
 'u_sur',
 'v_sur',
 'Uwind',
 'Vwind',
 'wetdry_mask_rho',
 'wetdry_mask_u',
 'wetdry_mask_v',
 'wetdry_mask_psi']
"""

import xarray as xr

infile = "6ad12b55-abf1-94640.nc"

dsin = xr.open_dataset(infile)


u = dsin['u_sur']
u2 = (u[1:,1:] + u[:-1,:-1]) / 2
v = dsin['u_sur']
v2 = (v[1:,1:] + v[:-1,:-1]) / 2

new_dims = {'eta_u':'lat', 'xi_u': 'lon'}
u2 = u2.swap_dims(new_dims)
v2 = v2.swap_dims(new_dims)
new_dims = {'eta_psi':'lat', 'xi_psi': 'lon'}
lat = dsin['lat_psi']
lat2 = lat.swap_dims(new_dims)
lon = dsin['lon_psi']
lon2 = lon.swap_dims(new_dims)



dsout = xr.Dataset(data_vars={'lat': lat2,
                              'lon': lon2,
                              'ocean_time': dsin['ocean_time'],
                              'u': u2,
                              'v': v2,},
                              )
dsout = dsout.drop_vars(['lon_psi', 'lat_psi'])

dsout.attrs.update({'note': "Converted from ROMS output from CO-OPS subsetting/Aggregation tool",
                    'Conventions': 'CF-1.4, SGRID-0.3',
                    'type': 'CF compliant curvilnear grid',
                    })

dsout.to_netcdf('curve_grid_example.nc')







