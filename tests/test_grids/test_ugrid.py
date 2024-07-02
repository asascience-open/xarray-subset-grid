"""
tests for ugrid code
"""
from pathlib import Path

import pytest
import xarray as xr

from xarray_subset_grid.grids import ugrid

EXAMPLE_DATA = Path(__file__).parent.parent.parent / "examples" / "example_data"

TEST_FILE1 = EXAMPLE_DATA / "SFBOFS_subset1.nc"

"""
SFBOFS_subset1.nc is a smallish subset of the SFBOFS FVCOM model

It was created by the "OFS subsetter"

So is the same as the OFS results, but with a few extra variables.

It has the following variables:


Original variables:

Grid definition:

The node_coordinates:
    float lon(node) ;
        lon:standard_name = "longitude" ;
        lon:long_name = "nodal longitude" ;
        lon:units = "degree_east" ;
    float lat(node) ;
        lat:standard_name = "latitude" ;
        lat:long_name = "nodal latitude" ;
        lat:units = "degree_north" ;

face_coordinates:

    float lonc(nele) ;
        lonc:standard_name = "longitude" ;
        lonc:long_name = "zonal longitude" ;
        lonc:units = "degree_east" ;
    float latc(nele) ;
        latc:standard_name = "latitude" ;
        latc:long_name = "zonal latitude" ;
        latc:units = "degree_north" ;

face_node_connectivity:
    int nv(three, nele) ;
        nv:long_name = "nodes surrounding an element" ;

face_face_connectivity:
    int nbe(three, nele) ;
        nbe:long_name = "elements surrounding anch element" ;

The depth coordinates (xarray is not sure what to do with these ...)

    float siglay(siglay, node) ;
        siglay:standard_name = "ocean_sigma/general_coordinate" ;
        siglay:long_name = "Sigma Layers" ;
        siglay:positive = "up" ;
        siglay:valid_min = -1.f ;
        siglay:valid_max = 0.f ;
        siglay:formula_terms = "sigma: siglay eta: zeta depth: h" ;
    float siglev(siglev, node) ;
        siglev:standard_name = "ocean_sigma/general_coordinate" ;
        siglev:long_name = "Sigma Levels" ;
        siglev:positive = "up" ;
        siglev:valid_min = -1.f ;
        siglev:valid_max = 0.f ;
        siglev:formula_terms = "sigma: siglay eta: zeta depth: h" ;

The time variable (coordinate variable?)

    float time(time) ;
        time:long_name = "time" ;
        time:units = "days since 2013-01-01 00:00:00" ;
        time:format = "defined reference date" ;
        time:time_zone = "UTC" ;


Data on the grid:


    float h(node) ;
        h:standard_name = "sea_floor_depth_below_geoid" ;
        h:long_name = "Bathymetry" ;
        h:units = "m" ;
        h:positive = "down" ;
        h:grid = "Bathymetry_Mesh" ;
        h:coordinates = "lat lon" ;
        h:type = "data" ;

    float zeta(time, node) ;
        zeta:standard_name = "sea_surface_height_above_geoid" ;
        zeta:long_name = "water surface elevation" ;
        zeta:units = "meters" ;
        zeta:positive = "up" ;
        zeta:grid = "Bathymetry_Mesh" ;
        zeta:type = "data" ;
        zeta:coordinates = "time lat lon" ;
        zeta:location = "node" ;
    float temp(time, siglay, node) ;
        temp:standard_name = "sea_water_temperature" ;
        temp:long_name = "temperature" ;
        temp:units = "degree_C" ;
        temp:grid = "fvcom_grid" ;
        temp:type = "data" ;
        temp:coordinates = "time siglay lat lon" ;
        temp:mesh = "fvcom_mesh" ;
        temp:location = "node" ;
    float salinity(time, siglay, node) ;
        salinity:standard_name = "sea_water_salinity" ;
        salinity:long_name = "salinity" ;
        salinity:units = "1e-3" ;
        salinity:grid = "fvcom_grid" ;
        salinity:type = "data" ;
        salinity:coordinates = "time siglay lat lon" ;
        salinity:mesh = "fvcom_mesh" ;
        salinity:location = "node" ;
    float u(time, siglay, nele) ;
        u:standard_name = "eastward_sea_water_velocity" ;
        u:long_name = "eastward water velocity" ;
        u:units = "meters s-1" ;
        u:grid = "fvcom_grid" ;
        u:type = "data" ;
        u:coordinates = "time siglay latc lonc" ;
        u:mesh = "fvcom_mesh" ;
        u:location = "face" ;
    float v(time, siglay, nele) ;
        v:standard_name = "northward_sea_water_velocity" ;
        v:long_name = "Northward Water Velocity" ;
        v:units = "meters s-1" ;
        v:grid = "fvcom_grid" ;
        v:type = "data" ;
        v:coordinates = "time siglay latc lonc" ;
        v:mesh = "fvcom_mesh" ;
        v:location = "face" ;
    float uwind_speed(time, nele) ;
        uwind_speed:standard_name = "eastward wind" ;
        uwind_speed:long_name = "eastward wind velocity" ;
        uwind_speed:units = "meters s-1" ;
        uwind_speed:grid = "fvcom_grid" ;
        uwind_speed:coordinates = "time latc lonc" ;
        uwind_speed:type = "data" ;
        uwind_speed:mesh = "fvcom_mesh" ;
        uwind_speed:location = "face" ;

    float vwind_speed(time, nele) ;
        vwind_speed:standard_name = "northward wind" ;
        vwind_speed:long_name = "northward wind velocity" ;
        vwind_speed:units = "meters s-1" ;
        vwind_speed:grid = "fvcom_grid" ;
        vwind_speed:coordinates = "time latc lonc" ;
        vwind_speed:type = "data" ;
        vwind_speed:mesh = "fvcom_mesh" ;
        vwind_speed:location = "face" ;

    int wet_nodes(time, node) ;
        wet_nodes:long_name = "Wet_Nodes" ;
        wet_nodes:grid = "fvcom_grid" ;
        wet_nodes:type = "data" ;
        wet_nodes:option_0 = "land" ;
        wet_nodes:option_1 = "water" ;
        wet_nodes:coordinates = "time lat lon" ;
        wet_nodes:mesh = "fvcom_mesh" ;
        wet_nodes:location = "node" ;

    int wet_cells(time, nele) ;
        wet_cells:long_name = "Wet_Cells" ;
        wet_cells:grid = "fvcom_grid" ;
        wet_cells:type = "data" ;
        wet_cells:option_0 = "land" ;
        wet_cells:option_1 = "water" ;
        wet_cells:coordinates = "time latc lonc" ;
        wet_cells:mesh = "fvcom_mesh" ;
        wet_cells:location = "face" ;


Extra variables (i.e. not using the grid)

    int nf_type(time) ;
        nf_type:long_name = "data from nowcast or forecast output" ;
        nf_type:option_1 = "data from nowcast output" ;
        nf_type:option_2 = "data from forecast output" ;
    char Times(time, DateStrLen) ;
        Times:time_zone = "UTC" ;

Added by the subsetter:

    float x(nply) ;
        x:standard_name = "longitude of the polygon vertices" ;
        x:long_name = "longitude" ;
        x:units = "degree_east" ;
    float y(nply) ;
        y:standard_name = "latitude of the polygon vertices" ;
        y:long_name = "latitude" ;
        y:units = "degree_north" ;
    char requested_times(two, DateStrLen) ;
        requested_times:standard_name = "time" ;
        requested_times:long_name = "requested time period" ;
        requested_times:time_zone = "UTC" ;
        requested_times:start_date = "2024-05-23T00:00" ;
        requested_times:end_date = "2024-05-25T00:00" ;

    int node(node) ;
        node:standard_name = "node number" ;
        node:long_name = "Mapping to original mesh node number" ;

    int cell(nele) ;
        cell:standard_name = "cell number" ;
        cell:long_name = "Mapping to original mesh cell number" ;




"""

# topology for TEST_FILE1
grid_topology = {'node_coordinates': ('lon', 'lat'),
                 'face_node_connectivity': 'nv',
                 'node_coordinates': ('lon', 'lat'),
                 'face_coordinates': ('lonc', 'latc'),
                 'face_face_connectivity': 'nbe'
                 }

def test_assign_ugrid_topology_no_connectivity():
    """
    It should raise if you don't give it anything.
    """
    ds = xr.open_dataset(EXAMPLE_DATA / "SFBOFS_subset1.nc")

    with pytest.raises(ValueError):
        ds = ugrid.assign_ugrid_topology(ds, face_face_connectivity='something')

    assert True


def test_assign_ugrid_topology_min():
    """
    FVCOM (SFBOFS):
    nv: face_node_connectivity

    Passing in only face_node_connectivity should be enough
    """
    ds = xr.open_dataset(EXAMPLE_DATA / "SFBOFS_subset1.nc")

    # make sure it's not there to start with
    with pytest.raises(KeyError):
        ds['mesh']

    ds = ugrid.assign_ugrid_topology(ds, face_node_connectivity='nv')

    # there are others, but these are the ones that really matter.
    mesh = ds['mesh'].attrs
    assert mesh['cf_role'] == 'mesh_topology'
    assert mesh['node_coordinates'] == 'lon lat'
    assert mesh['face_node_connectivity'] == 'nv'
    assert mesh['face_coordinates'] == 'lonc latc'
    assert mesh['face_face_connectivity'] == 'nbe'


def test_assign_ugrid_topology_dict():
    """
    FVCOM (SFBOFS):

    Test passing a dict in of the topology.

    """
    ds = xr.open_dataset(EXAMPLE_DATA / "SFBOFS_subset1.nc")

    # make sure it's not there to start with
    with pytest.raises(KeyError):
        ds['mesh']

    ds = ugrid.assign_ugrid_topology(ds, **grid_topology)

    # there are others, but these are the ones that really matter.
    mesh = ds['mesh'].attrs
    assert mesh['cf_role'] == 'mesh_topology'
    assert mesh['node_coordinates'] == 'lon lat'
    assert mesh['face_node_connectivity'] == 'nv'
    assert mesh['face_coordinates'] == 'lonc latc'
    assert mesh['node_coordinates'] == 'lon lat'
    assert mesh['face_face_connectivity'] == 'nbe'

def test_assign_ugrid_topology_existing_mesh_var():
    """
    It should raise if you don't give it anything.

    The existing one in the file:

    int mesh ;
        mesh:cf_role = "mesh_topology" ;
        mesh:long_name = "Topology data of 2D unstructured mesh" ;
        mesh:topology_dimension = 2LL ;
        mesh:node_coordinates = "mesh_node_lon mesh_node_lat" ;
        mesh:edge_node_connectivity = "mesh_edge_nodes" ;
        mesh:edge_coordinates = "mesh_edge_lon mesh_edge_lat" ;
        mesh:face_node_connectivity = "mesh_face_nodes" ;
        mesh:face_coordinates = "mesh_face_lon mesh_face_lat" ;
        mesh:face_face_connectivity = "mesh_face_links" ;
        mesh:boundary_node_connectivity = "mesh_boundary_nodes" ;
    """

    ds = xr.open_dataset(EXAMPLE_DATA / "small_ugrid_zero_based.nc")

    # assigning the one that's already there -- should be non-destructive
    ds_new = ugrid.assign_ugrid_topology(ds, face_node_connectivity='mesh_face_nodes')

    old_mesh_var = ds['mesh']
    new_mesh_var = ds_new['mesh']

    assert old_mesh_var.attrs == new_mesh_var.attrs


def test_assign_ugrid_topology_start_index_one():
    ds = xr.open_dataset(EXAMPLE_DATA / "SFBOFS_subset1.nc")
    ds = ugrid.assign_ugrid_topology(ds, **grid_topology)
    # bit fragile, but easy to write the test
    assert ds['nv'].attrs['start_index'] == 1
    assert ds['nbe'].attrs['start_index'] == 1


def test_assign_ugrid_topology_start_index_zero_specify():
    ds = xr.open_dataset(EXAMPLE_DATA / "small_ugrid_zero_based.nc")
    ds = ugrid.assign_ugrid_topology(ds, face_node_connectivity='mesh_face_nodes', start_index=0)

    # a bit fragile, but easy to write the test
    assert ds['mesh_face_nodes'].attrs['start_index'] == 0
    assert ds['mesh_edge_nodes'].attrs['start_index'] == 0
    assert ds['mesh_boundary_nodes'].attrs['start_index'] == 0

def test_assign_ugrid_topology_start_index_zero_infer():
    ds = xr.open_dataset(EXAMPLE_DATA / "small_ugrid_zero_based.nc")
    ds = ugrid.assign_ugrid_topology(ds, face_node_connectivity='mesh_face_nodes')

    # a bit fragile, but easy to write the test
    assert ds['mesh_face_nodes'].attrs['start_index'] == 0
    assert ds['mesh_edge_nodes'].attrs['start_index'] == 0
    assert ds['mesh_boundary_nodes'].attrs['start_index'] == 0


# NOTE: these tests are probably not complete -- but they are something.
#       we really should have a complete UGRID example to test with.
def test_grid_vars():
    """
    Check if the grid vars are defined properly
    """
    ds = xr.open_dataset(EXAMPLE_DATA / "SFBOFS_subset1.nc")

    ds = ugrid.assign_ugrid_topology(ds, **grid_topology)

    grid_vars = ds.subset_grid.grid_vars

    # ['mesh', 'nv', 'lon', 'lat', 'lonc', 'latc']
    assert grid_vars == set(['mesh', 'nv', 'nbe', 'lon', 'lat', 'lonc', 'latc'])


def test_data_vars():
    """
    Check if the grid vars are defined properly

    This is not currently working correctly!
    """
    ds = xr.open_dataset(EXAMPLE_DATA / "SFBOFS_subset1.nc")
    ds = ugrid.assign_ugrid_topology(ds, **grid_topology)

    data_vars = ds.subset_grid.data_vars

    assert set(data_vars) == set(['h',
                                  'zeta',
                                  'temp',
                                  'salinity',
                                  'u',
                                  'v',
                                  'uwind_speed',
                                  'vwind_speed',
                                  'wet_nodes',
                                  'wet_cells'
    ])


def test_extra_vars():
    """
    Check if the extra vars are defined properly

    This is not currently working correctly!
    """
    ds = xr.open_dataset(EXAMPLE_DATA / "SFBOFS_subset1.nc")
    ds = ugrid.assign_ugrid_topology(ds, **grid_topology)


    extra_vars = ds.subset_grid.extra_vars

    print([*ds])
    print(f"{extra_vars=}")
    assert extra_vars == set(['nf_type',
                              'Times',
                              ])


def test_coords():

    ds = xr.open_dataset(EXAMPLE_DATA / "SFBOFS_subset1.nc")
    ds = ugrid.assign_ugrid_topology(ds, **grid_topology)

    coords = ds.subset_grid.coords

    print(f'{coords=}')
    print(f'{ds.coords=}')

    assert set(coords) == set(['lon',
                               'lat',
                               'lonc',
                               'latc',
                               'time',
                               'siglay',
                               'siglev',
                               ])

