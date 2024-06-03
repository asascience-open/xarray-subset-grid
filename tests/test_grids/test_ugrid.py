"""
tests for ugrid code
"""
from pathlib import Path
import xarray as xr
from xarray_subset_grid.grids import ugrid

import pytest

EXAMPLE_DATA = Path(__file__).parent.parent.parent / "examples" / "example_data"

TEST_FILE1 = EXAMPLE_DATA / "SFBOFS_subset1.nc"

# topology for TEST_FILE1
grid_topology = {'node_coordinates': ('lon', 'lat'),
                 'face_node_connectivity': 'nv',
                 'node_coordinates': ('lon', 'lat'),
                 'face_coordinates': ('lonc', 'latc'),
                 'face_face_connectivity': 'nbe'
                 }

def test_assign_ugrid_topology_no_connectivity():
    ds = xr.open_dataset(EXAMPLE_DATA / "SFBOFS_subset1.nc")

    with pytest.raises(TypeError):
        ds = ugrid.assign_ugrid_topology(ds, face_face_connectivity='something')

    assert True


def test_assign_ugrid_topology_min():
    """
    FVCOM (SFBOFS):
    nv: face_node_connectivity

    testing passing in only the minimum required
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


# NOTE: these tests are probably not complete -- but they are something.
#       we really should have a complete UGRID example to test with.
def test_grid_vars():
    """
    Check if the grid vars are defined properly
    """
    ds = xr.open_dataset(EXAMPLE_DATA / "SFBOFS_subset1.nc")

    ds = ugrid.assign_ugrid_topology(ds, **grid_topology)

    grid_vars = ds.subset_grid.grid_vars

    print([*ds])
    print(grid_vars)
    # ['mesh', 'nv', 'lon', 'lat', 'lonc', 'latc']
    print(ds['mesh'].attrs)
    assert set(grid_vars) == set(['mesh', 'nv', 'nbe', 'lon', 'lat', 'lonc', 'latc'])



@pytest.mark.xfail
def test_data_vars():
    """
    Check if the grid vars are defined properly

    This is not currently working correctly!
    """
    ds = xr.open_dataset(EXAMPLE_DATA / "SFBOFS_subset1.nc")
    ds = ugrid.assign_ugrid_topology(ds, **grid_topology)

    data_vars = ds.subset_grid.data_vars

    print([*ds])
    print(data_vars)
    assert set(data_vars) == set([
        'x', 'y', 'cell', 'h', 'zeta', 'temp', 'salinity', 'u', 'v', 'uwind_speed', 'vwind_speed',
        'wet_nodes', 'wet_cells'
    ])


