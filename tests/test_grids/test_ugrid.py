"""
tests for ugrid code
"""
from pathlib import Path
import xarray as xr
from xarray_subset_grid.grids import ugrid

import pytest

EXAMPLE_DATA = Path(__file__).parent.parent.parent / "examples" / "example_data"


def test_assign_ugrid_topology_no_connectivity():
    ds = xr.open_dataset(EXAMPLE_DATA / "SFBOFS_subset1.nc")

    with pytest.raises(TypeError):
        ds = ugrid.assign_ugrid_topology(ds, face_face_connectivity='something')

    assert True


def test_assign_ugrid_topology():
    """
    FVCOM (SFBOFS):
    nv: face_node_connectivity

    (minimum required)
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

def test_assign_ugrid_topology_dict():
    """
    FVCOM (SFBOFS):

    Test passing a dict in of the topology.

    """
    ds = xr.open_dataset(EXAMPLE_DATA / "SFBOFS_subset1.nc")

    # make sure it's not there to start with
    with pytest.raises(KeyError):
        ds['mesh']

    grid_topology = {'node_coordinates': ('lon', 'lat'),
                     'face_node_connectivity': 'nv',
                     'node_coordinates': ('lon', 'lat'),
                     'face_coordinates': ('lonc', 'latc'),
                     }

    ds = ugrid.assign_ugrid_topology(ds, **grid_topology)

    # there are others, but these are the ones that really matter.
    mesh = ds['mesh'].attrs
    assert mesh['cf_role'] == 'mesh_topology'
    assert mesh['node_coordinates'] == 'lon lat'
    assert mesh['face_node_connectivity'] == 'nv'
    assert mesh['face_coordinates'] == 'lonc latc'
    assert mesh['node_coordinates'] == 'lon lat'


