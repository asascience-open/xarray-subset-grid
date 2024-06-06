from typing import Union

import numpy as np
import xarray as xr

from xarray_subset_grid.grid import Grid
from xarray_subset_grid.utils import (
    assign_ugrid_topology,
    normalize_polygon_x_coords,
    ray_tracing_numpy,
)


class UGrid(Grid):
    """Grid implementation for UGRID datasets

    UGRID is a grid type that is used to represent unstructured grids. It is used to represent grids where the elements
    are not regular, such as triangular or quadrilateral grids. UGRID is a standard that is used in the oceanographic
    community.

    In this specific grid implementation, we assume that the dataset has a variable that describes the mesh
    with the mesh_topology cf_role. This variable should have a face_node_connectivity attribute that describes the
    connectivity of the nodes to the elements. The face_node_connectivity attribute should be a 2D array where the
    first dimension is the number of elements and the second dimension is the number of nodes per element. The values
    in the array should be the indices of the nodes in the node variable that are connected to the element.

    The face_face_connectivity attribute is optional and describes the connectivity of the elements to each other. It
    should be a 2D array where the first dimension is the number of elements and the second dimension is the number of
    elements that are connected to the element.

    # TODO: Abstract away common subsetting methods to functions that can be cached for reuse
    """

    @staticmethod
    def recognize(ds: xr.Dataset) -> bool:
        """Recognize if the dataset matches the given grid"""
        try:
            mesh_key = ds.cf.cf_roles["mesh_topology"][0]
            mesh = ds[mesh_key]
        except Exception:
            return False

        return mesh.attrs.get("face_node_connectivity") is not None

    @property
    def name(self) -> str:
        """Name of the grid type"""
        return "ugrid"

    def grid_vars(self, ds: xr.Dataset) -> set[str]:
        """
        List of grid variables

        These variables are used to define the grid and thus should be kept
        when subsetting the dataset
        """
        mesh = ds.cf["mesh_topology"]
        vars = {mesh.name}
        if "face_node_connectivity" in mesh.attrs:
            vars.add(mesh.face_node_connectivity)
        if "face_face_connectivity" in mesh.attrs:
            vars.add(mesh.face_face_connectivity)
        if "node_coordinates" in mesh.attrs:
            node_coords = mesh.node_coordinates.split(" ")
            vars.update(node_coords)
        if "face_coordinates" in mesh.attrs:
            face_coords = mesh.face_coordinates.split(" ")
            vars.update(face_coords)

        return vars

    def data_vars(self, ds: xr.Dataset) -> set[str]:
        """
        Set of data variables

        These variables exist on the grid and are available to used for
        data analysis. These can be discarded when subsetting the dataset
        when they are not needed.

        Then all grid_vars are excluded as well.
        """
        mesh = ds.cf["mesh_topology"]
        dims = []

        # Use the coordinates as the source of truth, the face and node
        # dimensions are the same as the coordinates and any data variables
        # that do not contain either face or node dimensions can be ignored
        face_coord = mesh.face_coordinates.split(" ")[0]
        dims.extend(ds[face_coord].dims)
        node_coord = mesh.node_coordinates.split(" ")[0]
        dims.extend(ds[node_coord].dims)

        dims = set(dims)

        data_vars = {var for var in ds.data_vars if not set(ds[var].dims).isdisjoint(dims)}
        data_vars -= self.grid_vars(ds)

        return data_vars


    def subset_polygon(
        self, ds: xr.Dataset, polygon: Union[list[tuple[float, float]], np.ndarray]
    ) -> xr.Dataset:
        """Subset the dataset to the grid
        :param ds: The dataset to subset
        :param polygon: The polygon to subset to
        :return: The subsetted dataset
        """
        # For this grid type, we find all nodes that are connected to elements that are inside the polygon. To do this,
        # we first find all nodes that are inside the polygon and then find all elements that are connected to those nodes.
        mesh = ds.cf["mesh_topology"]
        has_face_face_connectivity = "face_face_connectivity" in mesh.attrs
        x_var, y_var = mesh.node_coordinates.split(" ")
        x, y = ds[x_var], ds[y_var]

        # If any nodes in an element are inside the polygon, the element is
        # inside the polygon so make sure all of the relevent nodes and
        # elements are unmasked
        x = x.values
        y = y.values
        polygon = normalize_polygon_x_coords(x, polygon)
        node_inside = ray_tracing_numpy(x, y, polygon)
        # NOTE: UGRIDS can be zero-indexed OR one-indexed!
        #       see the UGRID spec.
        tris = ds[mesh.face_node_connectivity].T - 1
        tri_mask = node_inside[tris]
        elements_inside = tri_mask.any(axis=1)
        tri_mask[elements_inside] = True
        node_inside[tris] = tri_mask

        # Re-index the nodes and elements to remove the masked ones
        selected_nodes = np.sort(np.unique(tris[elements_inside].values.flatten()))
        selected_elements = np.sort(np.unique(np.where(elements_inside)))
        face_node_new = np.searchsorted(
            selected_nodes, ds[mesh.face_node_connectivity].T[selected_elements]
        ).T
        if has_face_face_connectivity:
            face_face_new = np.searchsorted(
                selected_elements, ds[mesh.face_face_connectivity].T[selected_elements]
            ).T

        # Subset using xarrays select indexing, and overwrite the face_node_connectivity
        # and face_face_connectivity (if available) with the new indices
        ds_subset = ds.sel(node=selected_nodes, nele=selected_elements).drop_encoding()
        ds_subset[mesh.face_node_connectivity][:] = face_node_new
        if has_face_face_connectivity:
            ds_subset[mesh.face_face_connectivity][:] = face_face_new
        return ds_subset
