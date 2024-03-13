import numpy as np
import xarray as xr

from xarray_subset_grid.grid import Grid
from xarray_subset_grid.utils import ray_tracing_numpy


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
            mesh = ds.cf["mesh_topology"]
        except KeyError:
            return False

        return (
            mesh.attrs.get("cf_role") == "mesh_topology"
            and mesh.attrs.get("face_node_connectivity") is not None
        )

    @property
    def name(self) -> str:
        """Name of the grid type"""
        return "ugrid"

    def subset_polygon(
        self, ds: xr.Dataset, polygon: list[tuple[float, float]] | np.ndarray
    ) -> xr.Dataset:
        """Subset the dataset to the grid
        :param ds: The dataset to subset
        :param polygon: The polygon to subset to
        :return: The subsetted dataset
        """
        # For this grid type, we find all nodes that are connected to elements that are inside the polygon. To do this,
        # we first find all nodes that are inside the polygon and then find all elements that are connected to those nodes.
        mesh = ds.cf["mesh_topology"]
        x_var, y_var = mesh.node_coordinates.split(" ")
        x, y = ds[x_var], ds[y_var]

        # If any nodes in an element are inside the polygon, the element is inside the polygon so make sure all of the relevent nodes and elements are unmasked
        node_inside = ray_tracing_numpy(x.values, y.values, polygon)
        tris = ds[mesh.face_node_connectivity].T - 1
        tri_mask = node_inside[tris]
        elements_inside = tri_mask.any(axis=1)
        tri_mask[elements_inside] = True
        node_inside[tris] = tri_mask

        # Reindex the nodes and elements to remove the masked ones
        selected_nodes = np.sort(np.unique(tris[elements_inside].values.flatten()))
        selected_elements = np.sort(np.unique(np.where(elements_inside)))
        face_node_new = np.searchsorted(
            selected_nodes, ds[mesh.face_node_connectivity].T[selected_elements]
        ).T
        face_face_new = np.searchsorted(
            selected_elements, ds[mesh.face_face_connectivity].T[selected_elements]
        ).T

        # Subset using xarrays select indexing, and overwrite the face_node_connectivity and face_face_connectivity
        # with the new indices
        ds_subset = ds.sel(node=selected_nodes, nele=selected_elements)
        ds_subset[mesh.face_node_connectivity][:] = face_node_new
        ds_subset[mesh.face_face_connectivity][:] = face_face_new
        return ds_subset
