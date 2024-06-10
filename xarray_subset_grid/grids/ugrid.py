from typing import Union

import numpy as np
import xarray as xr

from xarray_subset_grid.grid import Grid
from xarray_subset_grid.utils import (
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

        # NOTE: When the first dimension is "nele", the face_node_connectivity
        #       is indexed by element first, then vertex. When the first dimension
        if ds[mesh.face_node_connectivity].dims[0] == "nele":
            transpose_face_node_connectivity = False
            face_node_connectivity = ds[mesh.face_node_connectivity]
        else:
            transpose_face_node_connectivity = True
            face_node_connectivity = ds[mesh.face_node_connectivity].T

        # If any nodes in an element are inside the polygon, the element is
        # inside the polygon so make sure all of the relevent nodes and
        # elements are unmasked
        x = x.values
        y = y.values
        polygon = normalize_polygon_x_coords(x, polygon)
        node_inside = ray_tracing_numpy(x, y, polygon)

        # NOTE: UGRIDS can be zero-indexed OR one-indexed!
        #       see the UGRID spec.
        tris = face_node_connectivity - 1
        tri_mask = node_inside[tris]
        elements_inside = tri_mask.any(axis=1)
        tri_mask[elements_inside] = True

        node_inside[tris] = tri_mask

        # Re-index the nodes and elements to remove the masked ones
        selected_nodes = np.sort(np.unique(tris[elements_inside].values.flatten()))
        selected_elements = np.sort(np.unique(np.where(elements_inside)))
        face_node_new = np.searchsorted(
            selected_nodes, face_node_connectivity[selected_elements]
        )
        if transpose_face_node_connectivity:
            face_node_new = face_node_new.T

        if has_face_face_connectivity:
            if ds[mesh.face_node_connectivity].dims[0] == "nele":
                transpose_face_face_connectivity = False
                face_face_connectivity = ds[mesh.face_face_connectivity]
            else:
                transpose_face_face_connectivity = True
                face_face_connectivity = ds[mesh.face_face_connectivity].T
            face_face_new = np.searchsorted(
                selected_elements, face_face_connectivity[selected_elements]
            )

            if transpose_face_face_connectivity:
                face_face_new = face_face_new.T

        # Subset using xarrays select indexing, and overwrite the face_node_connectivity
        # and face_face_connectivity (if available) with the new indices
        ds_subset = ds.sel(node=selected_nodes, nele=selected_elements).drop_encoding()
        ds_subset[mesh.face_node_connectivity][:] = face_node_new
        if has_face_face_connectivity:
            ds_subset[mesh.face_face_connectivity][:] = face_face_new
        return ds_subset

def assign_ugrid_topology(ds: xr.Dataset,
                          *,
                          face_node_connectivity: str,
                          face_face_connectivity: str = None,
                          node_coordinates: str = None,
                          face_coordinates: str = None,
                          start_index: int = None,
                          ) -> xr.Dataset:
    # Should this be "make entire dataset UGRID compliant ?"
    #  That would mean that the grid variables should all get metadata,
    #  such as "location"
    #  and we'd need to clean up coordinates that shouldn't be coordinates.
    #  ("node is one that's in the UGRID test file")
    """
    Assign the UGRID topology to the dataset

    Only the face_node_connectivity parameter is required.
    The face_face_connectivity parameter is optional.

    If the variable for face_node_connectivity is named nv, the function call should look like this:

    ```
    ds = assign_ugrid_topology(ds, face_node_connectivity="nv")
    ```

    This will assign a new variable to the dataset called "mesh" with the face_node_connectivity attribute. It will
    also assign the node_coordinates attribute to the dataset with the lon and lat coordinate variable names, introspected
    from the face_node_connectivity variables' dimensions.

    You can also optionally specify any of the following mesh topology variables by passing them as keyword arguments
    - node_coordinates: If not specified, the function will introspect the dataset for the longitude and latitude coordinate variable names using cf_xarray
    - face_face_connectivity
    - face_coordinates: If not specified, the function will introspect the dataset for the longitude and latitude coordinate variable names matching the face_node_connectivity variable
                        but do nothing if they are not found

    Args:
        ds (xr.Dataset): The dataset to assign the UGRID topology to
        face_node_connectivity (str): THe variable name of the face definitions

        face_face_connectivity: str = None,
        node_coordinates: str = None,
        face_coordinates: str = None,

    (See the UGRID conventions for descriptions of these)

    You can pass a dict in with all the grid topology variables:

    ```
        grid_topology = {'node_coordinates': ('lon', 'lat'),
                     'face_node_connectivity': 'nv',
                     'node_coordinates': ('lon', 'lat'),
                     'face_coordinates': ('lonc', 'latc'),
                     }
    ```
    """
    # All the possible attributes:
    # Required:
    # node_coordinates
    # face_node_connectivity

    # Optional:
    # face_dimension
    # edge_node_connectivity
    # edge_dimension
    # Optional attributes
    # face_edge_connectivity
    # face_face_connectivity
    # edge_face_connectivity
    # boundary_node_connectivity
    # face_coordinates
    # edge_coordinates

    # Get the variable name for the face_node_connectivity
    # face_node_connectivity = attrs.get("face_node_connectivity", None)
    # if face_node_connectivity is None:
    #     raise ValueError("The face_node_connectivity attribute is required")
    #face_face_connectivity = attrs.get("face_face_connectivity", None)

    # Get the longitude and latitude coordinate variable names
    # node_coords = attrs.get("node_coordinates", None)
    # face_coords = attrs.get("face_coordinates", None)

    node_coords = node_coordinates
    face_coords = face_coordinates

    if not face_coords:
        try:
            face_coords = ds[face_node_connectivity].cf.coordinates
            face_coords = [f"{coord[0]}" for coord in face_coords.values()]
        except AttributeError:
            face_coords = None

    if not node_coords:
        try:
            if face_coords:
                filter = face_coords
            else:
                filter = []

            coords = ds.cf.coordinates
            node_lon = [c for c in coords["longitude"] if c not in filter][0]
            node_lat = [c for c in coords["latitude"] if c not in filter][0]
            node_coords = [node_lon, node_lat]
        except AttributeError:
            raise ValueError(
                "The dataset does not have cf_compliant node coordinates longitude and latitude coordinates"
            )

    if not face_face_connectivity:
        # face_face_connectivity var should have same dimensions as
        # face_node_connectivity this is assuming that only one will match!
        for var_name, var in ds.variables.items():
            if var_name == face_node_connectivity:
                continue
            if var.dims == ds[face_node_connectivity].dims:
                face_face_connectivity = var_name
                break

    mesh_attrs = {
        "cf_role": "mesh_topology",
        "topology_dimension": np.int32(2),
        "node_coordinates": " ".join(node_coords),
        "face_node_connectivity": face_node_connectivity,
    }

    if face_coords:
        mesh_attrs["face_coordinates"] = " ".join(face_coords)
    if face_face_connectivity:
        mesh_attrs["face_face_connectivity"] = face_face_connectivity

    # Assign the mesh topology to the dataset
    ds = ds.assign(
        mesh=((), np.int32(0), mesh_attrs),
    )

    # check for start_index, and set it if there.
    if start_index is None:
        start_index = int(ds[face_node_connectivity].min())

    if not start_index in (0, 1):
        raise ValueError(f"start_index must be 1 or 0, not {start_index}")

    # assign the start_index to all the grid variables
    for var_name in (face_node_connectivity, face_face_connectivity):
        if var_name:
            var = ds[var_name]
            var.attrs.setdefault('start_index', start_index)

    return ds
