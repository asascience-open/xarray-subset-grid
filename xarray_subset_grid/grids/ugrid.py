import warnings
from types import SimpleNamespace

import numpy as np
import xarray as xr

from xarray_subset_grid.grid import Grid
from xarray_subset_grid.selector import Selector
from xarray_subset_grid.utils import (
    normalize_polygon_x_coords,
    ray_tracing_numpy,
)

ALL_MESH_VARS = (
    "node_coordinates",
    "face_coordinates",
    "edge_coordinates",
    "face_node_connectivity",
    "face_face_connectivity",
    "boundary_node_connectivity",
    "face_edge_connectivity",
    "edge_face_connectivity",
    "edge_node_connectivity",
)


class UGridSelector(Selector):
    polygon: list[tuple[float, float]] | np.ndarray

    _node_dimension: str
    _selected_nodes: np.ndarray

    _face_dimension: str
    _selected_elements: np.ndarray

    _face_node_connectivity_key: str
    _face_node_connectivity: np.ndarray

    _face_face_connectivity_key: str | None
    _face_face_connectivity: np.ndarray | None

    def __init__(
        self,
        name: str,
        polygon: list[tuple[float, float]] | np.ndarray,
        node_dimension: str,
        selected_nodes: np.ndarray,
        face_dimension: str,
        selected_elements: np.ndarray,
        face_node_connectivity_key: str,
        face_node_connectivity: np.ndarray,
        face_face_connectivity_key: str | None = None,
        face_face_connectivity: np.ndarray | None = None,
    ):
        super().__init__()
        self.name = name
        self.polygon = polygon
        self._node_dimension = node_dimension
        self._selected_nodes = selected_nodes
        self._face_dimension = face_dimension
        self._selected_elements = selected_elements
        self._face_node_connectivity_key = face_node_connectivity_key
        self._face_node_connectivity = face_node_connectivity
        self._face_face_connectivity_key = face_face_connectivity_key
        self._face_face_connectivity = face_face_connectivity

    def select(self, ds: xr.Dataset) -> xr.Dataset:
        # Subset using xarrays select indexing, and overwrite the face_node_connectivity
        # and face_face_connectivity (if available) with the new indices
        ds_subset = ds.sel(
            {
                self._node_dimension: self._selected_nodes,
                self._face_dimension: self._selected_elements,
            }
        )
        ds_subset[self._face_node_connectivity_key][:] = self._face_node_connectivity
        if (
            self._face_face_connectivity is not None
            and self._face_face_connectivity_key is not None
        ):
            ds_subset[self._face_face_connectivity_key][:] = self._face_face_connectivity
        return ds_subset


class UGrid(Grid):
    """Grid implementation for UGRID datasets.

    UGRID is a grid type that is used to represent unstructured grids.
    It is used to represent grids where the elements are not regular,
    such as triangular or quadrilateral grids. UGRID is a standard that
    is used in the oceanographic community.

    In this specific grid implementation, we assume that the dataset has
    a variable that describes the mesh with the mesh_topology cf_role.
    This variable should have a face_node_connectivity attribute that
    describes the connectivity of the nodes to the elements. The
    face_node_connectivity attribute should be a 2D array where the
    first dimension is the number of elements and the second dimension
    is the number of nodes per element. The values in the array should
    be the indices of the nodes in the node variable that are connected
    to the element.

    The face_face_connectivity attribute is optional and describes the
    connectivity of the elements to each other. It should be a 2D array
    where the first dimension is the number of elements and the second
    dimension is the number of elements that are connected to the
    element.

    # TODO: Abstract away common subsetting methods to functions that
    can be cached for reuse
    """

    @staticmethod
    def recognize(ds: xr.Dataset) -> bool:
        """Recognize if the dataset matches the given grid."""
        try:
            mesh_key = ds.cf.cf_roles["mesh_topology"][0]
            mesh = ds[mesh_key]
        except Exception:
            return False

        return mesh.attrs.get("face_node_connectivity") is not None

    @property
    def name(self) -> str:
        """Name of the grid type."""
        return "ugrid"

    def grid_vars(self, ds: xr.Dataset) -> set[str]:
        """List of grid variables.

        These variables are used to define the grid and thus should be
        kept when subsetting the dataset
        """
        mesh = ds.cf["mesh_topology"]
        vars = {mesh.name}
        for var_name in ALL_MESH_VARS:
            if var_name in mesh.attrs:
                if "coordinates" in var_name:
                    _node_coordinates = mesh.node_coordinates.split(" ")
                    vars.update(mesh.attrs[var_name].split(" "))
                else:
                    vars.add(mesh.attrs[var_name])
        return vars

    def data_vars(self, ds: xr.Dataset) -> set[str]:
        """Set of data variables.

        These variables exist on the grid and are available to used for
        data analysis. These can be discarded when subsetting the
        dataset when they are not needed.

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

    def compute_polygon_subset_selector(
        self, ds: xr.Dataset, polygon: list[tuple[float, float]], name: str = None
    ) -> Selector:

        # For this grid type, we find all nodes that are connected to elements that are inside
        # the polygon. To do this, we first find all nodes that are inside the polygon and then
        # find all elements that are connected to those nodes.
        try:
            mesh = ds.cf["mesh_topology"]
        except KeyError as err:
            raise ValueError("Dataset has no mesh topology variable") from err
        has_face_face_connectivity = "face_face_connectivity" in mesh.attrs
        x_var, y_var = mesh.node_coordinates.split(" ")
        x, y = ds[x_var], ds[y_var]
        node_dimension = x.dims[0]

        face_dimension = mesh.attrs.get("face_dimension", None)
        if not face_dimension:
            raise ValueError("face_dimension is required to subset UGRID datasets")
        face_node_indices_dimension = next(
            d for d in ds[mesh.face_node_connectivity].dims if d != face_dimension
        )

        # NOTE: When the first dimension is face_dimension, the face_node_connectivity
        #       is indexed by element first, then vertex. When the first dimension
        if ds[mesh.face_node_connectivity].dims[0] == face_dimension:
            transpose_face_node_connectivity = False
            face_node_connectivity = ds[mesh.face_node_connectivity]
        else:
            transpose_face_node_connectivity = True
            face_node_connectivity = ds[mesh.face_node_connectivity].T
        face_node_start_index = face_node_connectivity.attrs.get("start_index", None)
        if not face_node_start_index:
            warnings.warn("No start_index found in face_node_connectivity, assuming 0")
            face_node_start_index = 0

        # If any nodes in an element are inside the polygon, the element is
        # inside the polygon so make sure all of the relevant nodes and
        # elements are unmasked
        x = x.values
        y = y.values
        polygon = normalize_polygon_x_coords(x, polygon)
        node_inside = ray_tracing_numpy(x, y, polygon)

        # NOTE: UGRIDS can be zero-indexed OR one-indexed!
        #       see the UGRID spec.
        # NOTE: The face_node_connectivity may contain masked elements as
        #       per the UGRID spec. We set these to -1 and only slice the
        #       valid elements.
        tris = face_node_connectivity.fillna(-1) - face_node_start_index

        # It is possible that the last index in the face_node_connectivity
        # is masked for elements with > 3 nodes. We fill these with the first
        # index in the face_node_connectivity because it will get filtered out
        # when we slice with the unique nodes in the next step.
        first = tris.sel({face_node_indices_dimension: 0})
        last = tris.sel({face_node_indices_dimension: -1})
        filled_last = last.where(last >= 0, first)
        tris.loc[{face_node_indices_dimension: -1}] = filled_last

        # Store the index as the smallest possible signed integer type
        int_type = np.min_scalar_type(np.max(node_inside.shape))
        valid_tris = tris.astype(int_type)

        # Mask the elements that are not inside the polygon
        tri_mask = node_inside[valid_tris]
        elements_inside = tri_mask.any(axis=1)
        tri_mask[elements_inside] = True
        node_inside[valid_tris] = tri_mask

        # Re-index the nodes and elements to remove the masked ones
        selected_nodes = np.sort(np.unique(valid_tris[elements_inside].values.flatten()))
        selected_elements = np.sort(np.unique(np.where(elements_inside)))
        face_node_new = np.searchsorted(selected_nodes, face_node_connectivity[selected_elements])
        if transpose_face_node_connectivity:
            face_node_new = face_node_new.T

        if has_face_face_connectivity:
            if ds[mesh.face_node_connectivity].dims[0] == face_dimension:
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

        return UGridSelector(
            name=name or 'selector',
            polygon=polygon,
            node_dimension=node_dimension,
            selected_nodes=selected_nodes,
            face_dimension=face_dimension,
            selected_elements=selected_elements,
            face_node_connectivity_key=mesh.face_node_connectivity,
            face_node_connectivity=face_node_new,
            face_face_connectivity_key=mesh.face_face_connectivity
            if has_face_face_connectivity
            else None,
            face_face_connectivity=face_face_new if has_face_face_connectivity else None,
        )


def assign_ugrid_topology(
    ds: xr.Dataset,
    *,
    face_node_connectivity: str | None = None,
    face_face_connectivity: str | None = None,
    boundary_node_connectivity: str | None = None,
    face_edge_connectivity: str = None,
    edge_node_connectivity: str | None = None,
    edge_face_connectivity: str | None = None,
    node_coordinates: str | None = None,
    face_coordinates: str | None = None,
    edge_coordinates: str | None = None,
    face_dimension: str | None = None,
    edge_dimension: str | None = None,
    start_index: int | None = None,
    ) -> xr.Dataset:
    # Should this be "make the entire dataset UGRID compliant ?"
    #    That would mean that the grid variables should all get metadata,
    #    such as "location"
    #    and we'd need to clean up coordinates that shouldn't be coordinates.
    #    ("node is one that's in the UGRID test file")
    """
    Assign the UGRID topology to the dataset.

    This should make a unstructured grid file UGRID (and CF) compliant, and should
    allow the rest of the code to work.

    If the Dataset already has a cf_role: "mesh_topology" variable,
    its info will be used, and its attributes will be overridden by whatever is specified
    in this call.

    Only the face_node_connectivity parameter is required.

    Example: If the variable for face_node_connectivity is named nv, the function call should look
    like this:

    ```
    ds = assign_ugrid_topology(ds, face_node_connectivity="nv")
    ```

    This will assign a new variable to the dataset called "mesh" with
    the face_node_connectivity attribute (and cf_role: "mesh_topology").
    It will also assign the node_coordinates attribute to the dataset with
    the lon and lat coordinate variable names, introspected from the
    face_node_connectivity variables' dimensions.

    You can also optionally specify any of the following mesh topology variables by passing
    them as keyword arguments
    - node_coordinates: If not specified, the function will introspect the dataset
                        for the longitude and latitude coordinate variable names using
                        cf_xarray
    - face_face_connectivity
    - face_coordinates: If not specified, the function will introspect the dataset
                        for the longitude and latitude coordinate variable names matching
                        the face_node_connectivity variable, but do nothing if they are
                        not found

    Required Args:
        ds (xr.Dataset): The dataset to assign the UGRID topology to

    Required if not in the Dataset mesh variable already:
        face_node_connectivity (str): The variable name of the face definitions

    Optional Args:
        face_node_connectivity: str,
        face_face_connectivity: str = None,
        boundary_node_connectivity: str = None,
        face_edge_connectivity: str = None,
        edge_node_connectivity: str = None,
        edge_face_connectivity: str = None,
        node_coordinates: str = None,
        face_coordinates: str = None,
        edge_coordinates: str = None,
        face_dimension: str = None,
        edge_dimension: str = None,
        start_index: int = None,

    (See the UGRID conventions for descriptions of these)

    You can pass a dict in with all the grid topology variables::

    ```
        grid_topology = {'node_coordinates': ('lon', 'lat'),
                         'face_node_connectivity': 'nv',
                         'node_coordinates': ('lon', 'lat'),
                         'face_coordinates': ('lonc', 'latc'),
                         }

        assign_ugrid_topology(ds, **grid_topology)

    ```
    """
    # check for an existing mesh variable
    try:
        mesh_vars = ds.cf.cf_roles["mesh_topology"]
    except KeyError:
        # It's not there: create one
        mesh_attrs = {"cf_role": "mesh_topology"}
        ds = ds.assign(mesh=((), np.int32(0), mesh_attrs))
        mesh_attrs = ds["mesh"].attrs
    else:
        if len(mesh_vars) > 1:
            raise ValueError(f"This dataset has more than one mesh_topology variable: {mesh_vars}")
        mesh_attrs = ds[mesh_vars[0]].attrs

    # Use a SimpleNamespace for the attrs
    # it's easier to access than a dict, and easier to update than locals
    mesh = SimpleNamespace(
        topology_dimension = np.int32(2),
        face_node_connectivity=None,
        face_face_connectivity=None,
        boundary_node_connectivity=None,
        face_edge_connectivity=None,
        edge_node_connectivity=None,
        edge_face_connectivity=None,
        node_coordinates=None,
        face_coordinates=None,
        edge_coordinates=None,
        face_dimension=None,
        edge_dimension=None,
        start_index=None,
        )
    # Add in the existing ones from the Dataset mesh object
    mesh.__dict__.update(mesh_attrs)

    # Add in the ones passed in:
    variables = vars()
    mesh.__dict__.update({att: variables[att]
                          for att in ALL_MESH_VARS
                          if variables[att] is not None})
    mesh.start_index = start_index

    if mesh.face_node_connectivity is None:
        raise ValueError(
            "face_node_connectivity is a required parameter if it is not in the mesh_topology variable"  # noqa: E501
        )
    if mesh.face_coordinates is None:
        try:
            face_coordinates = ds[mesh.face_node_connectivity].cf.coordinates
            mesh.face_coordinates = " ".join(f"{coord[0]}"
                                             for coord in face_coordinates.values())
        except AttributeError:
            mesh.face_coordinates = None

    if mesh.node_coordinates is None:
        try:
            if mesh.face_coordinates:
                filter = mesh.face_coordinates.split()
            else:
                filter = []

            coords = ds.cf.coordinates
            node_lon = [c for c in coords["longitude"] if c not in filter][0]
            node_lat = [c for c in coords["latitude"] if c not in filter][0]
            mesh.node_coordinates = f"{node_lon} {node_lat}"
        except AttributeError:
            raise ValueError(
                "The dataset does not have cf_compliant node coordinates longitude and latitude coordinates"  # noqa: E501
            )

    if mesh.edge_coordinates is None:
        try:
            mesh.edge_coordinates = ds[mesh.edge_node_connectivity].cf.coordinates
            mesh.edge_coordinates = [f"{coord[0]}" for coord in edge_coordinates.values()]
        except (KeyError, AttributeError):
            mesh.edge_coordinates = None

    if mesh.face_face_connectivity is None:
        # face_face_connectivity var should have same dimensions as
        # face_node_connectivity this is assuming that only one will match!
        for var_name, var in ds.variables.items():
            if var_name == mesh.face_node_connectivity:
                continue
            if var.dims == ds[mesh.face_node_connectivity].dims:
                mesh.face_face_connectivity = var_name
                break


    if mesh.face_dimension is None:
        # The face_dimension attribute specifies which netcdf dimension is used to
        # indicate the index of the face in the connectivity arrays. This is needed
        # because some applications store the data with the fastest varying index
        # first, and some with that index last. The default is to use the num_faces
        # as fastest dimension; e.g. a (num_faces, 3) array for triangles,
        # but some applications might use a (3, num_faces) order, in which case
        # the face_dimension attribute is required to help the client code disambiguate.
        dims = ds[mesh.face_node_connectivity].dims
        mapping = [(dim, ds[dim].size) for dim in dims]
        mesh.face_dimension = next(x[0] for x in mapping if x[1] != 3 and x[1] != 4)

    # check for start_index, and set it if there.
    if mesh.start_index is None:
        mesh.start_index = int(ds[mesh.face_node_connectivity].min())

    if mesh.start_index not in (0, 1):
        raise ValueError(f"start_index must be 0 or 1, not {mesh.start_index}")

    # assign the start_index to all the grid variables
    for var in (v for v in ALL_MESH_VARS if "connectivity" in v):
        var_name = getattr(mesh, var)
        if var_name:
            try:
                var = ds[var_name]
                var.attrs.setdefault("start_index", mesh.start_index)
            except KeyError:
                warnings.warn(f"{var_name} in mesh_topology variable, but not in dataset")

    # push the non-None mesh attributes back into the variable
    for key, val in mesh.__dict__.items():
        # can't use truthiness, as 0 is false
        if not ((val is None) or (val == "")):
            mesh_attrs[key] = val

    return ds
