import cf_xarray  # noqa
import numpy as np
import xarray as xr


def ray_tracing_numpy(x, y, poly):
    """Find vertices inside of the given polygon

    From: https://stackoverflow.com/a/57999874

    Args:
        x (np.array): x-coordinates of the vertices
        y (np.array): y-coordinates of the vertices
        poly (np.array): polygon vertices
    """
    n = len(poly)
    inside = np.zeros(len(x), np.bool_)
    p2x = 0.0
    p2y = 0.0
    xints = 0.0
    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        idx = np.nonzero(
            (y > min(p1y, p2y)) & (y <= max(p1y, p2y)) & (x <= max(p1x, p2x))
        )[0]
        if p1y != p2y:
            xints = (y[idx] - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
        if p1x == p2x:
            inside[idx] = ~inside[idx]
        else:
            idxx = idx[x[idx] <= xints]
            inside[idxx] = ~inside[idxx]

        p1x, p1y = p2x, p2y
    return inside


def assign_ugrid_topology(ds: xr.Dataset, **attrs) -> xr.Dataset:
    """Assign the UGRID topology to the dataset

    Only the face_node_connectivity attribute is required. The face_face_connectivity attribute is optional. If
    the variable for face_node_connectivity is named nv, the function call should look like this:

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
        **attrs: The attributes to assign to the datasets mesh topology metadata (see function description for more details)
    """
    # Get the variable name for the face_node_connectivity
    face_node_connectivity = attrs.get("face_node_connectivity", None)
    if face_node_connectivity is None:
        raise ValueError("The face_node_connectivity attribute is required")
    face_face_connectivity = attrs.get("face_face_connectivity", None)

    # Get the longitude and latitude coordinate variable names
    node_coords = attrs.get("node_coordinates", None)
    face_coords = attrs.get("face_coordinates", None)

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

    mesh_attrs = {
        "cf_role": "mesh_topology",
        "topology_dimension": 2,
        "node_coordinates": " ".join(node_coords),
        "face_node_connectivity": face_node_connectivity,
    }

    if face_coords:
        mesh_attrs["face_coordinates"] = " ".join(face_coords)
    if face_face_connectivity:
        mesh_attrs["face_face_connectivity"] = face_face_connectivity

    # Assign the mesh topology to the dataset
    ds = ds.assign(
        mesh=((), 0, mesh_attrs),
    )

    return ds
