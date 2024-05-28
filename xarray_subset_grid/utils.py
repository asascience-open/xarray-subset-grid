import cf_xarray  # noqa
import numpy as np
import xarray as xr

def normalize_polygon_x_coords(x, poly):
    """
    Normalize the polygon x coordinates (longitude) to the
    same coord system as used by given x coordinates.

    e.g. If the longitude values are between 0 and 360, we need to normalize
    the polygon x coordinates to be between 0 and 360. Vice versa if the
    longitude values are between -180 and 180.

    If the x coords are between 0 and 180 (i.e. both will work), the polygon
    is not changed.

    NOTE: polygon is normalized in place!

    Args:
        x (np.array): x-coordinates of the vertices
        poly (np.array): polygon vertices
    """
    x_min, x_max = x.min(), x.max()

    poly_x = poly[:, 0]
    poly_x_min, poly_x_max = poly_x.min(), poly_x.max()

    if x_max > 180 and poly_x_max < 0:
        poly_x[poly_x < 0] += 360
    elif x_min < 0 and poly_x_max > 180:
        poly_x[poly_x > 180] -= 360

    poly[:, 0] = poly_x
    return poly

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
        if len(idx):
            if p1y != p2y:
                xints = (y[idx] - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
            if p1x == p2x:
                inside[idx] = ~inside[idx]
            else:
                idxx = idx[x[idx] <= xints]
                inside[idxx] = ~inside[idxx]

        p1x, p1y = p2x, p2y
    return inside


# This should probably be defined in ugrid.py -- but importing it there
#  for now.
def assign_ugrid_topology(ds: xr.Dataset,
                          *,
                          face_node_connectivity: str,
                          face_face_connectivity: str = None,
                          node_coordinates: str = None,
                          face_coordinates: str = None,
                          ) -> xr.Dataset:
    """
    Assign the UGRID topology to the dataset

    Only the face_node_connectivity parameter is required. The face_face_connectivity attribute is optional.
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


def convert_bytes(num):
    """
    This function will convert bytes to MB.... GB... etc
    from:
    https://stackoverflow.com/questions/12523586/python-format-size-application-converting-b-to-kb-mb-gb-tb
    """
    # stupid, but handy for demos, etc.

    step_unit = 1024 #1024 bad the size

    for x in ['bytes', 'KB', 'MB', 'GB', 'TB', 'PB']:
        if num < step_unit:
            return "%3.1f %s" % (num, x)
        num /= step_unit

