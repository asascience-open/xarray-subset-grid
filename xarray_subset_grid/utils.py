import warnings

import cf_xarray  # noqa
import numpy as np


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
    _poly_x_min, poly_x_max = poly_x.min(), poly_x.max()

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
        idx = np.nonzero((y > min(p1y, p2y)) & (y <= max(p1y, p2y)) & (x <= max(p1x, p2x)))[0]
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


# This is defined in ugrid.py
# this placeholder for backwards compatibility for a brief period
def assign_ugrid_topology(*args, **kwargs):
    warnings.warn(
        DeprecationWarning,
        "The function `assign_grid_topology` has been moved to the "
        "`grids.ugrid` module. It will not be able to be called from "
        "the utils `module` in the future.",
    )
    from .grids.ugrid import assign_ugrid_topology

    return assign_ugrid_topology(*args, **kwargs)


def format_bytes(num):
    """
    This function will convert bytes to MB.... GB... etc
    from:
    https://stackoverflow.com/questions/12523586/python-format-size-application-converting-b-to-kb-mb-gb-tb
    """
    # not much to it, but handy for demos, etc.

    step_unit = 1024  # 1024 bad the size

    for x in ["bytes", "KB", "MB", "GB", "TB", "PB"]:
        if num < step_unit:
            return f"{num:3.1f} {x}"
        num /= step_unit
