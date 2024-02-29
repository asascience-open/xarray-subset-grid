import numpy as np


def ray_tracing_numpy(x,y,poly):
    """Find vertices inside of the given polygon

    From: https://stackoverflow.com/a/57999874

    Args:
        x (np.array): x-coordinates of the vertices
        y (np.array): y-coordinates of the vertices
        poly (np.array): polygon vertices
    """
    n = len(poly)
    inside = np.zeros(len(x),np.bool_)
    p2x = 0.0
    p2y = 0.0
    xints = 0.0
    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        idx = np.nonzero((y > min(p1y,p2y)) & (y <= max(p1y,p2y)) & (x <= max(p1x,p2x)))[0]
        if p1y != p2y:
            xints = (y[idx]-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
        if p1x == p2x:
            inside[idx] = ~inside[idx]
        else:
            idxx = idx[x[idx] <= xints]
            inside[idxx] = ~inside[idxx]

        p1x,p1y = p2x,p2y
    return inside
