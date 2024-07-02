'''
Performance testing for point in polygon code

Conclusion:

Cython version is 10x faster for 1000 pts
Numpy version (in this project) is a touch faster for 200,000 pts.

NECOFS is ~ 200,000 k nodes

(for 20 point polygon)

So: for the big ones where it matters, numpy version is fine.

-- all versions gave the same result for a random test set.


Options tested:

Custom code in this project::

    def ray_tracing_numpy(x, y, poly):
        """Find vertices inside of the given polygon

        From: https://stackoverflow.com/a/57999874
        """

The code from Shapely:
https://shapely.readthedocs.io/en/stable/reference/shapely.contains_xy.html#shapely-contains-xy

Downside -- it's a big dependency!

The geometry_utils package:
https://github.com/NOAA-ORR-ERD/geometry_utils

This is a package I (Chris Barker) wrote to provide fast, simple computational geometry routines.
Downside: It's compiled code (Cython) and It's not properly packaged up, but could be if it
          proves useful.

This test code requires:
shapely
geometry_utils
'''

import numpy as np

# Using a 20 vertex polygon -- arbitrary decision as common max for
# bounding poly used for subsetting.
test_poly = np.array([
    [-69.0842494, 41.8576263],
    [-69.3834133, 41.6994390],
    [-69.4844079, 41.5818408],
    [-69.7009389, 41.5498641],
    [-70.0628678, 41.5884718],
    [-70.3054548, 41.6810850],
    [-70.6109682, 41.7607248],
    [-70.8657576, 41.9553727],
    [-71.1089099, 42.1369069],
    [-71.1294295, 42.4274792],
    [-70.8877302, 42.6500898],
    [-70.7118900, 42.7635708],
    [-70.4645152, 42.8363260],
    [-70.1066827, 42.8113145],
    [-69.9021696, 42.7796958],
    [-69.7686684, 42.7210923],
    [-69.4055325, 42.5535379],
    [-69.1527168, 42.3072355],
    [-68.9597074, 42.0243090],
    [-68.9939291, 41.9264228],
])

min_lon, min_lat = test_poly.min(axis=0)
max_lon, max_lat = test_poly.max(axis=0)

num_points = 200_000
test_lon = np.random.uniform(min_lon, max_lon, num_points)
test_lat = np.random.uniform(min_lat, max_lat, num_points)
test_points = np.c_[test_lon, test_lat]

from xarray_subset_grid.utils import ray_tracing_numpy

rtn_inside = ray_tracing_numpy(test_lon, test_lat, test_poly)
# 1000 pts
# In [13]: %timeit ray_tracing_numpy(test_lon, test_lat, test_poly)
# 344 µs ± 16.2 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
# 200,000 pts
# In [25]: %timeit ray_tracing_numpy(test_lon, test_lat, test_poly)
# 9.03 ms ± 303 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)



from geometry_utils import polygon_inside

geom = polygon_inside(test_poly, test_points)
# 1000 pts
# In [11]: %timeit polygon_inside(test_poly, test_points)
# 37.2 µs ± 1.25 µs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
# 200,000 pts
# In [26]: %timeit polygon_inside(test_poly, test_points)
# 10.1 ms ± 456 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

# Do they give the same answer?
assert np.array_equal(rtn_inside, geom)

from shapely import Polygon, contains_xy

sh_poly = Polygon(test_poly)

shapely = contains_xy(sh_poly, test_points)
# 1000 pts
# In [19]: %timeit shapely = contains_xy(sh_poly, test_points)
# 288 µs ± 9.48 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
# 200,000 pts
# %timeit shapely = contains_xy(sh_poly, test_points)
# 54.7 ms ± 1.4 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)



assert np.array_equal(rtn_inside, shapely)


