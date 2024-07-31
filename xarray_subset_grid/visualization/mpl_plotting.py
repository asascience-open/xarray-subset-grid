"""Some MPL based plotting utilities for working with grids.

NOTE: this could probably be built on existing packages -- worth a look.
"""

from matplotlib.collections import LineCollection
from matplotlib.tri import Triangulation


def plot_ugrid(axes, ds, nodes=False, node_numbers=False, face_numbers=False):
    """Plot a UGRID in the provided MPL axes.

    Note: this doesn't plot data on the gird, just the grid itself

    :param axes: an MPL axes object to plot on

    :param ds: an xarray dataset with UGRID complinat grid in it.

    :param nodes: If True, plot the nodes as dots

    :param node_numbers=False: If True, plot the node numbers

    :param face_numbers=False: If True, plot the face numbers
    """

    mesh_defs = ds[ds.cf.cf_roles["mesh_topology"][0]].attrs
    lon_var, lat_var = mesh_defs["node_coordinates"].split()
    nodes_lon, nodes_lat = (ds[n] for n in mesh_defs["node_coordinates"].split())
    faces = ds[mesh_defs["face_node_connectivity"]]

    if faces.shape[0] == 3:
        # swap order for mpl triangulation
        faces = faces.T
    start_index = faces.attrs.get("start_index")
    start_index = 0 if start_index is None else start_index
    faces = faces - start_index

    mpl_tri = Triangulation(nodes_lon, nodes_lat, faces)

    axes.triplot(mpl_tri)
    if face_numbers:
        try:
            face_lon, face_lat = (ds[n] for n in mesh_defs["face_coordinates"].split())
        except KeyError:
            raise ValueError('"face_coordinates" must be defined to plot the face numbers')
        for i, point in enumerate(zip(face_lon, face_lat)):
            axes.annotate(
                f"{i}",
                point,
                xytext=(0, 0),
                textcoords="offset points",
                horizontalalignment="center",
                verticalalignment="center",
                bbox={
                    "facecolor": "white",
                    "alpha": 1.0,
                    "boxstyle": "round,pad=0.0",
                    "ec": "white",
                },
            )

    # plot nodes
    if nodes:
        axes.plot(nodes_lon, nodes_lat, "o")
    # plot node numbers
    if node_numbers:
        for i, point in enumerate(zip(nodes_lon, nodes_lat)):
            axes.annotate(
                f"{i}",
                point,
                xytext=(2, 2),
                textcoords="offset points",
                bbox={
                    "facecolor": "white",
                    "alpha": 1.0,
                    "boxstyle": "round,pad=0.0",
                    "ec": "white",
                },
            )

    # boundaries -- if they are there.
    if "boundary_node_connectivity" in mesh_defs:
        bounds = ds[mesh_defs["boundary_node_connectivity"]]

        lines = []
        for bound in bounds.data:
            line = (
                (nodes_lon[bound[0]], nodes_lat[bound[0]]),
                (nodes_lon[bound[1]], nodes_lat[bound[1]]),
            )
            lines.append(line)
        lc = LineCollection(lines, linewidths=2, colors=(1, 0, 0, 1))
        axes.add_collection(lc)
