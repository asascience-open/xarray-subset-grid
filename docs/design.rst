.. _design:

API Design
==========

The API for this package is very much a work in progress.

Specifically, there are decisions still to be made around the most efficient way to deal with the xarray accessor: should the grid implementations hold references to the datasets or should they stay static and not require a reference to a given dataset so they can be reused without side effects.

The grid implementations should also be exposed correctly so that they can be used without the xarray accessor as well.


accessor
--------

.. automodule:: xarray_subset_grid.accessor
   :members:
   :undoc-members:
   :show-inheritance:

grid
----

.. automodule:: xarray_subset_grid.grid
   :members:
   :undoc-members:
   :show-inheritance:

utils
-----

.. automodule:: xarray_subset_grid.utils
   :members:
   :undoc-members:
   :show-inheritance:

