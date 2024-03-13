# xarray-subset-grid

Subset Xarray datasets in space while retaining the original grid for complex grid systems.

## Installation

This package is not yet released on pypi, for now install using git:

```
pip install xarray_subset_grid@git+https://github.com/asascience-open/xarray-subset-grid.git
```

## Usage

*Coming Soon*

For now, see the [example notebooks](./examples/)

## API Design

The API for this package is very much a **work in progress**. 

Specifically, there are decisions still to be made around the most efficient way to deal with the xarray accessor: should the grid implementations hold references to the datasets or should they stay static and not require a reference to a given dataset so they can be reused without side effects. 

The grid implementations should also be exposed correctly so that they can be used without the xarray accessor as well. 
