# xarray-subset-grid

Subset Xarray datasets in space while retaining the original grid for complex grid systems.

## Installation

### `pip`

This package is available on pypi:

```
python -m pip install xarray-subset-grid
```

### conda / pixi

This package is available on conda-forge:

```
conda install -c conda-forge xarray-subset-grid
```

## Usage

This package is designed to be used in conjunction with [`xarray`](https://xarray.dev/).
Given a [CF Compliant](https://cfconventions.org/) `xarray` dataset named `ds`, this package can be accessed using the `xsg` accessor:

```python
import numpy as np

# xarray_subset_gris should detect the grid type.
# To check what it found:
grid = ds.xsg

# subsetting to include only certain variables:
# Only temperature
ds_temp = ds.xsg.subset_vars(["temp"])

# subset by bounding box
ds_subset_bbox = ds.xsg.subset_bbox([-72, 32, -70, 35])

# or by polygon
poly = np.array(
    [
        [-72, 32],
        [-72, 33],
        [-73, 33],
        [-73, 31],
        [-72, 32],
    ]
)
ds_subset_poly = ds.xsg.subset_polygon(poly)
```

For full usage, see the [example notebooks](https://github.com/ioos/xarray-subset-grid/tree/main/docs/examples)
and the [Sphinx documentation on Read the Docs](https://xarray-subset-grid.readthedocs.io/).

## Development

### `pip`

First, create a new `virtualenv` and activate it:

```bash
python -m venv venv
source venv/bin/activate  # Linux and macOS
# venv\Scripts\activate   # Windows cmd
# venv\Scripts\Activate.ps1  # Windows PowerShell
```

Then install the project in local edit mode:

```bash
pip install -e .
```

Once installed, the tests can be run:

```bash
python -m pytest
```

Or alternatively run the notebooks in the same `virtualenv`

### `pixi`


Learn about `pixi` here: https://prefix.dev/

See the pixi docs for details, but for this setup:

There are three "environments" set up for pixi:

- `default`
- `dev`
- `examples`

And three "tasks":

- `lint`
- `test` : run most of the tests
- `test_all` : run the tests that access AWS -- i.e. download data directly.

To run the tests in an isolated environment:

```bash
pixi run -e dev test
```

Or with a specific python version:
```bash
pixi run -e test312 test
```

Options are: `test310` `test311` `test312` `test313`


To run a shell to do development work:

```bash
pixi shell -e dev
```

To run a shell in which you can run the examples
(notebooks and all that):

```bash
pixi shell -e examples
```

To run a shell with everything (dev and example deps:

```bash
pixi shell -e all
```

Finally, when the `pixi.toml` file is updated,
be sure to update the `pixi` lockfile:

```bash
pixi update
```

### `conda`

If you are using (or want to use) conda, you can install the dependencies with:

```
conda install --file conda_requirements.txt
```

That will get you the full set, including what you need to run the examples, etc.

If you need the development tools, you can also install:

```
conda install --file conda_requirements_dev.txt
```

(requirements should all be on the conda-forge channel)
