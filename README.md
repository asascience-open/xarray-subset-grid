# xarray-subset-grid

Subset Xarray datasets in space while retaining the original grid for complex grid systems.

## Installation

### `pip compatible`

This package is not yet released on pypi, for now install using git:

```
pip install xarray_subset_grid@git+https://github.com/asascience-open/xarray-subset-grid.git
```

Or clone the project from git and build / install it from there.

## Usage

This package is designed to be used in conjuction with [`xarray`](https://xarray.dev/). Given a [CF Compliant](https://cfconventions.org/) `xarray` dataset named `ds`, this package can be accessed using the `xsg` accessor:

```python
# Get the interprested grid class
grid = ds.xsg

# subset to only include temperature
ds_temp = ds.xsg.subset_vars(['temp'])

# subset by bounding box
ds_subset_bbox = ds.xsg.subset_bbox([-72, 32, -70, 35])

# or by polygon
poly = np.array([
    [-72, 32],
    [-72, 33], 
    [-73, 33], 
    [-73, 31], 
    [-72, 32],
])
ds_subset_poly = ds.xsg.subset_polygon(poly)
```

For full usage, see the [example notebooks](./examples/)

## Development

### `pip compatible`

First, create a new `virtualenv` and activate it:

```bash
python -m venv venv
source venv/bin.activate
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

And two "tasks": 

- `lint`
- `test`

To run the tests in an isolated environment:

```bash
pixi run -e dev test
```

To run a shell to do dev work:

```bash
pixi shell -e dev
```

That will set up a conda environment with all the develop dependencies.

To run a shell in which you can run the examples:

```bash
pixi shell -e examples
```
To run a shell with everything (dev and example deps:

```bash
pixi shell -e all
```

Finally, to when the `pyproject.toml` is updated, be sure to update the `pixi` lockfile:

```bash
pixi install
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