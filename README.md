# xarray-subset-grid

Subset Xarray datasets in space while retaining the original grid for complex grid systems.

## Installation

This package is not yet released on pypi, for now install using git:

```
pip install xarray_subset_grid@git+https://github.com/asascience-open/xarray-subset-grid.git
```

Or better yet: clone the project from git and build / install it from there.


## Usage

*Coming Soon*

For now, see the [example notebooks](./examples/)

## API Design

The API for this package is very much a **work in progress**. 

Specifically, there are decisions still to be made around the most efficient way to deal with the xarray accessor: should the grid implementations hold references to the datasets or should they stay static and not require a reference to a given dataset so they can be reused without side effects. 

The grid implementations should also be exposed correctly so that they can be used without the xarray accessor as well.

## Use with conda

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

## pixi

Another option for using conda packages -- this project has been set up to use the pixi environment management system

https://prefix.dev/

To use:

See the pixi docs for details, but for this setup:

There are three "environments" set up for pixi:

- `default`
- `dev`
- `examples`

And one "task": `test`

To run the tests in an isolated environment:

```
pixi run -e dev test
```

To run a shell to do dev work:

```
pixi shell -e dev
```

That will set up a conda environment with all the develop dependencies.

To run a shell in which you can run the examples:

```
pixi shell -e examples
```
To run a shell with everything (dev and example deps:

```
pixi shell -e all
```






