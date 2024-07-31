.. _installation:

Installation
============

This installation guide includes only the xarray-subset-grid installation instructions. Please
refer to `xarray-subset-grid Contributor's Guide <https://uxarray.readthedocs.io/en/latest/contributing.html>`_
for detailed information about how to contribute to the UXarray project.

We recommend using the `conda <https://conda.io/docs/>`__ or
`pixi <https://prefix.dev/>`__ package managers for your Python
environments. Please take some time to go over the
`README <https://github.com/asascience-open/xarray-subset-grid/blob/main/README.md>`__.


Conda
-----

If you are using (or want to use) `Conda <https://conda.io/en/latest>`__, you can install the dependencies with:
::

    conda install --file conda_requirements.txt

That will get you the full set, including what you need to run the examples, etc.

If you need the development tools, you can also install:
::

    conda install --file conda_requirements_dev.txt

(requirements should all be on the conda-forge channel)

Pixi
----

Another option for using conda packages -- this project has been set up to use the `pixi <https://prefix.dev/>`__ environment management system.
To use:

See the pixi docs for details, but for this setup:

There are three "environments" set up for pixi:
--- default
--- dev
--- examples

To run the tests in an isolated environment:
::

    pixi run -e dev test

To run a shell to do dev work:
::

    pixi shell -e dev

That will set up a conda environment with all the develop dependencies.

To run a shell in which you can run the examples:
::

    pixi shell -e examples

To run a shell with everything (dev and example deps:
::

    pixi shell -e all

PyPI
----

An alternative to Conda is using pip
::

    pip install xarray_subset_grid@git+https://github.com/asascience-open/xarray-subset-grid.git
    .. pip install xarray-subset-grid


Source (Github)
---------------
Install `Git <https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>`__
on your system if not already available (check with ``git --version`` at the command line.)

If you are interested in installing xarray-subset-grid from source, 
you will first need to get the latest version of the code
::

    git clone https://github.com/asascience-open/xarray-subset-grid.git
    cd xarray-subset-grid

Run the following command from the root-directory
::

    pip install .

