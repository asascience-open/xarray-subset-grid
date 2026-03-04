import nbformat as nbf

nb = nbf.v4.new_notebook()

# Proper kernel metadata so nbconvert knows to write .py not .txt
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.11.0",
    },
}

nb.cells = [
    nbf.v4.new_markdown_cell(
        "# Local Subset Example\n\n"
        "Demonstrates subsetting using a bundled local NetCDF file.\n"
        "No network access required."
    ),
    nbf.v4.new_code_cell(
        "import xarray as xr\n"
        "import numpy as np\n"
        "from pathlib import Path\n"
        "import xarray_subset_grid  # noqa: F401\n\n"
        "data_path = Path('../../tests/example_data') / 'AMSEAS-subset.nc'\n"
        "ds = xr.open_dataset(data_path)\n"
        "print('Dataset loaded:', ds)"
    ),
    nbf.v4.new_code_cell(
        "print('Grid type:', ds.xsg.grid)\n"
        "print('Grid vars:', ds.xsg.grid_vars)\n"
        "print('Data vars:', ds.xsg.data_vars)"
    ),
    nbf.v4.new_code_cell(
        "lats = ds.cf['latitude'].values\n"
        "lons = ds.cf['longitude'].values\n"
        "lat_mid = (float(lats.min()) + float(lats.max())) / 2\n"
        "lon_mid = (float(lons.min()) + float(lons.max())) / 2\n"
        "bbox = (float(lons.min()), float(lats.min()), lon_mid, lat_mid)\n"
        "ds_sub = ds.xsg.subset_bbox(bbox)\n"
        "print('Subsetted dataset:', ds_sub)\n"
        "print('Done.')"
    ),
]

nbf.write(nb, "docs/examples/local_subset_example.ipynb")
print("Notebook created successfully.")