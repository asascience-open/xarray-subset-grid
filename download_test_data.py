import zipfile
from pathlib import Path

import pooch


def download_data():
    """Download support data for tests and documentation."""
    url = "https://github.com/ioos/xarray-subset-grid/releases/download"
    version = "2026.02.09"

    fname = pooch.retrieve(
        url=f"{url}/{version}/data_files.zip",
        known_hash="sha256:675fb74b9a8c58a6b3bdcef691f7e5bf97a11a3f1065668525e5169f8be20343",
    )

    here = Path(__file__).resolve().parent
    print(fname)
    print(here)
    with zipfile.ZipFile(fname, "r") as zip_ref:
        zip_ref.extractall(here)


if __name__ == "__main__":
    download_data()
