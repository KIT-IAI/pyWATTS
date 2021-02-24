from typing import Dict

import xarray as xr


def _xarray_summary(x: Dict[str, xr.DataArray], name="summary"):
    """
    Print out basic information of the xarray dataset.
    """
    for key, da in x.items():
        print(f"======== Summary for {key} ============")
        print(da.to_dataframe(name).describe())
