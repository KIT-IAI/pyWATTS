import xarray as xr


def _xarray_summary(x: xr.DataArray, name="summary"):
    """
    Print out basic information of the xarray dataset.
    """
    print(x.to_dataframe(name).describe())
