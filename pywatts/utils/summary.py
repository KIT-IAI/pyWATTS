import xarray as xr


def _xarray_summary(x: xr.DataArray):
    """
    Print out basic information of the xarray dataset.
    """
    print(x.to_dataframe().describe())
