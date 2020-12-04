import xarray as xr


def _xarray_summary(x: xr.DataArray):
    """
    Print out basic information of the xarray dataset.
    """
    # TODO change test
    print(x.to_dataframe("test").describe())
