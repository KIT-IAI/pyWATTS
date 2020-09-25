import matplotlib.pyplot as plt
import xarray as xr


def _recursive_plot(da: xr.DataArray, filemanager, name, title="Title"):
    figure = plt.Figure()
    figure.suptitle(title)
    if len(da.dims) > 1:
        for i in range(len(da[da.dims[1]])):
            _recursive_plot(da[:, i], filemanager=filemanager, name=f"{da.dims[0]}_{i}_{name}", title=title)
    else:
        da.plot()
        plt.savefig(filemanager.get_path(f'{name}.png'))
        plt.clf()

    plt.close(figure)
