from typing import Dict

import matplotlib.pyplot as plt
import tikzplotlib
import xarray as xr
from pywatts.core.base_summary import BaseSummary
from sklearn.manifold import TSNE
import numpy as np

from pywatts.core.summary_object import SummaryObjectList


class TSNESummary(BaseSummary):
    """
    Summary that performs a TSNE to visualise the data. It is possible to specify mask by providing kwargs
    to the transform method that end with _masked.
    :param max_point: The maximum number of points per data that should be plotted
    :type max_point: int
    :param all_in_one_plot: Flag indicating if all input data should be visualised in the same plot. If not the column
                            GT is visualised with all other in separate plots.
    :type all_in_one_plot: Boolean
    :param tsne_params: Params for the TSNE Visualisation. See sklearn.
    :type tsne_params: Dict
    """

    def __init__(self, name="TSNE",
                 max_points=10000, all_in_one_plot=False, tsne_params=None):
        super().__init__(name)
        self.tsne_params = {} if tsne_params is None else tsne_params
        self.max_points = max_points
        self.all_in_one_plot = all_in_one_plot

    def get_params(self) -> Dict[str, object]:
        """"
        Get the params of the TSNE Summary
        :return: A dict containing all parameter of the tsne_summary
        :rtype: Dict
        """
        return {
            "tsne_params": self.tsne_params,
            "max_points": self.max_points,
            "all_in_one_plot": self.all_in_one_plot
        }

    def set_params(self, max_points=None, all_in_one_plot=None, tsne_params=None):
        """
        Set the params of the tsne summary.
        :param max_point: The maximum number of points per data that should be plotted
        :type max_point: int
        :param all_in_one_plot: Flag indicating if all input data should be visualised in the same plot. If not the column
                                GT is visualised with all other in separate plots.
        :type all_in_one_plot: Boolean
        :param tsne_params: Params for the TSNE Visualisation. See sklearn.
        :type tsne_params: Dict
        """
        if max_points is not None:
            self.max_points = max_points
        if all_in_one_plot is not None:
            self.all_in_one_plot = all_in_one_plot
        if tsne_params is not None:
            self.tsne_params = tsne_params

    def transform(self, file_manager, gt: xr.DataArray, **kwargs: xr.DataArray):
        """
        Calculates the TSNE and visualise it. kwargs that end with _masked are masks for the other input. E.g. gt_masked
        is a mask for gt.
        :param file_manager: The filemanager, it can be used to store data that corresponds to the summary as a file.
        :type: file_manager: FileManager
        :param gt: the gt dataset
        :type gt: xr.DataArray
        :param kwargs: the predictions
        :type kwargs: xr.DataArray

        :return: A SummaryObjectList with the paths to all plots.
        :rtype: SummaryObjectList
        """
        gt_data, kwargs_data = self._get_selected_values(gt, kwargs)
        summary = SummaryObjectList(self.name)

        if self.all_in_one_plot:
            path_png, path_tex = self._train_and_plot(file_manager, gt_data, kwargs_data)
            summary.set_kv("Path PNG", f"{path_png}\n")
            summary.set_kv("Path Tex", f"{path_tex}\n")
        else:
            for key, value in kwargs_data.items():
                path_png, path_tex = self._train_and_plot(file_manager, gt_data, {key: value}, suffix=key)
                summary.set_kv(f"Path PNG {key}", f"{path_png}\n")
                summary.set_kv(f"Path Tex {key}", f"{path_tex}\n")
        return summary

    def _train_and_plot(self, file_manager, gt_data, kwargs_data, suffix=None):
        tsne = TSNE(**self.tsne_params)
        data_final = [gt_data]
        data_final.extend(kwargs_data.values())
        result = tsne.fit_transform(np.concatenate(data_final))
        plt.scatter(result[:len(gt_data), 0], result[:len(gt_data), 1], alpha=0.5, label="Ground Truth")
        for i, key in enumerate(kwargs_data, 1):
            plt.scatter(result[len(gt_data) * i:len(gt_data) * (i + 1), 0],
                        result[len(gt_data) * i:len(gt_data) * (i + 1), 1], alpha=0.5,
                        label=key)
        plt.xlabel("x-tsne")
        plt.ylabel("y-tsne")
        plt.legend()
        path_png = file_manager.get_path(f"tsne/{self.name}{'_' + suffix if suffix is not None else ''}.png")
        path_tex = file_manager.get_path(f"tsne/{self.name}{'_' + suffix if suffix is not None else ''}.tex")
        plt.savefig(path_png)
        tikzplotlib.save(path_tex)
        plt.close()
        return path_png, path_tex

    def _get_selected_values(self, gt, kwargs):
        if "gt_masked" in kwargs:
            gt_mask = kwargs["gt_masked"].values
            gt_data = gt.values[gt_mask]
        else:
            gt_data = gt.values
        gt_data = gt_data[~np.any(np.isnan(gt_data), axis=1)]
        kwargs_data = {}
        for name in filter(lambda x: not x.endswith("_masked"), kwargs):
            synth_data = kwargs[name].values
            if name + "_masked" in kwargs:
                synth_mask = kwargs[name + "_masked"].values
                kwargs_data[name] = synth_data[synth_mask]
            else:
                kwargs_data[name] = synth_data
            kwargs_data[name] = kwargs_data[name][~np.any(np.isnan(kwargs_data[name]), axis=1)]

        max_points = min(self.max_points, len(gt_data), *list(map(lambda x: len(x), kwargs_data.values())))
        min_length = min(len(gt_data), *list(map(lambda x: len(x), kwargs_data.values())))
        rdx = np.random.choice(range(0, min_length), max_points, replace=False)

        return gt_data[rdx], {
            name: data[rdx] for name, data in kwargs_data.items()
        }
