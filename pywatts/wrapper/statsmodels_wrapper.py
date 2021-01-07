import pickle

import numpy as np
import statsmodels as sm
import xarray as xr

from pywatts.core.exceptions.kind_of_transform_does_not_exist_exception import KindOfTransformDoesNotExistException, \
    KindOfTransform
from pywatts.core.filemanager import FileManager
from pywatts.wrapper.base_wrapper import BaseWrapper


class StatsmodelsWrapper(BaseWrapper):
    """
        A wrapper class for statsmodels modules. Should only be used internally by the pipeline itself
        :param module: The statsmodels module to wrap
        :type module: statsmodels.base.BaseEstimator
        :param name: The name of the module
        :type name: str
        """

    # Alternative: sm.base.Model?
    def __init__(self, module: sm.tsa.base.TimeSeriesModel, name: str = None):
        if name is None:
            name = module.__class__.__name__
        super().__init__(name)
        self.module = module

    def fit(self, x: xr.Dataset, y: xr.Dataset = None):
        """
        Fit the sklearn module
        :param x: input data
        :param y: target data
        """
        x = self._dataset_to_statsmodels_input(x)
        y = self._dataset_to_statsmodels_input(y)
        self.module.fit(x, y)
        self.is_fitted = True

    @staticmethod
    def _dataset_to_statsmodels_input(x):
        if x is None:
            return None
        result = None
        for data_var in x.data_vars:
            data_array = x[data_var]
            if result is not None:
                result = np.concatenate([result, data_array.values.reshape((len(data_array.values), -1))], axis=1)
            else:
                result = data_array.values.reshape((len(data_array.values), -1))
        return result

    @staticmethod
    def _statsmodels_output_to_dataset(x: xr.Dataset, prediction, name: str) -> xr.Dataset:

        coords = (
            # first dimension is number of batches. We assume that this is the time.
            ("time", list(x.coords.values())[0].to_dataframe().index.array),
            *[(f"dim_{j}", list(range(size))) for j, size in enumerate(prediction.shape[1:])])

        data = {f"{name}": (tuple(map(lambda x: x[0], coords)), prediction),
                "time": list(x.coords.values())[0].to_dataframe().index.array}
        return xr.Dataset(data)

    def transform(self, x: xr.Dataset) -> xr.Dataset:
        def transform(self, x:xr.Dataset) -> xr.Dataset:
               x_np = self._dataset_to_statsmodels_input(x)
               prediction = self.module.transform(x_np)
               return self._statsmodels_output_to_dataset(x, prediction, self.name)
        Transforms a dataset or predicts the result with the wrapped statsmodels module
        :param x: the input dataset
        :return: the transformed output
        """
        x_np = self._dataset_to_statsmodels_input(x)

        if "transform" in dir(self.module):
            prediction = self.module.transform(x_np)
        elif "predict" in dir(self.module):
            prediction = self.module.predict(x_np)
        else:
            raise KindOfTransformDoesNotExistException(
                f"The statsmodels module in {self.name} does not have a predict or transform method",
                KindOfTransform.PREDICT_TRANSFORM)

        return self._statsmodels_output_to_dataset(x, prediction, self.name)

    def save(self, fm: FileManager):
        json = super().save(fm)
        file_path = fm.get_path(f'{self.name}.pickle')
        with open(file_path, 'wb') as outfile:
            pickle.dump(obj=self.module, file=outfile)
        json.update({"statsmodels_module": file_path})
        return json

    @classmethod
    def load(cls, load_information) -> 'StatsmodelsWrapper':
        """
        :param load_information: Information for reloading the StatsmodelsWrapper
        :type load_information: Dict
        :return: The reloaded StatsmodelsWrapper
        :rtype: StatsmodelsWrapper

        .. warning::
            This method use pickle for loading the module. Note that this is not safe.
            Consequently, load only modules you trust.
            For more details about pickling see https://docs.python.org/3/library/pickle.html
        """
        name = load_information["name"]
        with open(load_information["statsmodels_module"], 'rb') as pickle_file:
            module = pickle.load(pickle_file)
        module = cls(module=module, name=name)
        module.is_fitted = load_information["is_fitted"]
        return module
