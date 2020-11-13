import pickle

import numpy as np
import sklearn
import xarray as xr
from sklearn.base import TransformerMixin

from pywatts.core.exceptions.kind_of_transform_does_not_exist_exception import KindOfTransformDoesNotExistException, \
    KindOfTransform
from pywatts.core.filemanager import FileManager
from pywatts.wrapper.base_wrapper import BaseWrapper


class SKLearnWrapper(BaseWrapper):
    """
    A wrapper class for sklearn modules. Should only used internal by the pipeline itself
    :param module: The sklearn module to wrap
    :type module: sklearn.base.BaseEstimator
    :param name: The name of the module
    :type name: str
    """

    def __init__(self, module: sklearn.base.BaseEstimator, name: str = None):
        if name is None:
            name = module.__class__.__name__
        super().__init__(name)
        self.module = module

        if hasattr(self.module, 'inverse_transform'):
            self.has_inverse_transform = True

        if hasattr(self.module, 'predict_proba'):
            self.has_predict_proba = True

    def get_params(self):
        """
        Return the parameter of the slkearn module
        :return:
        """
        return self.module.get_params()

    def set_params(self, **kwargs):
        """
        Set the parameter of the internal sklearn module
        :param kwargs: The parameter of the internal sklearn module
        :return:
        """
        return self.module.set_params(**kwargs)

    def fit(self, x: xr.Dataset, y: xr.Dataset = None):
        """
        Fit the sklearn module
        :param x: input data
        :param y: target data
        """
        x = self._dataset_to_sklearn_input(x)
        y = self._dataset_to_sklearn_input(y)
        self.module.fit(x, y)
        self.is_fitted = True

    @staticmethod
    def _dataset_to_sklearn_input(x):
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
    def _sklearn_output_to_dataset(x: xr.Dataset, prediction, name: str) -> xr.Dataset:

        coords = (
            # first dimension is number of batches. We assume that this is the time.
            ("time", list(x.coords.values())[0].to_dataframe().index.array),
            *[(f"dim_{j}", list(range(size))) for j, size in enumerate(prediction.shape[1:])])

        data = {f"{name}": (tuple(map(lambda x: x[0], coords)), prediction),
                "time": list(x.coords.values())[0].to_dataframe().index.array}
        return xr.Dataset(data)

    def transform(self, x: xr.Dataset) -> xr.Dataset:
        """
        Transforms a dataset or predicts the result with the wrapped sklearn module
        :param x: the input dataset
        :return: the transformed output
        """
        x_np = self._dataset_to_sklearn_input(x)

        if isinstance(self.module, TransformerMixin):
            prediction = self.module.transform(x_np)
        elif "predict" in dir(self.module):
            prediction = self.module.predict(x_np)
        else:
            raise KindOfTransformDoesNotExistException(
                f"The sklearn-module in {self.name} does not have a predict or transform method",
                KindOfTransform.PREDICT_TRANSFORM)

        return self._sklearn_output_to_dataset(x, prediction, self.name)

    def inverse_transform(self, x: xr.Dataset) -> xr.Dataset:
        """
        Performs the inverse transform of a dataset with the wrapped sklearn module
        :param x: the input dataset
        :return: the transformed output
        """
        x_np = self._dataset_to_sklearn_input(x)
        if self.has_inverse_transform:
            prediction = self.module.inverse_transform(x_np)
        else:
            raise KindOfTransformDoesNotExistException(
                f"The sklearn-module in {self.name} does not have a inverse transform method",
                KindOfTransform.INVERSE_TRANSFORM)

        return self._sklearn_output_to_dataset(x, prediction, self.name)

    def predict_proba(self, x: xr.Dataset) -> xr.Dataset:
        """
        Performs the probabilistic transform of a dataset with the wrapped sklearn module
        :param x: the input dataset
        :return: the transformed output
        """
        x_np = self._dataset_to_sklearn_input(x)
        if self.has_predict_proba:
            prediction = self.module.predict_proba(x_np)
        else:
            raise KindOfTransformDoesNotExistException(
                f"The sklearn-module in {self.name} does not have a predict_proba method",
                KindOfTransform.PROBABILISTIC_TRANSFORM)

        return self._sklearn_output_to_dataset(x, prediction, self.name)

    def save(self, fm: FileManager):
        json = super().save(fm)
        file_path = fm.get_path(f'{self.name}.pickle')
        with open(file_path, 'wb') as outfile:
            pickle.dump(obj=self.module, file=outfile)
        json.update({"sklearn_module": file_path})
        return json

    @classmethod
    def load(cls, load_information) -> 'SKLearnWrapper':
        """
        :param load_information: Information for reloading the SklearnWrapper
        :type load_information: Dict
        :return: The reloaded SklearnWrapper
        :rtype: SKLearnWrapper

        .. warning::
            This method use pickle for loading the module. Note that this is not safe.
            Consequently, load only modules you trust.
            For more details about pickling see https://docs.python.org/3/library/pickle.html
        """
        name = load_information["name"]
        with open(load_information["sklearn_module"], 'rb') as pickle_file:
            module = pickle.load(pickle_file)
        module = cls(module=module, name=name)
        module.is_fitted = load_information["is_fitted"]
        return module