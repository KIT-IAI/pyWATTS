from typing import Dict

from pywatts.core.exceptions.wrong_parameter_exception import WrongParameterException
from pywatts.core.base import BaseTransformer
import xarray as xr


class WhiteLister(BaseTransformer):
    """
    Module to filter/ select variables from a given dataset

    :param target: Target variable which should be filtered/ selected
    :param name: Name of the WhiteLister
    """

    def get_params(self) -> Dict[str, object]:
        """
        Returns the parameter

        :return: Dict containing the target
        :rtype: Dict[str, object]
        """
        return {"target": self.target}

    def set_params(self, target):
        """
        Set the target column which should be filtered/ selected.

        :param target: The target column which should be filtered/ selected
        :type target: str
        """
        self.target = target

    def __init__(self, target, name="WhiteLister"):
        super().__init__(name)
        self.target = target

    def transform(self, x: xr.Dataset) -> xr.Dataset:
        """
        Filter the dataset according to the target variable

        :param x: Dataset which should be filtered
        :type x: xr.Dataset
        :return: the filtered dataset
        :rtype: xr.Dataset
        """
        if not self.target in x.data_vars.keys():
            # delete appended "_<element_id>
            raise WrongParameterException(f"{self.target} is not part of the data_vars: {list(x.data_vars.keys())}.",
                                          f"Either change self.target to one existing data_var or assert that the previous modules provide this data_var",
                                          module=self.name)
        return xr.Dataset({self.target: x[self.target]}, coords=x.coords)
