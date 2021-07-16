from typing import Union, Optional, Dict, List

import numpy as np
import xarray as xr

from pywatts.core.base import BaseTransformer
from pywatts.core.exceptions.wrong_parameter_exception import WrongParameterException
from pywatts.utils._xarray_time_series_utils import _get_time_indexes


class Differentiate(BaseTransformer):
    """
    Differentiation step to calculate the n-th order difference of a time series.
    By default the difference has not the same size as the input time series but
    padding is implemented by np.pad and specific arguments can be passed by pad_args.
    """

    def __init__(self, target_index: Union[str, List[str]] = None, name: str = "Differentiate",
                 n: Union[int, List[int]] = 1, axis: int = -1,
                 pad: bool = False, pad_args: Dict[str, object] = {}):
        """ Initialize the an differentiate processing step.

        :param target_index: Targed index or indizes for the xarray input
                             to calculate difference for.
        :type target_index: Union[str, List[str]]
        :param name: Name of this processing step (default 'Differentiate').
        :type name: str, optional
        :param n: N-th order difference specification (default 1).
                  Could also be an array if multiple differences should be calculated.
        :type n: Union[int, List[int]], optional
        :param axis: Axis to calculate the difference of (default -1 [last axis]).
        :type axis: int, optional
        :param pad: Enable or disable padding (default disabled).
        :type pad: bool, optional
        :param pad_args: Padding arguments for np.pad method (default zero leading padding).
        :type pad_args: Dict[str, object], optional
        """
        super().__init__(name)
        self.target_index = target_index
        self.n = n
        self.axis = axis
        self.pad = pad
        self.pad_args = pad_args

    def get_params(self) -> Dict[str, object]:
        """ Get parameters for Differentiate object.

        :return: Parameters as dict object.
        :rtype: Dict[str, object]
        """
        return {
            "target_index": self.target_index,
            "n": self.n,
            "axis": self.axis,
            "pad": self.pad,
            "pad_args": self.pad_args
        }

    def set_params(self, target_index: Optional[Union[str, List[str]]] = None,
                   n: Optional[Union[int, List[int]]] = None, axis: Optional[int] = None,
                   pad: Optional[bool] = None, pad_args: Optional[Dict[str, object]] = None):
        """ Set or change Differentiate object parameters.

        :param target_index: Targed index or indizes for the xarray input
                             to calculate difference for.
        :type target_index: Optional[Union[str, List[str]]], optional
        :param n: N-th order difference specification (default 1).
                  Could also be an array if multiple differences should be calculated.
        :type n: Optional[Union[int, List[int]]], optional
        :param axis: Axis to calculate the difference of (default -1 [last axis]).
        :type axis: Optional[int], optional
        :param pad: Enable or disable padding (default disabled).
        :type pad: Optional[bool], optional
        :param pad_args: Padding arguments for np.pad method (default zero leading padding).
        :type pad_args: Optional[Dict[str, object]], optional
        """
        if target_index is not None:
            self.target_index = target_index
        if n is not None:
            self.n = n
        if axis is not None:
            self.axis = axis
        if pad is not None:
            self.pad = pad
        if pad_args is not None:
            self.pad_args = pad_args

    def transform(self, x: xr.DataArray) -> xr.DataArray:
        """ Add n-th order differentiate to xarray dataset.

        :param x: Xarray dataset to apply differentiation on.
        :type x: xr.DataArray
        :return: Xarray dataset containing the n-th order differentiations.
        :rtype: xr.DataArray
        """
        # check parameters for non list types and make it a list
        if isinstance(self.n, int):
            ns = [self.n]
        else:
            ns = self.n

        if self.target_index is None:
            idxs = _get_time_indexes(x)
        elif isinstance(self.target_index, str):
            idxs = [self.target_index]
        else:
            idxs = self.target_index

        # check if idxs are valid idxs of the dataset
        for idx in idxs:
            if idx not in x:
                raise WrongParameterException(f"Index {idx} not in dataset!",
                                              "Assert that the previous modules provide the correct index.",
                                              module=self.name)

        # iterate over xarray indizes and n-th orders
        # and apply the differentiation on xarray dataset
        for idx in idxs:
            for n in ns:
                diff = np.diff(x[idx], n=n, axis=self.axis)

                # dims needed for multidim DataArray initialization
                # otherwise will lead to conflicts when dim_0 already set
                dims = list(x[idx].dims)

                if self.pad:
                    # pad if padding is enabled by using np.pad
                    # and correct padding widths for dimensions
                    original_size = x[idx].shape[self.axis]
                    pad_width = [(0, 0) for _ in range(len(diff.shape))]
                    pad_width[self.axis] = (original_size - diff.shape[self.axis], 0)
                    diff = np.pad(diff, pad_width=pad_width, **self.pad_args)
                else:
                    # if differentiate is not padded the dims of the differences aren't
                    # the same as before. So, we need to change dim[axis] name.
                    dims[self.axis] = f"{dims[self.axis]}_d{n}"

                # finally, add difference to xarray dataset
                x[f"{idx}_d{n}"] = xr.DataArray(diff, dims=dims)

        return x
