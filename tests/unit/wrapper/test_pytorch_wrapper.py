import unittest

import pytest
import xarray as xr
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler

from pywatts.wrapper.pytorch_wrapper import PyTorchWrapper


class TestPyTorchWrapper(unittest.TestCase):

    def get_sequential_model(self, d_in=5, d_out=1, h=10):
        # d_in is input dimension
        # h is hidden dimension
        # d_out is output dimension

        model = torch.nn.Sequential(
            torch.nn.Linear(d_in, h),
            torch.nn.ReLU(),
            torch.nn.Linear(h, d_out),
        )

        return model

    def test_set_params(self):
        model = self.get_sequential_model()
        wrapper = PyTorchWrapper(model=model,
                                 fit_kwargs={"batch_size": 8, "epochs": 1},
                                 compile_kwargs={"loss": "mse", "optimizer": "Adam", "metrics": ["mse"]})

        self.assertEqual(wrapper.get_params()["fit_kwargs"]["batch_size"], 8)
        wrapper.set_params(fit_kwargs={"batch_size": 12, "epochs": 1}, )
        self.assertEqual(wrapper.get_params()["fit_kwargs"]["batch_size"], 12)

    def test_fit_PyTorchModel(self):
        model = self.get_sequential_model(d_in=1, d_out=1, h=5)
        wrapper = PyTorchWrapper(model=model,
                                 fit_kwargs={"batch_size": 8, "epochs": 1},
                                 compile_kwargs={"loss": "mse", "optimizer": "Adam", "metrics": ["mse"]})

        time = pd.date_range('2000-01-01', freq='24H', periods=5)
        x = xr.DataArray([1, 2, 3, 4, 5], dims=["time"], coords={'time': time})
        y = xr.DataArray([1, 2, 3, 4, 5], dims=["time"], coords={'time': time})

        wrapper.fit(x=x, target=y)

    def test_fit_transform_PyTorchModel(self):
        model = self.get_sequential_model(d_in=1, d_out=1, h=5)
        wrapper = PyTorchWrapper(model=model,
                                 fit_kwargs={"batch_size": 8, "epochs": 1},
                                 compile_kwargs={"loss": "mse", "optimizer": "Adam", "metrics": ["mse"]})

        time = pd.date_range('2000-01-01', freq='24H', periods=5)
        x = xr.DataArray([1, 2, 3, 4, 5], dims=["time"], coords={'time': time})
        y = xr.DataArray([1, 2, 3, 4, 5], dims=["time"], coords={'time': time})

        wrapper.fit(x=x, target=y)
        y_pred = wrapper.transform(x=x)

        self.assertIsNotNone(y_pred)
