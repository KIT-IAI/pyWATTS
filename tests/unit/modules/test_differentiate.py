import unittest

import numpy as np
import pandas as pd
import xarray as xr

from pywatts.modules import Differentiate
from pywatts.core.exceptions.wrong_parameter_exception import WrongParameterException


class TestResampler(unittest.TestCase):
    def setUp(self):
        self.start_time = "2000-01-01"
        self.number_of_households = 10
        self.times = 365
        self.linear_data = np.arange(0, self.times)
        self.load_data = np.random.randint(40, 80, (self.times))
        self.multiple_households = np.random.randint(40, 80, (self.number_of_households, self.times))
        self.dataset = xr.Dataset({
                "linear": ("time", self.linear_data),
                # example load data with two dimensions
                "load": (["time", "giga_watts"], self.load_data.reshape(-1, 1)),
                # example load as usual one dimensional vector
                "load[GW]": ("time", self.load_data),
                # example load data with three dimensions
                "load_households": (
                    ["household", "time", "giga_watts"],
                    self.multiple_households.transpose().reshape((self.number_of_households, self.times, 1))
                ),
                # another example load data with two dimensions
                "load_households[GW]": (["household", "time"], self.multiple_households)
            }, coords={
                "time": pd.date_range(self.start_time, freq="D", periods=self.times),
                "household": [f"house_{x}" for x in range(self.number_of_households)]
            }
        )

    def tearDown(self):
        self.linear_data = None
        self.load_data = None
        self.multiple_households = None
        self.dataset = None

    def test_set_get_params(self):
        # check parameters a set in __init__ and returned in get_params correctly
        # also set new parameters via set_params and check get_params
        params = {
            "target_index": "INDEX",
            "n": 1,
            "axis": 2,
            "pad": False,
            "pad_args": {}
        }
        obj = Differentiate(**params)
        self.assertEqual(obj.get_params(), params)
        new_params = {
            "target_index": "NEW_INDEX",
            "n": [1, 2, 3],
            "axis": -1,
            "pad": True,
            "pad_args": {"mode": "empty"}
        }
        obj.set_params(**new_params)
        self.assertEqual(obj.get_params(), new_params)

    def test_linear_diff(self, target_index="linear"):
        # test if the first and second derivative is correct
        obj = Differentiate(target_index=target_index, n=[1, 2])
        diff = obj.transform(self.dataset)
        self.assertTrue((diff[f"{target_index}_d1"] == np.ones(self.times - 1)).all())
        self.assertTrue((diff[f"{target_index}_d2"] == np.zeros(self.times - 2)).all())

    def test_one_dimensional(self, target_index="load[GW]"):
        # test differentiate method for one dimensional vector without and with padding
        # one time with default values and one time with custom values for padding
        for n in [1, 5, 100]:
            # default parameter
            obj = Differentiate(target_index=target_index, n=n)
            dataset = obj.transform(self.dataset.copy())
            self.assertEqual(len(dataset[f"{target_index}_d{n}"]), len(self.dataset[target_index]) - n)

            # try default padding
            obj = Differentiate(target_index=target_index, n=n, pad=True)
            dataset = obj.transform(self.dataset.copy())
            self.assertTrue((dataset[f"{target_index}_d{n}"][:n] == np.zeros(n, dtype=np.int)).all())

            # try specific padding
            obj = Differentiate(target_index=target_index, n=n, pad=True, pad_args={"constant_values": 1})
            dataset = obj.transform(self.dataset.copy())
            self.assertTrue((dataset[f"{target_index}_d{n}"][:n] == np.ones(n, dtype=np.int)).all())

    def test_two_dimensional(self, target_index=["load", "load_households[GW]"]):
        # test differentiate method for two dimensional time-series with and without padding
        # default parameter
        obj = Differentiate(target_index=target_index, n=1)
        dataset = obj.transform(self.dataset.copy())
        for idx in target_index:
            self.assertEqual(dataset[f"{idx}_d{1}"].shape[-1], self.dataset[idx].shape[-1] - 1)

        # different axis
        obj = Differentiate(target_index=target_index, n=1, axis=1)
        dataset = obj.transform(self.dataset.copy())
        for idx in target_index:
            self.assertEqual(dataset[f"{idx}_d{1}"].shape[1], self.dataset[idx].shape[1] - 1)

        # with padding
        obj = Differentiate(target_index=target_index, n=[1, 2], pad=True)
        dataset = obj.transform(self.dataset.copy())
        for idx in target_index:
            self.assertEqual(dataset[f"{idx}_d{1}"].shape[-1], self.dataset[idx].shape[-1])
            self.assertEqual(dataset[f"{idx}_d{2}"].shape[-1], self.dataset[idx].shape[-1])
            self.assertTrue((dataset[f"{idx}_d{1}"][:, :1] == np.zeros(1, dtype=np.int)).all())
            self.assertTrue((dataset[f"{idx}_d{2}"][:, :2] == np.zeros(2, dtype=np.int)).all())

    def test_three_dimensional(self, target_index="load_households"):
        # test differentiate method for two dimensional time-series with and without padding
        # default parameter
        obj = Differentiate(target_index=target_index, n=1)
        dataset = obj.transform(self.dataset.copy())
        self.assertEqual(dataset[f"{target_index}_d{1}"].shape[-1], self.dataset[target_index].shape[-1] - 1)

        # different axis
        obj = Differentiate(target_index=target_index, n=1, axis=1)
        dataset = obj.transform(self.dataset.copy())
        self.assertEqual(dataset[f"{target_index}_d{1}"].shape[1], self.dataset[target_index].shape[1] - 1)

        obj = Differentiate(target_index=target_index, n=1, axis=2)
        dataset = obj.transform(self.dataset.copy())
        self.assertEqual(dataset[f"{target_index}_d{1}"].shape[2], self.dataset[target_index].shape[2] - 1)

        # with padding and different axis
        obj = Differentiate(target_index=target_index, n=[1, 2], axis=0, pad=True)
        dataset = obj.transform(self.dataset.copy())
        self.assertEqual(dataset[f"{target_index}_d{1}"].shape[1], self.dataset[target_index].shape[1])
        self.assertEqual(dataset[f"{target_index}_d{2}"].shape[1], self.dataset[target_index].shape[1])
        self.assertTrue((dataset[f"{target_index}_d{1}"][:1, :, :] == np.zeros(1, dtype=np.int)).all())
        self.assertTrue((dataset[f"{target_index}_d{2}"][:2, :, :] == np.zeros(2, dtype=np.int)).all())

        obj = Differentiate(target_index=target_index, n=[1, 2], axis=1, pad=True)
        dataset = obj.transform(self.dataset.copy())
        self.assertEqual(dataset[f"{target_index}_d{1}"].shape[1], self.dataset[target_index].shape[1])
        self.assertEqual(dataset[f"{target_index}_d{2}"].shape[1], self.dataset[target_index].shape[1])
        self.assertTrue((dataset[f"{target_index}_d{1}"][:, :1, :] == np.zeros(1, dtype=np.int)).all())
        self.assertTrue((dataset[f"{target_index}_d{2}"][:, :2, :] == np.zeros(2, dtype=np.int)).all())
        obj = Differentiate(target_index=target_index, n=[1, 2], pad=True)

        dataset = obj.transform(self.dataset.copy())
        self.assertEqual(dataset[f"{target_index}_d{1}"].shape[-1], self.dataset[target_index].shape[-1])
        self.assertEqual(dataset[f"{target_index}_d{2}"].shape[-1], self.dataset[target_index].shape[-1])
        self.assertTrue((dataset[f"{target_index}_d{1}"][:, :, :1] == np.zeros(1, dtype=np.int)).all())
        self.assertTrue((dataset[f"{target_index}_d{2}"][:, :, :2] == np.zeros(2, dtype=np.int)).all())

    def test_target_index_none(self):
        # test if None as target_index uses time indexes to differentiate
        obj = Differentiate(target_index=None, n=1)
        diff = obj.transform(self.dataset)
        self.assertTrue((diff["time_d1"] == np.full(self.times - 1, np.timedelta64(1, "D"))).all())

    def test_wrong_parameter(self):
        # test if exception is thrown for not existing dataset target index
        obj = Differentiate(target_index="NotExisting")
        self.assertRaises(WrongParameterException, lambda: obj.transform(self.dataset))
