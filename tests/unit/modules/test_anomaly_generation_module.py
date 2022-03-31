import unittest

import numpy as np
import pandas as pd
import xarray as xr

from pywatts.modules.generation.anomaly_generation_module import AnomalyGeneration
from pywatts.core.exceptions.wrong_parameter_exception import WrongParameterException


class TestResampler(unittest.TestCase):

    def setUp(self):
        self.constant_load = np.ones(10)
        anomalies = np.full(10, True)
        anomalies[[0, -1]] = False
        self.anomalies = anomalies
        self.dataset = xr.Dataset({
                "load_1d": ("time", self.constant_load),
                # example load data with two dimensions
                "load_2d": (["time", "household"], self.constant_load.reshape(-1, 1)),
                # anomaly labels
                "anomalies": ("time", self.anomalies)
            }, coords={
                "time": pd.date_range("2000-01-01", freq="D", periods=len(self.constant_load)),
                "household": ["house"]
            }
        )

    def tearDown(self):
        self.constant_load = None
        self.anomalies = None
        self.dataset = None

    def test_set_get_params(self):
        params = {
            "count": 1,
            "anomaly": "gap",
            "anomaly_params": {},
            "length_params": {},
            "seed": 0,
        }
        module = AnomalyGeneration(**params)
        self.assertEqual(module.get_params(), params)

        new_params = {
            "count": 2,
            "anomaly": "constant",
            "anomaly_params": {"param": 1},
            "length_params": {"param": 1},
            "seed": 1,
        }
        old_params = params.copy()
        for key in new_params.keys():
            params[key] = new_params[key]
            module.set_params(**{key: new_params[key]})
            self.assertEqual(module.get_params(), params)

        module.set_params(**old_params)
        self.assertEqual(module.get_params(), old_params)

    def test_anomaly_type(self):
        anomaly_types = ["gap", "constant", "negate", "outlier"]
        for type in anomaly_types:
            # all should run without exception
            module = AnomalyGeneration(anomaly=type, count=1)
            module.transform(x=self.dataset["load_1d"].copy())
            module = AnomalyGeneration(anomaly=type, count=1)
            module.transform(x=self.dataset["load_1d"].copy(), labels=self.dataset["anomalies"].copy())

    def test_anomaly_count(self):
        # without previous labels
        for count in range(1, len(self.dataset.time) + 1):
            module = AnomalyGeneration(count=count)
            data = module.transform(x=self.dataset["load_1d"].copy())
            self.assertEqual(data["labels"].sum(), count)
            module = AnomalyGeneration(count=count)
            data = module.transform(x=self.dataset["load_2d"].copy())
            self.assertEqual(data["labels"].sum(), count)
        module = AnomalyGeneration(count=len(self.dataset.time) + 1)
        self.assertRaises(Exception, lambda: module.transform(x=self.dataset["load_1d"].copy()))
        module = AnomalyGeneration(count=len(self.dataset.time) + 1)
        self.assertRaises(Exception, lambda: module.transform(x=self.dataset["load_2d"].copy()))

        # with previous labels
        module1 = AnomalyGeneration(count=1)
        module2 = AnomalyGeneration(count=1)
        data1 = self.dataset["load_1d"].copy()
        data2 = self.dataset["load_2d"].copy()
        labels1 = None
        labels2 = None
        for i in range(1, len(self.dataset.time) + 1):
            data = module1.transform(x=data1, labels=labels1)
            data1 = data['AnomalyGeneration']
            labels1 = data['labels']
            print((labels1 != 0).sum())
            self.assertEqual((labels1 != 0).sum(), i)
            data = module2.transform(x=data2, labels=labels2)
            data2 = data['AnomalyGeneration']
            labels2 = data['labels']
            self.assertEqual((labels2.values.flatten() != 0).sum(), i)
        self.assertRaises(Exception, lambda: module1.transform(x=data1, labels=labels1))
        self.assertRaises(Exception, lambda: module2.transform(x=data2, labels=labels2))

    def test_anomaly_labels(self):
        # an one by another
        module = AnomalyGeneration(count=1)
        for _ in range((~self.anomalies).sum()):
            data = module.transform(x=self.dataset["load_1d"], labels=self.dataset["anomalies"])
        # this should fail
        self.assertRaises(Exception, lambda: module.transform(x=data["AnomalyGeneration"], labels=data["labels"]))

        # all at once
        module = AnomalyGeneration(count=(~self.anomalies).sum() + 1)
        self.assertRaises(Exception, lambda: module.transform(x=self.dataset["load_1d"], labels=self.dataset["anomalies"]))

    def test_anomaly_length(self):
        # normal distribution testing
        module = AnomalyGeneration(count=1, length_params={"distribution": "normal", "mean": 1, "std": 0})
        module.transform(x=self.dataset["load_1d"].copy(), labels=self.dataset["anomalies"].copy())
        module = AnomalyGeneration(count=1, length_params={"distribution": "normal", "mean": 2, "std": 0})
        self.assertRaises(WrongParameterException, lambda: module.transform(
            x=self.dataset["load_1d"].copy(), labels=self.dataset["anomalies"].copy()))

        # uniform distribution testing
        module = AnomalyGeneration(count=1, length_params={"distribution": "uniform", "min": 1, "max": 1})
        module.transform(x=self.dataset["load_1d"].copy(), labels=self.dataset["anomalies"].copy())
        module = AnomalyGeneration(count=1, length_params={"distribution": "uniform", "min": 2, "max": 2})
        self.assertRaises(WrongParameterException, lambda: module.transform(
            x=self.dataset["load_1d"].copy(), labels=self.dataset["anomalies"].copy()))

    def test_anomaly_gap(self):
        module = AnomalyGeneration(anomaly="gap", count=1)
        data = module.transform(x=self.dataset["load_1d"].copy(), labels=self.dataset["anomalies"].copy())
        self.assertEqual(np.isnan(data["AnomalyGeneration"]).sum(), 1)
        data = module.transform(x=data["AnomalyGeneration"], labels=data["labels"])
        self.assertEqual(np.isnan(data["AnomalyGeneration"]).sum(), 2)

        data = self.dataset.copy(deep=True)
        module = AnomalyGeneration(anomaly="gap", count=2)
        data = module.transform(x=self.dataset["load_1d"].copy(), labels=self.dataset["anomalies"].copy())
        self.assertEqual(np.isnan(data["AnomalyGeneration"]).sum(), 2)

    def test_anomaly_constant(self):
        module = AnomalyGeneration(anomaly="constant", count=1, anomaly_params={"constant": 0})
        data = module.transform(x=self.dataset["load_1d"].copy(), labels=self.dataset["anomalies"].copy())
        self.assertEqual((data["AnomalyGeneration"] == 0).sum(), 1)
        data = module.transform(x=data["AnomalyGeneration"], labels=data["labels"])
        self.assertEqual((data["AnomalyGeneration"] == 0).sum(), 2)

        module = AnomalyGeneration(anomaly="constant", count=2, anomaly_params={"constant": 0})
        data = module.transform(x=self.dataset["load_1d"].copy(), labels=self.dataset["anomalies"].copy())
        self.assertEqual((data["AnomalyGeneration"] == 0).sum(), 2)

    def test_anomaly_negate(self):
        module = AnomalyGeneration(anomaly="negate", count=1)
        data = module.transform(x=self.dataset["load_1d"].copy(), labels=self.dataset["anomalies"].copy())
        self.assertEqual((data["AnomalyGeneration"] == -1).sum(), 1)
        data = module.transform(x=data["AnomalyGeneration"], labels=data["labels"])
        self.assertEqual((data["AnomalyGeneration"] == -1).sum(), 2)

        module = AnomalyGeneration(anomaly="negate", count=2)
        data = module.transform(x=self.dataset["load_1d"].copy(), labels=self.dataset["anomalies"].copy())
        self.assertEqual((data["AnomalyGeneration"] == -1).sum(), 2)

    def test_anomaly_outlier(self):
        # mean
        module = AnomalyGeneration(anomaly="outlier", count=1,
                                  anomaly_params={"outlier_sign": "positive", "outlier_type": "mean"})
        data = module.transform(x=self.dataset["load_1d"].copy(), labels=self.dataset["anomalies"].copy())
        self.assertEqual((data["AnomalyGeneration"] > 1).sum(), 1)
        data = module.transform(x=data["AnomalyGeneration"], labels=data["labels"])
        self.assertEqual((data["AnomalyGeneration"] > 1).sum(), 2)

        module = AnomalyGeneration(anomaly="outlier", count=2,
                                  anomaly_params={"outlier_sign": "positive", "outlier_type": "mean"})
        data = module.transform(x=self.dataset["load_1d"].copy(), labels=self.dataset["anomalies"].copy())
        self.assertEqual((data["AnomalyGeneration"] > 1).sum(), 2)

        # multiple
        module = AnomalyGeneration(anomaly="outlier", count=1,
                                  anomaly_params={"outlier_sign": "positive", "outlier_type": "multiple"})
        data = module.transform(x=self.dataset["load_1d"].copy(), labels=self.dataset["anomalies"].copy())
        self.assertEqual((data["AnomalyGeneration"] > 1).sum(), 1)
        data = module.transform(x=data["AnomalyGeneration"], labels=data["labels"])
        self.assertEqual((data["AnomalyGeneration"] > 1).sum(), 2)

        module = AnomalyGeneration(anomaly="outlier", count=2,
                                  anomaly_params={"outlier_sign": "positive", "outlier_type": "multiple"})
        data = module.transform(x=self.dataset["load_1d"].copy(), labels=self.dataset["anomalies"].copy())
        self.assertEqual((data["AnomalyGeneration"] > 1).sum(), 2)
