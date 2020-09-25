import unittest

import pytest
import xarray as xr
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
from sklearn.multioutput import MultiOutputRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from pywatts.wrapper.sklearn_wrapper import SKLearnWrapper


class TestSklearnWrapper(unittest.TestCase):

    def test_get_params(self):
        scaler = StandardScaler()
        wrapper = SKLearnWrapper(module=scaler)
        self.assertEqual(wrapper.get_params(), scaler.get_params())

    def test_set_params(self):
        scaler = StandardScaler()
        wrapper = SKLearnWrapper(module=scaler)
        self.assertEqual(scaler.get_params()["with_mean"], True)
        wrapper.set_params(with_mean=False, )
        self.assertEqual(scaler.get_params()["with_mean"], False)

    def test_fit_TransformerMixin(self):
        scaler = StandardScaler()
        wrapper = SKLearnWrapper(module=scaler)
        self.assertFalse("mean_" in scaler.__dir__())

        wrapper.fit(xr.Dataset({"test": xr.DataArray([1, 2, 3, 4, 5])}))

        self.assertTrue("mean_" in scaler.__dir__())
        self.assertIsNotNone(scaler.mean_)

    def test_fit_RegressorMixin(self):
        lin_reg = LinearRegression()
        wrapper = SKLearnWrapper(module=lin_reg)
        self.assertFalse("coef_" in lin_reg.__dir__())

        wrapper.fit(xr.Dataset({"test": xr.DataArray([1, 2, 3, 4, 5])}),
                    xr.Dataset({"test": xr.DataArray([2, 2, 2, 2, 2])}))

        self.assertTrue("coef_" in lin_reg.__dir__())
        self.assertIsNotNone(lin_reg.coef_)

    def test_transform_RegressorMixin(self):
        svr = SVR()
        wrapper = SKLearnWrapper(module=svr)

        wrapper.fit(xr.Dataset({"test": xr.DataArray([1, 2, 3, 4, 5])}),
                    xr.Dataset({"test": xr.DataArray([2, 2, 2, 2, 2])}))

        time = pd.date_range('2000-01-08', freq='24H', periods=1)

        result = wrapper.transform(xr.Dataset({"test": ("time", xr.DataArray([1])), "time": time}))
        assert result.to_array().values[0, 0] == 2.0
        self.assertEqual(result.to_array().shape, (1, 1))

    def test_DensityMixin(self):
        gauss_density = GaussianMixture(n_components=2)
        wrapper = SKLearnWrapper(module=gauss_density)

        time = pd.date_range('2000-01-01', freq='24H', periods=10)
        time2 = pd.date_range('2000-01-08', freq='24H', periods=1)
        ds = xr.Dataset(
            {'BAR': (['time'], [2, 4, 5, 4, 3, 2, 1, 5, 5, 5]),
             'time': time})

        ds_predict_label1 = xr.Dataset(
            {'BAR': (['time'], [5]),
             'time': time2})
        wrapper.fit(ds)
        ds_predict_label0 = xr.Dataset(
            {'BAR': (['time'], [2]),
             'time': time2})

        result1 = wrapper.transform(ds_predict_label1)
        result0 = wrapper.transform(ds_predict_label0)

        assert result1.to_array().values[0, 0] != result0.to_array().values[0, 0]

        self.assertEqual(result1.to_array().shape, (1, 1))
        self.assertEqual(result0.to_array().shape, (1, 1))

    def test_fit_regression_multiple_datavariables(self):
        time = pd.date_range('2000-01-01', freq='24H', periods=7)
        time2 = pd.date_range('2000-01-08', freq='24H', periods=1)

        ds = xr.Dataset(
            {'BAR': (['time'], [2, 2, 2, 2, 3, 3, 3]),
             'BAR2': (['time'], [4, 4, 4, 4, 6, 6, 6]),
             'time': time})

        target = xr.Dataset(
            {'FOO': (['time'], [6, 6, 6, 6, 9, 9, 9]),
             'time': time})

        lin_reg = LinearRegression()
        wrapper = SKLearnWrapper(module=lin_reg)
        self.assertFalse("coef_" in lin_reg.__dir__())

        wrapper.fit(ds, target)

        result = wrapper.transform(xr.Dataset({"BAR": (["time"], xr.DataArray([2])),
                                               "BAR2": (["time"], xr.DataArray([4])),
                                               "time": time2}))
        self.assertAlmostEqual(result.to_array().values[0, 0, 0], 6.0)
        self.assertEqual(result.to_array().shape, (1, 1, 1))

    def test_fit_ClusterMixin(self):
        kmeans = KMeans(n_clusters=2)
        wrapper = SKLearnWrapper(module=kmeans)
        # self.assertFalse("coef_" in lin_reg.__dir__())

        time = pd.date_range('2000-01-01', freq='24H', periods=10)
        time2 = pd.date_range('2000-01-08', freq='24H', periods=1)
        ds = xr.Dataset(
            {'BAR': (['time'], [2, 4, 5, 4, 2, 2, 1, 5, 5, 5]),
             'time': time})

        ds_predict_label1 = xr.Dataset(
            {'BAR': (['time'], [5]),
             'time': time2})
        wrapper.fit(ds)
        ds_predict_label0 = xr.Dataset(
            {'BAR': (['time'], [2]),
             'time': time2})

        result1 = wrapper.transform(ds_predict_label1)
        result0 = wrapper.transform(ds_predict_label0)

        # Assert that both tested datapoints are in different clusters
        assert result1.to_array().values[0, 0].argmax() != result0.to_array().values[0, 0].argmax()

        self.assertEqual(result1.to_array().shape, (1, 1, 2))
        self.assertEqual(result0.to_array().shape, (1, 1, 2))

    def test_fit_ClassifierMixin(self):
        svc = SVC()
        wrapper = SKLearnWrapper(module=svc)
        time = pd.date_range('2000-01-01', freq='24H', periods=5)
        time2 = pd.date_range('2000-01-08', freq='24H', periods=1)

        wrapper.fit(xr.Dataset({"test": ("time", xr.DataArray([1, 2, 3, 4, 5])),
                                "time": time}),
                    xr.Dataset({"test": ("time", xr.DataArray([0, 0, 1, 1, 1])),
                                "time": time}))

        result = wrapper.transform(xr.Dataset({"test": ("time", xr.DataArray([1])),
                                               "time": time2}))
        assert result.to_array().values[0, 0] == 0
        self.assertEqual(result.to_array().shape, (1, 1))

    @pytest.mark.xfail
    def test_fit_BiClusterMixin(self):
        # Currently biclustering is not supported
        assert False

    def test_transform_multiple_output(self):
        lin_reg = LinearRegression()
        multi_regressor = MultiOutputRegressor(lin_reg)
        wrapper = SKLearnWrapper(module=multi_regressor)

        wrapper.fit(xr.Dataset({"test": xr.DataArray([1, 2, 3, 4, 5])}),
                    xr.Dataset({"target1": xr.DataArray([2, 2, 2, 2, 2]),
                                "target2": xr.DataArray([3, 3, 3, 3, 3])}))

        time = pd.date_range('2000-01-08', freq='24H', periods=1)

        result = wrapper.transform(xr.Dataset({"test": ("time", xr.DataArray([1])), "time": time}))
        self.assertAlmostEqual(result.to_array().values[0, 0, 0], 2.0)
        self.assertAlmostEqual(result.to_array().values[0, 0, 1], 3.0)
        self.assertEqual(result.to_array().shape, (1, 1, 2))

    def test_meta_estimator_mixin(self):
        svr = SVR()
        multi_regressor = MultiOutputRegressor(svr)
        wrapper = SKLearnWrapper(module=multi_regressor)

        wrapper.fit(xr.Dataset({"test": xr.DataArray([1, 2, 3, 4, 5])}),
                    xr.Dataset({"target1": xr.DataArray([2, 2, 2, 2, 2]),
                                "target2": xr.DataArray([3, 3, 3, 3, 3])}))

        time = pd.date_range('2000-01-08', freq='24H', periods=1)

        result = wrapper.transform(xr.Dataset({"test": ("time", xr.DataArray([1])), "time": time}))
        self.assertAlmostEqual(result.to_array().values[0, 0, 0], 2.0)
        self.assertAlmostEqual(result.to_array().values[0, 0, 1], 3.0)
        self.assertEqual(result.to_array().shape, (1, 1, 2))
