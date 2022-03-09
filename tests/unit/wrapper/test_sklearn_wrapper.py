import unittest

import pytest
import xarray as xr
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
from sklearn.multioutput import MultiOutputRegressor
from sklearn.feature_selection import SelectKBest, f_regression

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from pywatts.modules import SKLearnWrapper


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

        wrapper.fit(test=xr.DataArray([1, 2, 3, 4, 5]))

        self.assertTrue("mean_" in scaler.__dir__())
        self.assertIsNotNone(scaler.mean_)

    def test_transform_TransformerMixin(self):
        scaler = StandardScaler()
        wrapper = SKLearnWrapper(module=scaler)
        self.assertFalse("mean_" in scaler.__dir__())
        time = pd.date_range('2000-01-08', freq='24H', periods=5)
        test = xr.DataArray([2, 2, 2, 2, 2], dims=["time"], coords={'time': time})

        wrapper.fit(test=test)
        result = wrapper.transform(test=test)

        self.assertListEqual(list(result), [0, 0, 0, 0, 0])
        self.assertEqual(result.shape, (5, 1))

    def test_fit_RegressorMixin(self):
        lin_reg = LinearRegression()
        wrapper = SKLearnWrapper(module=lin_reg)
        self.assertFalse("coef_" in lin_reg.__dir__())

        wrapper.fit(test=xr.DataArray([1, 2, 3, 4, 5]),
                    target=xr.DataArray([2, 2, 2, 2, 2]))

        self.assertTrue("coef_" in lin_reg.__dir__())
        self.assertIsNotNone(lin_reg.coef_)

    def test_transform_RegressorMixin(self):
        svr = SVR()
        wrapper = SKLearnWrapper(module=svr)
        time = pd.date_range('2000-01-08', freq='24H', periods=1)
        bar = xr.DataArray([1], dims=["time"], coords={'time': time})

        wrapper.fit(test=xr.DataArray([1, 2, 3, 4, 5]),
                    target=xr.DataArray([2, 2, 2, 2, 2]))

        result = wrapper.transform(bar=bar)
        assert result["target"].values[0] == 2.0
        self.assertEqual(result["target"].shape, (1, 1))

    def test_fit_SelectorMixin(self):
        kbest = SelectKBest(score_func=f_regression, k=1)
        wrapper = SKLearnWrapper(module=kbest)

        eps = 0.001
        wrapper.fit(feature1=xr.DataArray([x + eps for x in [2, 2, 3, 4, 4]]),
                    feature2=xr.DataArray([1, 2, 3, 4, 5]),
                    target=xr.DataArray([2, 2, 3, 4, 4]))

        self.assertTrue("scores_" in kbest.__dir__())
        self.assertIsNotNone(kbest.scores_)

    def test_transform_SelectorMixin(self):
        kbest = SelectKBest(score_func=f_regression, k=1)
        wrapper = SKLearnWrapper(module=kbest)
        eps = 0.001
        time = pd.date_range('2000-01-08', freq='24H', periods=5)
        target = xr.DataArray([2, 2, 3, 4, 4], dims=["time"], coords={'time': time})
        feature1 = target + eps
        feature2 = xr.DataArray([4, 4, 3, 2, 2], dims=["time"], coords={'time': time}) + eps

        wrapper.fit(feature1=feature1,
                    feature2=feature2,
                    target=target)

        result = wrapper.transform(feature1=feature1, feature2=feature2)

        self.assertListEqual(list(result), list(feature2))
        self.assertEqual(result.shape, (5, 1))

    def test_DensityMixin(self):
        gauss_density = GaussianMixture(n_components=2)
        wrapper = SKLearnWrapper(module=gauss_density)

        time = pd.date_range('2000-01-01', freq='24H', periods=10)
        time2 = pd.date_range('2000-01-08', freq='24H', periods=1)

        bar = xr.DataArray([2, 4, 5, 4, 3, 2, 1, 5, 5, 5], dims=["time"], coords={'time': time})
        wrapper.fit(bar=bar)

        bar1 = xr.DataArray([5], dims=["time"], coords={'time': time2})
        bar2 = xr.DataArray([2], dims=["time"], coords={'time': time2})

        result1 = wrapper.transform(bar=bar1)
        result0 = wrapper.transform(bar=bar2)

        assert result1.values[0] != result0.values[0]

        self.assertEqual(result1.shape, (1,))
        self.assertEqual(result0.shape, (1,))

    def test_fit_regression_multiple_datavariables(self):
        time = pd.date_range('2000-01-01', freq='24H', periods=7)
        time2 = pd.date_range('2000-01-08', freq='24H', periods=1)

        bar = xr.DataArray([2, 2, 2, 2, 3, 3, 3], dims=["time"], coords={'time': time})
        foo = xr.DataArray([4, 4, 4, 4, 6, 6, 6], dims=["time"], coords={'time': time})
        target = xr.DataArray([6, 6, 6, 6, 9, 9, 9], dims=["time"], coords={'time': time})

        lin_reg = LinearRegression()
        wrapper = SKLearnWrapper(module=lin_reg)
        self.assertFalse("coef_" in lin_reg.__dir__())

        wrapper.fit(bar=bar, foo=foo, target=target)
        result = wrapper.transform(bar=xr.DataArray([2], dims=["time"], coords={'time': time2}),
                                   foo=xr.DataArray([4], dims=["time"], coords={'time': time2}))
        self.assertAlmostEqual(result["target"].values[0, 0], 6.0)
        self.assertEqual(result["target"].shape, (1, 1))

    def test_fit_ClusterMixin(self):
        kmeans = KMeans(n_clusters=2)
        wrapper = SKLearnWrapper(module=kmeans)
        # self.assertFalse("coef_" in lin_reg.__dir__())

        time = pd.date_range('2000-01-01', freq='24H', periods=10)
        time2 = pd.date_range('2000-01-08', freq='24H', periods=1)

        bar = xr.DataArray([2, 4, 5, 4, 2, 2, 1, 5, 5, 5], dims=["time"], coords={'time': time})
        foo1 = xr.DataArray([5], dims=["time"], coords={'time': time2})
        foo2 = xr.DataArray([2], dims=["time"], coords={'time': time2})

        wrapper.fit(bar=bar)

        result1 = wrapper.transform(foo=foo1)
        result0 = wrapper.transform(foo=foo2)

        # Assert that both tested datapoints are in different clusters
        assert result1.values[0].argmax() != result0.values[0].argmax()

        self.assertEqual(result1.shape, (1, 2))
        self.assertEqual(result0.shape, (1, 2))

    def test_fit_ClassifierMixin(self):
        svc = SVC()
        wrapper = SKLearnWrapper(module=svc)
        time = pd.date_range('2000-01-01', freq='24H', periods=5)
        time2 = pd.date_range('2000-01-08', freq='24H', periods=1)
        bar = xr.DataArray([1, 2, 3, 4, 5], dims=["time"], coords={'time': time})
        foo = xr.DataArray([1], dims=["time"], coords={'time': time2})
        target = xr.DataArray([0, 0, 1, 1, 1], dims=["time"], coords={'time': time})

        wrapper.fit(bar=bar, target=target)

        result = wrapper.transform(bar=foo)
        assert result["target"].values[0] == 0
        self.assertEqual(result["target"].shape, (1, 1))

    @pytest.mark.xfail
    def test_fit_BiClusterMixin(self):
        # Currently biclustering is not supported
        assert False

    def test_transform_multiple_output(self):
        lin_reg = LinearRegression()
        multi_regressor = MultiOutputRegressor(lin_reg)
        wrapper = SKLearnWrapper(module=multi_regressor)
        time = pd.date_range('2000-01-01', freq='24H', periods=5)
        time2 = pd.date_range('2000-01-08', freq='24H', periods=1)

        bar = xr.DataArray([1, 2, 3, 4, 5], dims=["time"], coords={'time': time})
        foo = xr.DataArray([1], dims=["time"], coords={'time': time2})
        target = xr.DataArray([2, 2, 2, 2, 2], dims=["time"], coords={'time': time})
        target2 = xr.DataArray([3, 3, 3, 3, 3], dims=["time"], coords={'time': time})

        wrapper.fit(bar=bar, target1=target, target2=target2)

        result = wrapper.transform(bar=foo)
        self.assertAlmostEqual(result["target1"].values[0], 2.0)
        self.assertAlmostEqual(result["target2"].values[0], 3.0)
        self.assertEqual(result["target1"].shape, (1, 1))
        self.assertEqual(result["target2"].shape, (1, 1))
