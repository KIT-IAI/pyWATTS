import unittest
import pandas as pd
import xarray as xr

from pywatts.core.exceptions.wrong_parameter_exception import WrongParameterException
from pywatts.modules import Ensemble


class TestEnsemble(unittest.TestCase):
    def setUp(self) -> None:
        self.ensemble = Ensemble()

    def tearDown(self) -> None:
        self.ensemble = None

    def test_get_params(self):
        params = self.ensemble.get_params()

        self.assertEqual(params,
                         {
                             "weights": None,
                             "k_best": None,
                             "loss_metric": Ensemble.LossMetric.RMSE
                         })

    def test_set_params(self):
        self.assertEqual(self.ensemble.get_params(),
                         {
                             "weights": None,
                             "k_best": None,
                             "loss_metric": Ensemble.LossMetric.RMSE
                         })
        self.ensemble.set_params(weights=[0, 1, 2], k_best=5, loss_metric=Ensemble.LossMetric.MAE)
        self.assertEqual(self.ensemble.get_params(),
                         {
                             "weights": [0, 1, 2],
                             "k_best": 5,
                             "loss_metric": Ensemble.LossMetric.MAE
                         })

    def test_transform_averaging(self):
        # averaging
        time = pd.date_range('2002-01-01', freq='24H', periods=7)

        da1 = xr.DataArray([2, 3, 4, 5, 6, 7, 8], dims=["time"], coords={'time': time})
        da2 = xr.DataArray([3, 4, 5, 6, 7, 8, 9], dims=["time"], coords={'time': time})

        # ensemble does not depend on the given target
        da_target = xr.DataArray([100, 200, 300, 400, 500, 600, 700], dims=["time"], coords={'time': time})

        self.ensemble.fit(y1=da1, y2=da2, target=da_target)
        result = self.ensemble.transform(y1=da1, y2=da2, target=da_target)

        # weights must be None
        expected_weights = None
        weights = self.ensemble.weights_
        self.assertEqual(weights, expected_weights)

        expected_result = xr.DataArray([2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5], dims=["time"], coords={'time': time})
        xr.testing.assert_equal(result, expected_result)

    def test_transform_averaging_kbest_auto(self):
        # averaging k-best with k based on loss values
        time = pd.date_range('2002-01-01', freq='24H', periods=7)

        da1 = xr.DataArray([2, 3, 4, 5, 6, 7, 8], dims=["time"], coords={'time': time})
        da2 = xr.DataArray([3, 4, 5, 6, 7, 8, 9], dims=["time"], coords={'time': time})
        da3 = xr.DataArray([2, 3, 4, 5, 6, 7, 8], dims=["time"], coords={'time': time})
        da4 = xr.DataArray([5, 6, 7, 8, 9, 10, 11], dims=["time"], coords={'time': time})
        da5 = xr.DataArray([12, 13, 14, 15, 16, 17, 18], dims=["time"], coords={'time': time})

        # ensemble depends on the given target
        da_target = xr.DataArray([3, 4, 5, 6, 7, 8, 9], dims=["time"], coords={'time': time})

        self.ensemble.set_params(k_best="auto", loss_metric=Ensemble.LossMetric.RMSE)

        self.ensemble.fit(y1=da1, y2=da2, y3=da3, y4=da4, y5=da5, target=da_target)
        result = self.ensemble.transform(y1=da1, y2=da2, y3=da3, y4=da4, y5=da5, target=da_target)

        # automated k-estimation based on loss  must set the weight of da5 to 0
        expected_weights = [0.25, 0.25, 0.25, 0.25, 0.0]
        weights = self.ensemble.weights_
        self.assertListEqual(weights, expected_weights)

        expected_result = xr.DataArray([3, 4, 5, 6, 7, 8, 9],
                                       dims=["time"], coords={'time': time})
        xr.testing.assert_equal(result, expected_result)

    def test_transform_averaging_kbest(self):
        # averaging k-best with given k
        time = pd.date_range('2002-01-01', freq='24H', periods=7)

        da1 = xr.DataArray([2, 3, 4, 5, 6, 7, 8], dims=["time"], coords={'time': time})
        da2 = xr.DataArray([3, 4, 5, 6, 7, 8, 9], dims=["time"], coords={'time': time})
        da3 = xr.DataArray([4, 5, 6, 7, 8, 9, 10], dims=["time"], coords={'time': time})

        # ensemble depends on the given target
        da_target = xr.DataArray([2, 3, 4, 5, 6, 7, 8], dims=["time"], coords={'time': time})

        self.ensemble.set_params(k_best=2, loss_metric=Ensemble.LossMetric.RMSE)

        self.ensemble.fit(y1=da1, y2=da2, y3=da3, target=da_target)
        result = self.ensemble.transform(y1=da1, y2=da2, y3=da3, target=da_target)

        # automated k-estimation based on loss must set the weight of da3 to 0
        expected_weights = [0.5, 0.5, 0.0]
        weights = self.ensemble.weights_
        self.assertListEqual(weights, expected_weights)

        expected_result = xr.DataArray([2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5],
                                       dims=["time"], coords={'time': time})
        xr.testing.assert_equal(result, expected_result)

    def test_transform_weighting(self):
        # weighting based on given weights
        time = pd.date_range('2002-01-01', freq='24H', periods=7)

        da1 = xr.DataArray([2, 3, 4, 5, 6, 7, 8], dims=["time"], coords={'time': time})
        da2 = xr.DataArray([3, 4, 5, 6, 7, 8, 9], dims=["time"], coords={'time': time})

        # ensemble does not depend on the given target
        da_target = xr.DataArray([100, 200, 300, 400, 500, 600, 700], dims=["time"], coords={'time': time})

        def _fit_transform():
            self.ensemble.fit(y1=da1, y2=da2, target=da_target)
            return self.ensemble.transform(y1=da1, y2=da2, target=da_target)

        # drop da2 via weight
        self.ensemble.set_params(weights=[1, 0])
        result = _fit_transform()

        expected_weights = [1, 0]
        weights = self.ensemble.weights_
        self.assertListEqual(weights, expected_weights)

        expected_result = xr.DataArray([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dims=["time"], coords={'time': time})
        xr.testing.assert_equal(result, expected_result)

        # drop da1 via weight
        self.ensemble.set_params(weights=[0, 1])
        result = _fit_transform()

        expected_weights = [0, 1]
        weights = self.ensemble.weights_
        self.assertListEqual(weights, expected_weights)

        expected_result = xr.DataArray([3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dims=["time"], coords={'time': time})
        xr.testing.assert_equal(result, expected_result)

        # equal weights => averaging
        self.ensemble.set_params(weights=[1, 1])
        result = _fit_transform()

        expected_weights = [0.5, 0.5]
        weights = self.ensemble.weights_
        self.assertListEqual(weights, expected_weights)

        expected_result = xr.DataArray([2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5], dims=["time"], coords={'time': time})
        xr.testing.assert_equal(result, expected_result)

        # overweight da1
        self.ensemble.set_params(weights=[3, 1])
        result = _fit_transform()

        expected_weights = [0.75, 0.25]
        weights = self.ensemble.weights_
        self.assertListEqual(weights, expected_weights)

        expected_result = xr.DataArray([2.25, 3.25, 4.25, 5.25, 6.25, 7.25, 8.25], dims=["time"], coords={'time': time})
        xr.testing.assert_equal(result, expected_result)

        # overweight da2
        self.ensemble.set_params(weights=[1, 3])
        result = _fit_transform()

        expected_weights = [0.25, 0.75]
        weights = self.ensemble.weights_
        self.assertListEqual(weights, expected_weights)

        expected_result = xr.DataArray([2.75, 3.75, 4.75, 5.75, 6.75, 7.75, 8.75], dims=["time"], coords={'time': time})
        xr.testing.assert_equal(result, expected_result)

    def test_transform_weighting_kbest_auto(self):
        # weighting based on given weights and k based on loss values
        time = pd.date_range('2002-01-01', freq='24H', periods=7)

        da1 = xr.DataArray([2, 3, 4, 5, 6, 7, 8], dims=["time"], coords={'time': time})
        da2 = xr.DataArray([3, 4, 5, 6, 7, 8, 9], dims=["time"], coords={'time': time})
        da3 = xr.DataArray([2, 3, 4, 5, 6, 7, 8], dims=["time"], coords={'time': time})
        da4 = xr.DataArray([5, 6, 7, 8, 9, 10, 11], dims=["time"], coords={'time': time})
        da5 = xr.DataArray([12, 13, 14, 15, 16, 17, 18], dims=["time"], coords={'time': time})

        # ensemble depends on the given target
        da_target = xr.DataArray([3, 4, 5, 6, 7, 8, 9], dims=["time"], coords={'time': time})

        self.ensemble.set_params(weights=[3, 1, 3, 1, 100], k_best="auto", loss_metric=Ensemble.LossMetric.RMSE)

        self.ensemble.fit(y1=da1, y2=da2, y3=da3, y4=da4, y5=da5, target=da_target)
        result = self.ensemble.transform(y1=da1, y2=da2, y3=da3, y4=da4, y5=da5, target=da_target)

        # although da5 was given a weight of 100, automated k-estimation on loss must set the weight to 0
        # and weights must be normalized
        expected_weights = [0.375, 0.125, 0.375, 0.125, 0.0]
        weights = self.ensemble.weights_
        self.assertListEqual(weights, expected_weights)

        expected_result = xr.DataArray([2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5],
                                       dims=["time"], coords={'time': time})
        xr.testing.assert_equal(result, expected_result)

    def test_transform_weighting_kbest(self):
        # weighting based on given weights and k
        time = pd.date_range('2002-01-01', freq='24H', periods=7)

        da1 = xr.DataArray([2, 3, 4, 5, 6, 7, 8], dims=["time"], coords={'time': time})
        da2 = xr.DataArray([3, 4, 5, 6, 7, 8, 9], dims=["time"], coords={'time': time})
        da3 = xr.DataArray([4, 5, 6, 7, 8, 9, 10], dims=["time"], coords={'time': time})
        da4 = xr.DataArray([5, 6, 7, 8, 9, 10, 11], dims=["time"], coords={'time': time})
        da5 = xr.DataArray([12, 13, 14, 15, 16, 17, 18], dims=["time"], coords={'time': time})

        # ensemble depends on the given target
        da_target = xr.DataArray([2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5], dims=["time"], coords={'time': time})

        self.ensemble.set_params(weights=[1, 3, 3, 1, 100], k_best=2, loss_metric=Ensemble.LossMetric.RMSE)

        self.ensemble.fit(y1=da1, y2=da2, y3=da3, y4=da4, y5=da5, target=da_target)
        result = self.ensemble.transform(y1=da1, y2=da2, y3=da3, y4=da4, y5=da5, target=da_target)

        # weights of da1 and da2 must be normalized and weights da3, da4 and da5 must be set to 0
        expected_weights = [0.25, 0.75, 0.0, 0.0, 0.0]
        weights = self.ensemble.weights_
        self.assertListEqual(weights, expected_weights)

        expected_result = xr.DataArray([2.75, 3.75, 4.75, 5.75, 6.75, 7.75, 8.75],
                                       dims=["time"], coords={'time': time})
        xr.testing.assert_equal(result, expected_result)

    def test_transform_weighting_auto(self):
        # weighting with weights based on loss values
        time = pd.date_range('2002-01-01', freq='24H', periods=7)

        da1 = xr.DataArray([2, 3, 4, 5, 6, 7, 8], dims=["time"], coords={'time': time})
        da2 = xr.DataArray([3, 4, 5, 6, 7, 8, 9], dims=["time"], coords={'time': time})

        # ensemble depends on the given target
        self.ensemble.set_params(weights="auto", loss_metric=Ensemble.LossMetric.RMSE)

        # overweight da1 since it closer to the target
        da_target = xr.DataArray([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], dims=["time"], coords={'time': time})

        self.ensemble.fit(y1=da1, y2=da2, target=da_target)
        result = self.ensemble.transform(y1=da1, y2=da2, target=da_target)

        expected_weights = [0.75, 0.25]
        weights = self.ensemble.weights_
        self.assertListEqual(weights, expected_weights)

        expected_result = xr.DataArray([2.25, 3.25, 4.25, 5.25, 6.25, 7.25, 8.25], dims=["time"], coords={'time': time})
        xr.testing.assert_equal(result, expected_result)

        # overweight da2 since it closer to the target
        da_target = xr.DataArray([3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5], dims=["time"], coords={'time': time})

        self.ensemble.fit(y1=da1, y2=da2, target=da_target)
        result = self.ensemble.transform(y1=da1, y2=da2, target=da_target)

        expected_weights = [0.25, 0.75]
        weights = self.ensemble.weights_
        self.assertListEqual(weights, expected_weights)

        expected_result = xr.DataArray([2.75, 3.75, 4.75, 5.75, 6.75, 7.75, 8.75], dims=["time"], coords={'time': time})
        xr.testing.assert_equal(result, expected_result)

        # equal weights since the target is the average of da1 and da2
        da_target = xr.DataArray([2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5], dims=["time"], coords={'time': time})

        self.ensemble.fit(y1=da1, y2=da2, target=da_target)
        result = self.ensemble.transform(y1=da1, y2=da2, target=da_target)

        expected_weights = [0.5, 0.5]
        weights = self.ensemble.weights_
        self.assertListEqual(weights, expected_weights)

        expected_result = xr.DataArray([2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5], dims=["time"], coords={'time': time})
        xr.testing.assert_equal(result, expected_result)

    def test_transform_weighting_auto_kbest_auto(self):
        # weighting k-best with weights and k based on loss values
        time = pd.date_range('2002-01-01', freq='24H', periods=7)

        da1 = xr.DataArray([2, 3, 4, 5, 6, 7, 8], dims=["time"], coords={'time': time})
        da2 = xr.DataArray([2, 3, 4, 5, 6, 7, 8], dims=["time"], coords={'time': time})
        da3 = xr.DataArray([3, 4, 5, 6, 7, 8, 9], dims=["time"], coords={'time': time})
        da4 = xr.DataArray([3, 4, 5, 6, 7, 8, 9], dims=["time"], coords={'time': time})
        da5 = xr.DataArray([12, 13, 14, 15, 16, 17, 18], dims=["time"], coords={'time': time})

        # ensemble depends on the given target
        da_target = xr.DataArray([2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5], dims=["time"], coords={'time': time})

        self.ensemble.set_params(weights="auto", k_best="auto", loss_metric=Ensemble.LossMetric.RMSE)

        self.ensemble.fit(y1=da1, y2=da2, y3=da3, y4=da4, y5=da5, target=da_target)
        result = self.ensemble.transform(y1=da1, y2=da2, y3=da3, y4=da4, y5=da5, target=da_target)

        # weights of da1, da2, da3, and da4 must be equal since the target is the average of da1, da2, da3, and da4
        # and weight of da5 must be set to 0
        expected_weights = [0.25, 0.25, 0.25, 0.25, 0.0]
        weights = self.ensemble.weights_
        self.assertListEqual(weights, expected_weights)

        expected_result = xr.DataArray([2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5],
                                       dims=["time"], coords={'time': time})
        xr.testing.assert_equal(result, expected_result)

    def test_transform_weighting_auto_kbest(self):
        # weighting k-best with weights based on loss values and given k
        time = pd.date_range('2002-01-01', freq='24H', periods=7)

        da1 = xr.DataArray([2, 3, 4, 5, 6, 7, 8], dims=["time"], coords={'time': time})
        da2 = xr.DataArray([3, 4, 5, 6, 7, 8, 9], dims=["time"], coords={'time': time})
        da3 = xr.DataArray([4, 5, 5, 6, 7, 8, 9], dims=["time"], coords={'time': time})
        da4 = xr.DataArray([5, 6, 5, 6, 7, 8, 9], dims=["time"], coords={'time': time})
        da5 = xr.DataArray([12, 13, 14, 15, 16, 17, 18], dims=["time"], coords={'time': time})

        # ensemble depends on the given target
        da_target = xr.DataArray([2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5], dims=["time"], coords={'time': time})

        self.ensemble.set_params(weights="auto", k_best=2, loss_metric=Ensemble.LossMetric.RMSE)

        self.ensemble.fit(y1=da1, y2=da2, y3=da3, y4=da4, y5=da5, target=da_target)
        result = self.ensemble.transform(y1=da1, y2=da2, y3=da3, y4=da4, y5=da5, target=da_target)

        # weights of da1, and da2 must be equal since the target is the average of da1 and da2
        # and weights of da3, da4 and da5 must be set to 0
        expected_weights = [0.5, 0.5, 0.0, 0.0, 0.0]
        weights = self.ensemble.weights_
        self.assertListEqual(weights, expected_weights)

        expected_result = xr.DataArray([2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5],
                                       dims=["time"], coords={'time': time})
        xr.testing.assert_equal(result, expected_result)

    def test_wrong_parameter(self):
        time = pd.date_range('2002-01-01', freq='24H', periods=7)

        da1 = xr.DataArray([2, 3, 4, 5, 6, 7, 8], dims=["time"], coords={'time': time})
        da2 = xr.DataArray([3, 4, 5, 6, 7, 8, 9], dims=["time"], coords={'time': time})
        da3 = xr.DataArray([4, 5, 5, 6, 7, 8, 9], dims=["time"], coords={'time': time})
        da_target = xr.DataArray([2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5], dims=["time"], coords={'time': time})

        # test if exception is thrown if k > number of loss values
        self.ensemble.set_params(k_best=4, loss_metric=Ensemble.LossMetric.RMSE)
        self.assertRaises(WrongParameterException, lambda: self.ensemble.fit(y1=da1, y2=da2, y3=da3, target=da_target))

        # test if exception is thrown if len(weights) != given forecasts
        self.ensemble.set_params(weights=[1, 2], loss_metric=Ensemble.LossMetric.RMSE)
        self.assertRaises(WrongParameterException, lambda: self.ensemble.fit(y1=da1, y2=da2, y3=da3, target=da_target))
