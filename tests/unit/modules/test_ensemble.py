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
                             "loss": None
                         })

    def test_set_params(self):
        self.assertEqual(self.ensemble.get_params(),
                         {
                             "weights": None,
                             "k_best": None,
                             "loss": None
                         })
        self.ensemble.set_params(weights=[0, 1, 2], k_best=5, loss=[3, 2, 1])
        self.assertEqual(self.ensemble.get_params(),
                         {
                             "weights": [0, 1, 2],
                             "k_best": 5,
                             "loss": [3, 2, 1]
                         })

    def test_transform_averaging(self):
        # averaging
        time = pd.date_range('2002-01-01', freq='24H', periods=7)

        da1 = xr.DataArray([2, 3, 4, 5, 6, 7, 8], dims=["time"], coords={'time': time})
        da2 = xr.DataArray([3, 4, 5, 6, 7, 8, 9], dims=["time"], coords={'time': time})

        result = self.ensemble.transform(y1=da1, y2=da2)

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

        self.ensemble.set_params(k_best="auto", loss=[1, 2, 1, 4, 10])

        self.ensemble.fit(y1=da1, y2=da2, y3=da3, y4=da4, y5=da5)
        result = self.ensemble.transform(y1=da1, y2=da2, y3=da3, y4=da4, y5=da5)

        expected_result = xr.DataArray([3, 4, 5, 6, 7, 8, 9],
                                       dims=["time"], coords={'time': time})

        xr.testing.assert_equal(result, expected_result)

    def test_transform_averaging_kbest(self):
        # waveraging k-best with given k
        time = pd.date_range('2002-01-01', freq='24H', periods=7)

        da1 = xr.DataArray([2, 3, 4, 5, 6, 7, 8], dims=["time"], coords={'time': time})
        da2 = xr.DataArray([3, 4, 5, 6, 7, 8, 9], dims=["time"], coords={'time': time})
        da3 = xr.DataArray([4, 5, 6, 7, 8, 9, 10], dims=["time"], coords={'time': time})

        self.ensemble.set_params(k_best=2, loss=[1, 2, 3])

        self.ensemble.fit(y1=da1, y2=da2, y3=da3)
        result = self.ensemble.transform(y1=da1, y2=da2, y3=da3)

        expected_result = xr.DataArray([2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5],
                                       dims=["time"], coords={'time': time})

        xr.testing.assert_equal(result, expected_result)

    def test_transform_weighting(self):
        # weighting based on given weights
        time = pd.date_range('2002-01-01', freq='24H', periods=7)

        da1 = xr.DataArray([2, 3, 4, 5, 6, 7, 8], dims=["time"], coords={'time': time})
        da2 = xr.DataArray([3, 4, 5, 6, 7, 8, 9], dims=["time"], coords={'time': time})

        self.ensemble.set_params(weights=[1, 0])
        result = self.ensemble.transform(y1=da1, y2=da2)
        expected_result = xr.DataArray([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dims=["time"], coords={'time': time})
        xr.testing.assert_equal(result, expected_result)

        self.ensemble.set_params(weights=[0, 1])
        result = self.ensemble.transform(y1=da1, y2=da2)
        expected_result = xr.DataArray([3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dims=["time"], coords={'time': time})
        xr.testing.assert_equal(result, expected_result)

        self.ensemble.set_params(weights=[1, 1])
        result = self.ensemble.transform(y1=da1, y2=da2)
        expected_result = xr.DataArray([2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5], dims=["time"], coords={'time': time})
        xr.testing.assert_equal(result, expected_result)

        self.ensemble.set_params(weights=[0.75, 0.25])
        result = self.ensemble.transform(y1=da1, y2=da2)
        expected_result = xr.DataArray([2.25, 3.25, 4.25, 5.25, 6.25, 7.25, 8.25], dims=["time"], coords={'time': time})
        xr.testing.assert_equal(result, expected_result)

        self.ensemble.set_params(weights=[0.25, 0.75])
        result = self.ensemble.transform(y1=da1, y2=da2)
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

        self.ensemble.set_params(weights=[3, 1, 3, 1, 100], k_best="auto", loss=[1, 2, 1, 4, 10])

        self.ensemble.fit(y1=da1, y2=da2, y3=da3, y4=da4, y5=da5)
        result = self.ensemble.transform(y1=da1, y2=da2, y3=da3, y4=da4, y5=da5)

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

        self.ensemble.set_params(weights=[1, 3, 3, 1, 100], k_best=2, loss=[1, 2, 3, 4, 10])

        self.ensemble.fit(y1=da1, y2=da2, y3=da3, y4=da4, y5=da5)
        result = self.ensemble.transform(y1=da1, y2=da2, y3=da3, y4=da4, y5=da5)

        expected_result = xr.DataArray([2.75, 3.75, 4.75, 5.75, 6.75, 7.75, 8.75],
                                       dims=["time"], coords={'time': time})

        xr.testing.assert_equal(result, expected_result)

    def test_transform_weighting_auto(self):
        # weighting with weights based on loss values
        time = pd.date_range('2002-01-01', freq='24H', periods=7)

        da1 = xr.DataArray([2, 3, 4, 5, 6, 7, 8], dims=["time"], coords={'time': time})
        da2 = xr.DataArray([3, 4, 5, 6, 7, 8, 9], dims=["time"], coords={'time': time})

        self.ensemble.set_params(weights="auto", loss=[2, 6])
        self.ensemble.fit(y1=da1, y2=da2)
        result = self.ensemble.transform(y1=da1, y2=da2)
        expected_result = xr.DataArray([2.25, 3.25, 4.25, 5.25, 6.25, 7.25, 8.25], dims=["time"], coords={'time': time})
        xr.testing.assert_equal(result, expected_result)

        self.ensemble.set_params(weights="auto", loss=[6, 2])
        self.ensemble.fit(y1=da1, y2=da2)
        result = self.ensemble.transform(y1=da1, y2=da2)
        expected_result = xr.DataArray([2.75, 3.75, 4.75, 5.75, 6.75, 7.75, 8.75], dims=["time"], coords={'time': time})
        xr.testing.assert_equal(result, expected_result)

        self.ensemble.set_params(weights="auto", loss=[2, 2])
        self.ensemble.fit(y1=da1, y2=da2)
        result = self.ensemble.transform(y1=da1, y2=da2)
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

        self.ensemble.set_params(weights="auto", k_best="auto", loss=[1, 1, 3, 3, 10])

        self.ensemble.fit(y1=da1, y2=da2, y3=da3, y4=da4, y5=da5)
        result = self.ensemble.transform(y1=da1, y2=da2, y3=da3, y4=da4, y5=da5)

        expected_result = xr.DataArray([2.25, 3.25, 4.25, 5.25, 6.25, 7.25, 8.25],
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

        self.ensemble.set_params(weights="auto", k_best=2, loss=[1, 3, 4, 5, 10])

        self.ensemble.fit(y1=da1, y2=da2, y3=da3, y4=da4, y5=da5)
        result = self.ensemble.transform(y1=da1, y2=da2, y3=da3, y4=da4, y5=da5)

        expected_result = xr.DataArray([2.25, 3.25, 4.25, 5.25, 6.25, 7.25, 8.25],
                                       dims=["time"], coords={'time': time})

        xr.testing.assert_equal(result, expected_result)

    def test_wrong_parameter(self):
        time = pd.date_range('2002-01-01', freq='24H', periods=7)

        da1 = xr.DataArray([2, 3, 4, 5, 6, 7, 8], dims=["time"], coords={'time': time})
        da2 = xr.DataArray([3, 4, 5, 6, 7, 8, 9], dims=["time"], coords={'time': time})
        da3 = xr.DataArray([4, 5, 5, 6, 7, 8, 9], dims=["time"], coords={'time': time})

        # test if exception is thrown if number of loss values != number of forecasts
        self.ensemble.set_params(weights="auto", k_best=2, loss=[1, 3, 4, 5, 10])
        self.assertRaises(WrongParameterException, lambda: self.ensemble.fit(y1=da1, y2=da2, y3=da3))
        self.ensemble = Ensemble()

        # test if exception is thrown if k > number of loss values
        self.ensemble.set_params(k_best=4, loss=[1, 3, 4])
        self.assertRaises(WrongParameterException, lambda: self.ensemble.fit(y1=da1, y2=da2, y3=da3))
        self.ensemble = Ensemble()

        # test if exception is thrown if weights="auto" but no loss is given
        self.ensemble.set_params(weights="auto", loss=[])
        self.assertRaises(WrongParameterException, lambda: self.ensemble.fit(y1=da1, y2=da2, y3=da3))
        self.ensemble = Ensemble()

        # test if exception is thrown if k_best is defined but no loss is given
        self.ensemble.set_params(k_best=4, loss=[])
        self.assertRaises(WrongParameterException, lambda: self.ensemble.fit(y1=da1, y2=da2, y3=da3))
        self.ensemble = Ensemble()
