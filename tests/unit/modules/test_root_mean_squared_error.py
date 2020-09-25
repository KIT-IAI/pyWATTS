import unittest
import xarray as xr
import pandas as pd

from pywatts.modules.root_mean_squared_error import RmseCalculator


class TestRMSECalculator(unittest.TestCase):

    def setUp(self) -> None:
        self.rmse_calculator = RmseCalculator()

    def tearDown(self) -> None:
        self.rmse_calculator = None

    def test_get_params(self):
        self.assertEqual(self.rmse_calculator.get_params(),
                         {
                             "target": "target",
                             "predictions": ["predictions"]
                         })

    def test_set_params(self):
        self.assertEqual(self.rmse_calculator.get_params(),
                         {
                             "target": "target",
                             "predictions": ["predictions"]
                         })
        self.rmse_calculator.set_params(target="testCol", prediction=["predictCol1", "predictCol2"])
        self.assertEqual(self.rmse_calculator.get_params(),
                         {
                             "target": "testCol",
                             "predictions": ["predictCol1", "predictCol2"]
                         })

    def test_transform(self):
        self.rmse_calculator.set_params(target="testCol", prediction=["predictCol1", "predictCol2"])

        time = pd.to_datetime(['2015-06-03 00:00:00', '2015-06-03 01:00:00',
                               '2015-06-03 02:00:00', '2015-06-03 03:00:00',
                               '2015-06-03 04:00:00'])

        result_time = pd.to_datetime(['2015-06-03 04:00:00'])

        test_data = xr.Dataset({"testCol": ("time", xr.DataArray([-2, -1, 0, 1, 2])),
                                "predictCol1": ("time", xr.DataArray([2, -3, 3, 1, -2])),
                                "predictCol2": ("time", xr.DataArray([4, 4, 3, -2, 1])), "time": time})

        test_result = self.rmse_calculator.transform(test_data)

        dimension = ["predictCol1", "predictCol2"]
        expected_result = xr.Dataset({"RMSE": (["time", "Result"], [[3., 4.]])}, coords={"Result": dimension, "time": result_time})

        xr.testing.assert_equal(test_result, expected_result)
