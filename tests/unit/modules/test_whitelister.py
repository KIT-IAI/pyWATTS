import unittest
import xarray as xr

from pywatts.modules.whitelister import WhiteLister


class TestWhitelister(unittest.TestCase):
    def setUp(self) -> None:
        self.whitelister = WhiteLister(target="target")

    def test_get_params(self):
        self.assertEqual(
            self.whitelister.get_params(),
            {
                "target": "target"
            }
        )

    def test_set_params(self):
        self.assertEqual(
            self.whitelister.get_params(),
            {
                "target": "target"
            }
        )
        self.whitelister.set_params(target="better_target")
        self.assertEqual(
            self.whitelister.get_params(),
            {
                "target": "better_target"
            }
        )

    def test_transform(self):
        x = xr.Dataset({
            "target" : xr.DataArray([1,2,3,4,5]),
            "42":xr.DataArray([42,42,42,42,42])
        })
        self.assertEqual(list(x.data_vars.keys()), ["target", "42"])
        result = self.whitelister.transform(x)
        self.assertEqual(list(result.data_vars.keys()), ["target"])
        self.assertEqual(list(x.data_vars.keys()), ["target", "42"])


    def test_transform_target_not_part(self):
        x = xr.Dataset({
            "target_12" : xr.DataArray([1,2,3,4,5]),
            "42":xr.DataArray([42,42,42,42,42])
        })
        self.assertRaises(Exception, self.whitelister.transform, x)
