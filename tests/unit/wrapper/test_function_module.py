import os
import unittest
from unittest.mock import MagicMock, patch

from pywatts.modules import FunctionModule


class TestFunctionModule(unittest.TestCase):

    def setUp(self) -> None:
        # define mock functions for fit and transform
        self.fit_function = MagicMock()
        self.transform_function = MagicMock()

        # define two function modules one using only a transform method
        # and another one using fit and transform methods
        self.transform_module = FunctionModule(self.transform_function)
        self.fit_transform_module = FunctionModule(self.transform_function, self.fit_function)

    def tearDown(self) -> None:
        self.transform_module = None

    def test_transform(self):
        data_mock = MagicMock()
        self.transform_module.fit(input_data=data_mock)
        self.transform_module.transform(input_data=data_mock)

        self.transform_function.assert_called_once_with(input_data=data_mock)

    def test_fit_transform(self):
        data_mock = MagicMock()
        self.fit_transform_module.fit(input_data=data_mock)
        self.fit_transform_module.transform(input_data=data_mock)

        self.fit_function.assert_called_once_with(input_data=data_mock)
        self.transform_function.assert_called_once_with(input_data=data_mock)

    @patch('pywatts.modules.wrappers.function_module.cloudpickle')
    @patch("builtins.open")
    def test_save(self, open_mock, cloudpickle_mock):
        fm_mock = MagicMock()
        fm_mock.get_path.return_value = os.path.join("testpath", "FunctionModule.pickle")
        json = self.transform_module.save(fm_mock)
        open_mock.assert_called_once_with(os.path.join("testpath", "FunctionModule.pickle"), 'wb')
        cloudpickle_mock.dump.assert_called_once()
        self.assertEqual(json, {
            "params": {},
            "is_fitted": True,
            "name": "FunctionModule",
            "class": "FunctionModule",
            "module": "pywatts.modules.wrappers.function_module",
            "pickled_module": os.path.join("testpath", "FunctionModule.pickle")
        })

    @patch('pywatts.modules.wrappers.function_module.cloudpickle')
    @patch("builtins.open")
    def test_load(self, open_mock, cloudpickle_mock):
        self.transform_module.load({
            "pickled_module": "pickle_path.pickle"
        })
        open_mock.assert_called_once_with("pickle_path.pickle", "rb")
        cloudpickle_mock.load.assert_called_once()
