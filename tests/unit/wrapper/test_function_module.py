import os
import unittest
from unittest.mock import MagicMock, patch

from pywatts.wrapper.function_module import FunctionModule


class TestFunctionModule(unittest.TestCase):

    def setUp(self) -> None:
        self.mocked_function = MagicMock()
        self.function_module = FunctionModule(self.mocked_function)

    def tearDown(self) -> None:
        self.function_module = None

    def test_transform(self):
        data_mock = MagicMock()
        self.function_module.transform(data_mock)

        self.mocked_function.assert_called_once_with(data_mock)

    @patch('pywatts.wrapper.function_module.cloudpickle')
    @patch("builtins.open")
    def test_save(self, open_mock, cloudpickle_mock):
        fm_mock = MagicMock()
        fm_mock.get_path.return_value = os.path.join("testpath", "FunctionModule.pickle")
        json = self.function_module.save(fm_mock)
        open_mock.assert_called_once_with(os.path.join("testpath", "FunctionModule.pickle"), 'wb')
        cloudpickle_mock.dump.assert_called_once()
        self.assertEqual(json, {
            "params": {},
            "name": "FunctionModule",
            "class": "FunctionModule",
            "module": "pywatts.wrapper.function_module",
            "pickled_module": os.path.join("testpath", "FunctionModule.pickle")
        })

    @patch('pywatts.wrapper.function_module.cloudpickle')
    @patch("builtins.open")
    def test_load(self, open_mock, cloudpickle_mock):
        self.function_module.load({
            "pickled_module": "pickle_path.pickle"
        })
        open_mock.assert_called_once_with("pickle_path.pickle", "rb")
        cloudpickle_mock.load.assert_called_once()
