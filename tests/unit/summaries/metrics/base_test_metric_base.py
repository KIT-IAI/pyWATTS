from abc import ABC, abstractmethod
from unittest.mock import MagicMock, patch


class BaseTestMetricBase(ABC):
    def setUp(self) -> None:
        self.metric = self.get_metric()(name="NAME")


    def tearDown(self) -> None:
        self.metric = None

    def test_get_params(self):
        self.assertEqual(self.metric.get_params(),
                         {'offset': 0, "cuts":[]})

    def test_set_params(self):
        self.metric.set_params(offset=24, cuts=[("Test", "test")])
        self.assertEqual(self.metric.get_params(),
                         {'offset': 24,
                          "cuts": [("Test", "test")]})


    @patch("builtins.open")
    @patch("pywatts.summaries.metric_base.cloudpickle")
    def test_save(self, cloudpickle_mock, open_mock):
        fm_mock = MagicMock()
        fm_mock.get_path.return_value = "filter_path"
        filter_mock = MagicMock()

        rmse = self.get_metric()(name="NAME", filter_method=filter_mock)

        json = rmse.save(fm_mock)

        fm_mock.get_path.assert_called_once_with("NAME_filter.pickle")
        open_mock.assert_called_once_with("filter_path", "wb")

        cloudpickle_mock.dump.assert_called_once_with(filter_mock, open_mock().__enter__.return_value)
        self.assertEqual(json["filter"], "filter_path")
        self.assertEqual(json["params"], {"offset": 0, "cuts" : []})


    @patch("builtins.open")
    @patch("pywatts.summaries.metric_base.cloudpickle")
    def test_load(self, cloudpickle_mock, open_mock):
        filter_mock = MagicMock()
        cloudpickle_mock.load.return_value = filter_mock

        metric = self.get_metric().load(self.load_information)

        open_mock.assert_called_once_with("filter_path", "rb")
        cloudpickle_mock.load.assert_called_once_with(open_mock().__enter__.return_value)

        self.assertEqual(metric.name, "NAME")
        self.assertEqual(metric.filter_method, filter_mock)
        self.assertEqual(metric.get_params(), {"offset": 24, "cuts": []})

    @abstractmethod
    def get_metric(self):
        pass