from unittest.mock import MagicMock, patch, call
from unittest import TestCase

import numpy as np
import pandas as pd
import xarray as xr
import pywatts.summaries
from pywatts.summaries import TSNESummary


class TestTSNESummary(TestCase):
    def setUp(self) -> None:
        self.tsne = TSNESummary(max_points=100, tsne_params={}, all_in_one_plot=True)

    def tearDown(self) -> None:
        self.tsne = None

    def test_get_set_params(self):
        self.assertEqual(
            self.tsne.get_params(),
            {"max_points": 100, "tsne_params": {}, "all_in_one_plot": True}
        )
        self.tsne.set_params(max_points=42, all_in_one_plot=False, tsne_params="BLUB")
        self.assertEqual(
            self.tsne.get_params(),
            {"max_points": 42, "tsne_params": "BLUB", "all_in_one_plot": False}
        )

    @patch("pywatts.summaries.tsne_visualisation.TSNE")
    @patch("pywatts.summaries.tsne_visualisation.tikzplotlib")
    @patch("pywatts.summaries.tsne_visualisation.plt")
    def test_transform_all(self, plt_mock, tikz_mock, tsne_mock):
        time = pd.to_datetime(['2015-06-03 00:00:00', '2015-06-03 01:00:00',
                               '2015-06-03 02:00:00', '2015-06-03 03:00:00',
                               '2015-06-03 04:00:00'])

        test_data = xr.Dataset({"testCol": (["time", "horizon"], xr.DataArray([[-2], [-1], [0], [1], [2]]).data),
                                "predictCol1": (["time", "horizon"], xr.DataArray([[2], [-3], [3], [1], [-2]]).data),
                                "predictCol2": (["time", "horizon"], xr.DataArray([[-2], [-1], [0], [1], [2]]).data),
                                "time": time})

        fm_mock = MagicMock()
        fm_mock.get_path.return_value = "HELLO"
        tsne_instance_mock = MagicMock()
        tsne_mock.return_value = tsne_instance_mock

        test_result = self.tsne.transform(file_manager=fm_mock, gt=test_data['testCol'],
                                          pred1=test_data['predictCol1'], pred2=test_data['predictCol2'])

        tsne_mock.assert_called_once()
        tsne_instance_mock.fit_transform.assert_called_once()
        np.testing.assert_equal(np.sort(tsne_instance_mock.fit_transform.call_args.args[0].flatten()),
                                np.sort(np.array([-2, -1, 0, 1, 2, 2, -3, 3, 1, -2, -2, -1, 0, 1, 2])))
        tikz_mock.save.assert_called_once_with("HELLO")
        plt_mock.savefig.assert_called_once_with("HELLO")
        fm_mock.get_path.assert_has_calls(calls=[call('tsne/TSNE.png'), call('tsne/TSNE.tex')])

        self.assertEqual(test_result.k_v, {'Path PNG': 'HELLO\n', 'Path Tex': 'HELLO\n'})

    @patch("pywatts.summaries.tsne_visualisation.TSNE")
    @patch("pywatts.summaries.tsne_visualisation.tikzplotlib")
    @patch("pywatts.summaries.tsne_visualisation.plt")
    def test_transform_multiple_plots(self, plt_mock, tikz_mock, tsne_mock):
        self.tsne.set_params(all_in_one_plot=False)
        time = pd.to_datetime(['2015-06-03 00:00:00', '2015-06-03 01:00:00',
                               '2015-06-03 02:00:00', '2015-06-03 03:00:00',
                               '2015-06-03 04:00:00'])

        test_data = xr.Dataset({"testCol": (["time", "horizon"], xr.DataArray([[-2], [-1], [0], [1], [2]]).data),
                                "predictCol1": (["time", "horizon"], xr.DataArray([[2], [-3], [3], [1], [-2]]).data),
                                "predictCol2": (["time", "horizon"], xr.DataArray([[-2], [-1], [0], [1], [2]]).data),
                                "time": time})

        fm_mock = MagicMock()
        fm_mock.get_path.return_value = "HELLO"
        tsne_instance_mock = MagicMock()
        tsne_mock.return_value = tsne_instance_mock

        test_result = self.tsne.transform(file_manager=fm_mock, gt=test_data['testCol'],
                                          pred1=test_data['predictCol1'], pred2=test_data['predictCol2'])

        tsne_mock.assert_called()
        tsne_instance_mock.fit_transform.assert_called()
        print(tsne_instance_mock.fit_transform.call_args_list[0].args)
        np.testing.assert_equal(np.sort(tsne_instance_mock.fit_transform.call_args_list[0].args[0].flatten()),
                                np.sort(np.array([-2, -1, 0, 1, 2, 2, -3, 3, 1, -2])))
        np.testing.assert_equal(np.sort(tsne_instance_mock.fit_transform.call_args_list[1].args[0].flatten()),
                                np.sort(np.array([-2, -1, 0, 1, 2, -2, -1, 0, 1, 2])))

        tikz_mock.save.assert_called()
        plt_mock.savefig.assert_called()
        fm_mock.get_path.assert_has_calls(calls=[call('tsne/TSNE_pred1.png'), call('tsne/TSNE_pred1.tex'),
                                                 call('tsne/TSNE_pred2.png'), call('tsne/TSNE_pred2.tex')])
        self.assertEqual(test_result.k_v, {'Path PNG pred1': 'HELLO\n', 'Path Tex pred1': 'HELLO\n',
                                           'Path PNG pred2': 'HELLO\n', 'Path Tex pred2': 'HELLO\n'})

    @patch("pywatts.summaries.tsne_visualisation.TSNE")
    @patch("pywatts.summaries.tsne_visualisation.tikzplotlib")
    @patch("pywatts.summaries.tsne_visualisation.plt")
    def test_masked(self, plt_mock, tikz_mock, tsne_mock):
        self.tsne.set_params(all_in_one_plot=False)
        time = pd.to_datetime(['2015-06-03 00:00:00', '2015-06-03 01:00:00',
                               '2015-06-03 02:00:00', '2015-06-03 03:00:00',
                               '2015-06-03 04:00:00'])

        test_data = xr.Dataset({"testCol": (["time", "horizon"], xr.DataArray([[1], [2], [3], [4], [5]]).data),
                                "predictCol1": (["time", "horizon"], xr.DataArray([[6], [6], [6], [6], [6]]).data),
                                "masked_gt": ("time", xr.DataArray([True, False, False, True, False]).data),
                                "masked_col1": ("time", xr.DataArray([True, True, False, False, True]).data),
                                "time": time
                                })

        fm_mock = MagicMock()
        fm_mock.get_path.return_value = "HELLO"
        tsne_instance_mock = MagicMock()
        tsne_mock.return_value = tsne_instance_mock

        test_result = self.tsne.transform(file_manager=fm_mock, gt=test_data['testCol'], pred1=test_data['predictCol1'],
                                          gt_masked=test_data["masked_gt"], pred1_masked=test_data["masked_col1"])

        tsne_mock.assert_called()
        tsne_instance_mock.fit_transform.assert_called()
        np.testing.assert_equal(np.sort(tsne_instance_mock.fit_transform.call_args_list[0].args[0].flatten()),
                                np.sort(np.array([1, 4, 6, 6])))

        tikz_mock.save.assert_called()
        plt_mock.savefig.assert_called()
        fm_mock.get_path.assert_has_calls(calls=[call('tsne/TSNE_pred1.png'), call('tsne/TSNE_pred1.tex')])
        self.assertEqual(test_result.k_v, {'Path PNG pred1': 'HELLO\n', 'Path Tex pred1': 'HELLO\n'})
