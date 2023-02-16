import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import xarray as xr

from pywatts.modules import DriftInformation, SyntheticConcecptDriftInsertion


class TestDriftInformation(unittest.TestCase):

    def test_get_drift(self):
        manipulator_mock = MagicMock()
        drift_information = DriftInformation(manipulator=manipulator_mock, length=2,
                                             position=pd.Timestamp(year=2042, month=12, day=24))
        manipulator_mock.return_value = np.array([[1], [2]])
        drift = drift_information.get_drift()

        np.testing.assert_array_equal(drift, np.array([1, 2]))
        manipulator_mock.assert_called_with(2)
        self.assertEqual(drift_information.position, pd.Timestamp(year=2042, month=12, day=24))
        self.assertEqual(drift_information.length, 2)


class TestCDInsertion(unittest.TestCase):

    def setUp(self):
        self.drift_information = MagicMock()
        self.drift_information.get_drift.return_value = np.array([1, 2])
        self.drift_information.position = pd.Timestamp(year=2000, month=1, day=1)
        self.drift_information.length = 2
        self.cd_insertion = SyntheticConcecptDriftInsertion(drift_information=[self.drift_information])

    def tearDown(self):
        self.manipulator_mock = None
        self.drift_information = None
        self.cd_insertion = None

    def test_set_get_params(self):
        self.assertEqual(
            self.cd_insertion.get_params(),
            {"drift_information": [self.drift_information],
            'name': 'Concept Drift Generation'}
        )
        drift_information = MagicMock()
        self.cd_insertion.set_params(drift_information=[drift_information])
        self.assertEqual(
            self.cd_insertion.get_params(),
            {"drift_information": [drift_information],'name': 'Concept Drift Generation'}
        )

    def test_add_one_cd(self):
        time = pd.date_range('2000-01-01', freq='24H', periods=7)
        da = xr.DataArray([1, 2, 3, 4, 5, 6, 7], dims=['time'], coords={"time": time})
        expected_result = xr.DataArray([2, 4, 5, 6, 7, 8, 9], dims=['time'], coords={"time": time})
        result = self.cd_insertion.transform(da)
        xr.testing.assert_equal(result, expected_result)
        self.drift_information.get_drift.assert_called_once_with()

    def test_add_multiple_cd(self):
        drift_information_2 = MagicMock()
        drift_information_2.get_drift.return_value = np.array([-2, -3, -4])
        drift_information_2.position = pd.Timestamp(year=2000, month=1, day=4)
        drift_information_2.length = 3
        self.cd_insertion = SyntheticConcecptDriftInsertion(drift_information=[self.drift_information,
                                                                                drift_information_2])
        time = pd.date_range('2000-01-01', freq='24H', periods=7)
        da = xr.DataArray([1, 2, 3, 4, 5, 6, 7], dims=['time'], coords={"time": time})

        expected_result = xr.DataArray([2, 4, 5, 4, 4, 4, 5], dims=['time'], coords={"time": time})
        result = self.cd_insertion.transform(da)
        xr.testing.assert_equal(result, expected_result)
        self.drift_information.get_drift.assert_called_once_with()
        drift_information_2.get_drift.assert_called_once_with()

    def test_add_drift_partial_outside_of_input_ts(self):
        drift_information = MagicMock()
        drift_information.get_drift.return_value = np.array([-2, -3, -4])
        drift_information.position = pd.Timestamp(year=2000, month=1, day=6)
        drift_information.length = 3
        self.cd_insertion = SyntheticConcecptDriftInsertion(drift_information=[drift_information])
        time = pd.date_range('2000-01-01', freq='24H', periods=7)
        da = xr.DataArray([1, 2, 3, 4, 5, 6, 7], dims=['time'], coords={"time": time})

        expected_result = xr.DataArray([1, 2, 3, 4, 5, 4, 4], dims=['time'], coords={"time": time})
        result = self.cd_insertion.transform(da)
        xr.testing.assert_equal(result, expected_result)
        drift_information.get_drift.assert_called_once_with()

    def test_add_drift_complete_outside_of_input_ts(self):
        drift_information = MagicMock()
        drift_information.get_drift.return_value = np.array([-2, -3, -4])
        drift_information.position = pd.Timestamp(year=2000, month=1, day=8)
        drift_information.length = 3
        self.cd_insertion = SyntheticConcecptDriftInsertion(drift_information=[drift_information])
        time = pd.date_range('2000-01-01', freq='24H', periods=7)
        da = xr.DataArray([1, 2, 3, 4, 5, 6, 7], dims=['time'], coords={"time": time})

        expected_result = xr.DataArray([1, 2, 3, 4, 5, 6, 7], dims=['time'], coords={"time": time})
        result = self.cd_insertion.transform(da)
        xr.testing.assert_equal(result, expected_result)
        # Is not called since complete drift is outside of the input data.
        drift_information.get_drift.assert_not_called()

    @patch('pywatts.modules.generation.synthetic_concept_drift.cloudpickle')
    @patch("builtins.open")
    def test_save(self, open_mock, cloudpickle_mock):
        fm_mock = MagicMock()
        fm_mock.get_path.return_value = os.path.join("path", f"{self.cd_insertion.name}_drift_information.pickle")
        json = self.cd_insertion.save(fm_mock)
        open_mock.assert_called_once_with(
            os.path.join("path", f"{self.cd_insertion.name}_drift_information.pickle"), 'wb')
        cloudpickle_mock.dump.assert_called_once()
        self.assertEqual(json, {
            "params": {},
            "name": self.cd_insertion.name,
            "class": "SyntheticConcecptDriftInsertion",
            "module": "pywatts.modules.generation.synthetic_concept_drift",
            "drift_information": os.path.join("path", f'{self.cd_insertion.name}_drift_information.pickle')
        })


    @patch('pywatts.modules.generation.synthetic_concept_drift.cloudpickle')
    @patch("builtins.open")
    def test_load(self, open_mock, cloudpickle_mock):
        self.cd_insertion.load({
            "params": {},
            "name": self.cd_insertion.name,
            "class": "SyntheticConcecptDriftGeneration",
            "module": "pywatts.modules.generation.synthetic_concept_drift",
            "drift_information": "path_to_drift_information.pickle"
        })
        open_mock.assert_called_once_with("path_to_drift_information.pickle", "rb")
        cloudpickle_mock.load.assert_called_once()
