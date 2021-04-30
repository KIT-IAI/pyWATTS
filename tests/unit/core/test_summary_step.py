import os
import unittest
from unittest.mock import MagicMock, patch, call

import numpy as np
import pandas as pd
import xarray as xr

from pywatts.core.computation_mode import ComputationMode
from pywatts.core.summary_step import SummaryStep


class TestSummaryStep(unittest.TestCase):
    def setUp(self) -> None:
        self.module_mock = MagicMock()
        self.module_mock.transform.return_value = "#I AM MARKDOWN"
        self.module_mock.name = "test"

        self.result_mock = MagicMock()
        self.step_mock = MagicMock()
        self.step_mock.get_result.return_value = self.result_mock
        self.step_mock.id = 2
        self.fm_mock = MagicMock()
        self.summary = SummaryStep(self.module_mock, {"input": self.step_mock}, self.fm_mock)

    def tearDown(self) -> None:
        self.module_mock = None
        self.step_mock = None
        self.summary = None
        self.fm_mock = None

    def test_get_summary(self):
        result = self.summary.get_summary()

        self.step_mock.get_result.assert_called_once_with(None, None)
        self.module_mock.transform.assert_called_once_with(self.fm_mock, input=self.result_mock)
        self.assertEqual(result, "#I AM MARKDOWN")

    def test_load(self):
        self.fail()

    def test_store(self):
        self.fail()
