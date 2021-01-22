import unittest

import numpy as np
import pandas as pd
import xarray as xr

from pywatts.core.exceptions.wrong_parameter_exception import WrongParameterException
from pywatts.modules.calendar_extraction import CalendarExtraction


class TestCalendarExtraction(unittest.TestCase):
    def setUp(self):
        start_time = "2000-01-01"
        times = 365
        self.da = xr.DataArray(data=pd.date_range(start_time, freq="D", periods=times))
        self.calendar_features = ["year", "month", "day", "weekday", "hour", "weekend", "workday", "holiday"]

    def tearDown(self):
        self.da = None

    def test_get_set_params(self):
        # define parameters and check if they are set correctly
        calendar_feature = "year"
        encoding = "sine"
        continent = "EUROPE"
        country = "Germany"

        params = {
            "calendar_feature": calendar_feature,
            "encoding": encoding,
            "continent": continent,
            "country": country
        }
        calendar_extraction = CalendarExtraction(**params)
        self.assertEqual(params, calendar_extraction.get_params())

        # define new params and try to set them
        new_params = {
            "calendar_feature": "day",
            "encoding": "numerical",
            "continent": "ASIA",
            "country": "China"
        }
        calendar_extraction.set_params(**new_params)
        self.assertEqual(new_params, calendar_extraction.get_params())

    def test_invalid_input(self):
        # test invalid continent and country inputs and also correct ones.
        try:
            CalendarExtraction(continent="Europe", country="Germany")
            CalendarExtraction(continent="europe", country="Germany")
            CalendarExtraction(continent="USA", country="UnitedStates")
            CalendarExtraction(continent="usa", country="UnitedStates")
            CalendarExtraction(continent="ASIA", country="China")
            CalendarExtraction(continent="asia", country="China")
            # try some specific region
            CalendarExtraction(continent="USA", country="Alabama")
            CalendarExtraction(continent="Europe", country="BadenWurttemberg")
        except Exception:
            self.fail("Assume valid continent and country should pass.")
        # assume wrong continent or country will fail
        self.assertRaises(WrongParameterException, CalendarExtraction, continent="Asia", country="Germany")
        self.assertRaises(WrongParameterException, CalendarExtraction, continent="Asia", country="USA")
        self.assertRaises(WrongParameterException, CalendarExtraction, continent="Europe", country="USA")
        # assume invalid continent or country will fail
        self.assertRaises(WrongParameterException, CalendarExtraction, continent="Europe", country="UNKNOWN")
        self.assertRaises(WrongParameterException, CalendarExtraction, continent="UNKNOWN", country="Germany")
        self.assertRaises(WrongParameterException, CalendarExtraction, continent="UNKNOWN", country="UNKNOWN")

    def test_numerical_encoding(self):
        # check if features are encoded correctly for numerical encoding
        for calendar_feature in self.calendar_features:
            calendar_extraction = CalendarExtraction(calendar_feature=calendar_feature, encoding="numerical")
            x = calendar_extraction.transform(self.da)
            if calendar_feature == "hour":
                self.assertEqual(x[0], 0, "Expecting hour feature starting from 0.")
            if calendar_feature == "day":
                self.assertEqual(x[0], 0, "Expecting day feature starting from 0.")
                self.assertEqual(x[30], 30, "Expecting day feature starting from 0.")
                self.assertEqual(x[31], 0, "Expecting day feature starting from 0.")
            if calendar_feature == "weekday":
                self.assertEqual(x[0], 5, "Expecting weekday feature starting from 0.")
                self.assertEqual(x[1], 6, "Expecting weekday feature starting from 0.")
                self.assertEqual(x[2], 0, "Expecting weekday feature starting from 0.")
            if calendar_feature == "month":
                self.assertEqual(x[0], 0, "Expecting month feature starting from 0.")
                self.assertEqual(x[30], 0, "Expecting month feature starting from 0.")
                self.assertEqual(x[31], 1, "Expecting month feature starting from 0.")
            if calendar_feature == "year":
                self.assertTrue((np.unique(x) == 2000).all())

            if calendar_feature == "workday":
                self.assertEqual(x[0], 0, "Expect weekend isn't a workday.")
                self.assertEqual(x[1], 0, "Expect weekend isn't a workday.")
                self.assertEqual(x[2], 1, "Expect monday is a workday.")
            if calendar_feature == "weekend":
                self.assertEqual(x[0], 1, "Weekend not correct recognized (weekend should be one).")
                self.assertEqual(x[1], 1, "Weekend not correct recognized (weekend should be one).")
                self.assertEqual(x[2], 0, "Weekend not correct recognized (Monday shouldn't be one).")
            if calendar_feature == "holiday":
                self.assertEqual(x[0], 1, "Expect 01.01.2000 is a holiday.")
                self.assertEqual(x[1], 0, "Expect sunday the 02.01.2000 isn't a holiday.")

    def test_sine_encoding(self):
        # check if features are encoded correctly for sine encoding
        for calendar_feature in self.calendar_features:
            calendar_extraction = CalendarExtraction(calendar_feature=calendar_feature, encoding="sine")
            x = calendar_extraction.transform(self.da)
            if calendar_feature == "hour":
                self.assertEqual(x[0], np.sin(2 * np.pi * 0 / 23), "Expecting hour feature starting from 0.")
            if calendar_feature == "day":
                self.assertEqual(x[0], np.sin(2 * np.pi * 0 / 30), "Expecting day feature starting from 0.")
                self.assertEqual(x[30], np.sin(2 * np.pi * 30 / 30), "Expecting day feature starting from 0.")
                self.assertEqual(x[31], np.sin(2 * np.pi * 0 / 30), "Expecting day feature starting from 0.")
            if calendar_feature == "weekday":
                self.assertEqual(x[0], np.sin(2 * np.pi * 5 / 6), "Expecting weekday feature starting from 0.")
                self.assertEqual(x[1], np.sin(2 * np.pi * 6 / 6), "Expecting weekday feature starting from 0.")
                self.assertEqual(x[2], np.sin(2 * np.pi * 0 / 6), "Expecting weekday feature starting from 0.")
            if calendar_feature == "month":
                self.assertEqual(x[0], np.sin(2 * np.pi * 0 / 11), "Expecting month feature starting from 0.")
                self.assertEqual(x[30], np.sin(2 * np.pi * 0 / 11), "Expecting month feature starting from 0.")
                self.assertEqual(x[31], np.sin(2 * np.pi * 1 / 11), "Expecting month feature starting from 0.")
            if calendar_feature == "year":
                self.assertTrue((np.unique(x) == 2000).all())

            if calendar_feature == "workday":
                self.assertEqual(x[0], 0, "Expect weekend isn't a workday.")
                self.assertEqual(x[1], 0, "Expect weekend isn't a workday.")
                self.assertEqual(x[2], 1, "Expect monday is a workday.")
            if calendar_feature == "weekend":
                self.assertEqual(x[0], 1, "Weekend not correct recognized (weekend should be one).")
                self.assertEqual(x[1], 1, "Weekend not correct recognized (weekend should be one).")
                self.assertEqual(x[2], 0, "Weekend not correct recognized (Monday shouldn't be one).")
            if calendar_feature == "holiday":
                self.assertEqual(x[0], 1, "Expect 01.01.2000 is a holiday.")
                self.assertEqual(x[1], 0, "Expect sunday the 02.01.2000 isn't a holiday.")
