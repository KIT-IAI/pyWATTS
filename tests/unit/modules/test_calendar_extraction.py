import unittest

import numpy as np
import pandas as pd
import xarray as xr

from pywatts.core.exceptions.wrong_parameter_exception import WrongParameterException
from pywatts.modules import CalendarExtraction, CalendarFeature


class TestCalendarExtraction(unittest.TestCase):
    def setUp(self):
        start_time = "2000-01-01"
        times = 365
        self.da = xr.DataArray(data=pd.date_range(start_time, freq="D", periods=times))
        self.calendar_features = [CalendarFeature.year, CalendarFeature.month, CalendarFeature.day,
                                  CalendarFeature.weekday, CalendarFeature.hour, CalendarFeature.workday,
                                  CalendarFeature.holiday, CalendarFeature.hour_sine, CalendarFeature.hour_cos,
                                  CalendarFeature.weekday_sine, CalendarFeature.weekday_cos, CalendarFeature.month_sine,
                                  CalendarFeature.month_cos, CalendarFeature.day_sine, CalendarFeature.day_cos,
                                  CalendarFeature.monday, CalendarFeature.tuesday, CalendarFeature.wednesday,
                                  CalendarFeature.thursday, CalendarFeature.saturday, CalendarFeature.sunday]

    def tearDown(self):
        self.da = None

    def test_get_set_params(self):
        # define parameters and check if they are set correctly
        continent = "EUROPE"
        country = "Germany"

        params = {
            "features": [CalendarFeature.year],
            "continent": continent,
            "country": country
        }
        calendar_extraction = CalendarExtraction(**params)
        self.assertEqual(params, calendar_extraction.get_params())

        # define new params and try to set them
        new_params = {
            "features": CalendarFeature.day,
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

    def test_encoding(self):
        # check if features are encoded correctly for numerical encoding
        for calendar_feature in self.calendar_features:
            calendar_extraction = CalendarExtraction(features=[calendar_feature])
            x = calendar_extraction.transform(self.da)
            if calendar_feature == CalendarFeature.hour:
                self.assertEqual(x[0], 0, "Expecting hour feature starting from 0.")
            if calendar_feature == CalendarFeature.day:
                self.assertEqual(x[0], 0, "Expecting day feature starting from 0.")
                self.assertEqual(x[30], 30, "Expecting day feature starting from 0.")
                self.assertEqual(x[31], 0, "Expecting day feature starting from 0.")
            if calendar_feature == CalendarFeature.weekday:
                self.assertEqual(x[0], 5, "Expecting weekday feature starting from 0.")
                self.assertEqual(x[1], 6, "Expecting weekday feature starting from 0.")
                self.assertEqual(x[2], 0, "Expecting weekday feature starting from 0.")
            if calendar_feature == CalendarFeature.month:
                self.assertEqual(x[0], 0, "Expecting month feature starting from 0.")
                self.assertEqual(x[30], 0, "Expecting month feature starting from 0.")
                self.assertEqual(x[31], 1, "Expecting month feature starting from 0.")
            if calendar_feature == CalendarFeature.year:
                self.assertTrue((np.unique(x) == 2000).all())
            if calendar_feature == CalendarFeature.workday:
                self.assertEqual(x[0], 0, "Expect weekend isn't a workday.")
                self.assertEqual(x[1], 0, "Expect weekend isn't a workday.")
                self.assertEqual(x[2], 1, "Expect monday is a workday.")
            if calendar_feature == CalendarFeature.weekend:
                self.assertEqual(x[0], 1, "Weekend not correct recognized (weekend should be one).")
                self.assertEqual(x[1], 1, "Weekend not correct recognized (weekend should be one).")
                self.assertEqual(x[2], 0, "Weekend not correct recognized (Monday shouldn't be one).")
            if calendar_feature == CalendarFeature.holiday:
                self.assertEqual(x[0], 1, "Expect 01.01.2000 is a holiday.")
                self.assertEqual(x[1], 0, "Expect sunday the 02.01.2000 isn't a holiday.")
            if calendar_feature == CalendarFeature.hour_sine:
                self.assertEqual(x[0], np.sin(np.pi * 2 * 0 / 23), "Expecting hour feature starting from 0.")
            if calendar_feature == CalendarFeature.day_sine:
                self.assertEqual(x[0], np.sin(np.pi * 2 * 0 / 31), "Expecting day feature starting from 0.")
                self.assertEqual(x[30], np.sin(np.pi * 2 * 30 / 31), "Expecting day feature starting from 0.")
                self.assertEqual(x[31], np.sin(np.pi * 2 * 0 / 3), "Expecting day feature starting from 0.")
            if calendar_feature == CalendarFeature.weekday_sine:
                self.assertEqual(x[0], np.sin(np.pi * 2 * 5 / 6), "Expecting weekday feature starting from 0.")
                self.assertEqual(x[1], np.sin(np.pi * 2 * 6 / 6), "Expecting weekday feature starting from 0.")
                self.assertEqual(x[2], np.sin(np.pi * 2 * 0 / 6), "Expecting weekday feature starting from 0.")
            if calendar_feature == CalendarFeature.month_sine:
                self.assertEqual(x[0], np.sin(np.pi * 2 * 0 / 11), "Expecting month feature starting from 0.")
                self.assertEqual(x[30], np.sin(np.pi * 2 * 0 / 11), "Expecting month feature starting from 0.")
                self.assertEqual(x[31], np.sin(np.pi * 2 * 1 / 11), "Expecting month feature starting from 0.")
            if calendar_feature == CalendarFeature.month_cos:
                self.assertEqual(x[0], np.cos(np.pi * 2 * 0 / 11), "Expecting month feature starting from 1.")
                self.assertEqual(x[30], np.cos(np.pi * 2 * 0 / 11), "Expecting month feature starting from 1.")
                self.assertEqual(x[31], np.cos(np.pi * 2 * 1 / 11), "Expecting month feature starting from 1.")
            if calendar_feature == CalendarFeature.hour_cos:
                self.assertEqual(x[0], np.cos(np.pi * 2 * 0 / 23), "Expecting hour feature starting from 1.")
            if calendar_feature == CalendarFeature.day_cos:
                self.assertEqual(x[0], np.cos(np.pi * 2 * 0 / 31), "Expecting day feature starting from 1.")
                self.assertEqual(x[30], np.cos(np.pi * 2 * 30 / 31), "Expecting day feature starting from 1.")
                self.assertEqual(x[31], np.cos(np.pi * 2 * 0 / 31), "Expecting day feature starting from 1.")
            if calendar_feature == CalendarFeature.weekday_cos:
                self.assertEqual(x[0], np.cos(np.pi * 2 * 5 / 6), "Expecting weekday feature starting from 1.")
                self.assertEqual(x[1], np.cos(np.pi * 2 * 6 / 6), "Expecting weekday feature starting from 1.")
                self.assertEqual(x[2], np.cos(np.pi * 2 * 0 / 6), "Expecting weekday feature starting from 1.")
            if calendar_feature == CalendarFeature.monday:
                self.assertEqual(x[0], 0)
                self.assertEqual(x[2], 1)
            if calendar_feature == CalendarFeature.tuesday:
                self.assertEqual(x[0], 0)
                self.assertEqual(x[3], 1)
            if calendar_feature == CalendarFeature.wednesday:
                self.assertEqual(x[0], 0)
                self.assertEqual(x[4], 1)
            if calendar_feature == CalendarFeature.thursday:
                self.assertEqual(x[0], 0)
                self.assertEqual(x[5], 1)
            if calendar_feature == CalendarFeature.friday:
                self.assertEqual(x[0], 0)
                self.assertEqual(x[6], 1)
            if calendar_feature == CalendarFeature.saturday:
                self.assertEqual(x[1], 0)
                self.assertEqual(x[0], 1)
            if calendar_feature == CalendarFeature.sunday:
                self.assertEqual(x[0], 0)
                self.assertEqual(x[1], 1)

