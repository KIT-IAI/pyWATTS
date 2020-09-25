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
        self.dataset = xr.Dataset({
            }, coords={
                "time": pd.date_range(start_time, freq="D", periods=times)
            }
        )

    def tearDown(self):
        self.dataset = None

    def test_get_set_params(self):
        # define parameters and check if they are set correctly
        time_index = "TIME_INDEX"
        encoding = "sine"
        prefix = "PREFIX"
        continent = "EUROPE"
        country = "Germany"

        params = {
            "time_index": time_index,
            "encoding": encoding,
            "prefix": prefix,
            "continent": continent,
            "country": country
        }
        calendar_extraction = CalendarExtraction(**params)
        self.assertEqual(params, calendar_extraction.get_params())

        # define new params and try to set them
        new_params = {
            "time_index": "NEW_INDEX",
            "encoding": "numerical",
            "prefix": "",
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

    def test_features_added(self):
        # check if features are added to xarray dataset
        calendar_extraction = CalendarExtraction(time_index="time")
        x = calendar_extraction.transform(self.dataset)
        self.assertIn("year", x, "Expected year to be added to xarray dataset.")
        self.assertIn("month", x, "Expected month to be added to xarray dataset.")
        self.assertIn("day", x, "Expected day to be added to xarray dataset.")
        self.assertIn("weekday", x, "Expected weekday to be added to xarray dataset.")
        self.assertIn("hour", x, "Expected hour to be added to xarray dataset.")
        self.assertIn("weekend", x, "Expected weekend to be added to xarray dataset.")
        self.assertIn("workday", x, "Expected workday to be added to xarray dataset.")
        self.assertIn("holiday", x, "Expected holiday to be added to xarray dataset.")

    def test_features_added_with_prefix(self):
        # check if features are added with prefix to xarray dataset
        prefix = "prefix_"
        calendar_extraction = CalendarExtraction(time_index="time", prefix=prefix)
        x = calendar_extraction.transform(self.dataset)
        self.assertIn(f"{prefix}year", x, "Expected year with prefix to be added to xarray dataset.")
        self.assertIn(f"{prefix}month", x, "Expected month with prefix to be added to xarray dataset.")
        self.assertIn(f"{prefix}day", x, "Expected day with prefix to be added to xarray dataset.")
        self.assertIn(f"{prefix}weekday", x, "Expected weekday with prefix to be added to xarray dataset.")
        self.assertIn(f"{prefix}hour", x, "Expected hour with prefix to be added to xarray dataset.")
        self.assertIn(f"{prefix}weekend", x, "Expected weekend with prefix to be added to xarray dataset.")
        self.assertIn(f"{prefix}workday", x, "Expected workday with prefix to be added to xarray dataset.")
        self.assertIn(f"{prefix}holiday", x, "Expected holiday with prefix to be added to xarray dataset.")

    def test_numerical_encoding(self):
        # check if features are encoded correctly for numerical encoding
        calendar_extraction = CalendarExtraction(time_index="time", encoding="numerical")
        x = calendar_extraction.transform(self.dataset)
        self.assertTrue((np.unique(x.year) == 2000).all())
        self.assertEqual(x.month[0], 0, "Expecting month feature starting from 0.")
        self.assertEqual(x.month[30], 0, "Expecting month feature starting from 0.")
        self.assertEqual(x.month[31], 1, "Expecting month feature starting from 0.")
        self.assertEqual(x.day[0], 0, "Expecting day feature starting from 0.")
        self.assertEqual(x.day[30], 30, "Expecting day feature starting from 0.")
        self.assertEqual(x.day[31], 0, "Expecting day feature starting from 0.")
        self.assertEqual(x.weekday[0], 5, "Expecting weekday feature starting from 0.")
        self.assertEqual(x.weekday[1], 6, "Expecting weekday feature starting from 0.")
        self.assertEqual(x.weekday[2], 0, "Expecting weekday feature starting from 0.")
        self.assertEqual(x.hour[0], 0, "Expecting hour feature starting from 0.")
        self.assertEqual(x.weekend[0], 1, "Weekend not correct recognized (weekend should be one).")
        self.assertEqual(x.weekend[1], 1, "Weekend not correct recognized (weekend should be one).")
        self.assertEqual(x.weekend[2], 0, "Weekend not correct recognized (Monday shouldn't be one).")
        self.assertEqual(x.workday[0], 0, "Expect weekend isn't a workday.")
        self.assertEqual(x.workday[1], 0, "Expect weekend isn't a workday.")
        self.assertEqual(x.workday[2], 1, "Expect monday is a workday.")
        self.assertEqual(x.holiday[0], 1, "Expect 01.01.2000 is a holiday.")
        self.assertEqual(x.holiday[1], 0, "Expect sunday the 02.01.2000 isn't a holiday.")

    def test_sine_encoding(self):
        # check if features are encoded correctly for sine encoding
        calendar_extraction = CalendarExtraction(time_index="time", encoding="sine")
        x = calendar_extraction.transform(self.dataset)

        # expect year isn't encoded with sine because it isn't a cyclic feature
        self.assertTrue((np.unique(x.year) == 2000).all())
        self.assertEqual(x.month[0], np.sin(2 * np.pi * 0 / 11), "Expecting month feature starting from 0.")
        self.assertEqual(x.month[30], np.sin(2 * np.pi * 0 / 11), "Expecting month feature starting from 0.")
        self.assertEqual(x.month[31], np.sin(2 * np.pi * 1 / 11), "Expecting month feature starting from 0.")
        self.assertEqual(x.day[0], np.sin(2 * np.pi * 0 / 30), "Expecting day feature starting from 0.")
        self.assertEqual(x.day[30], np.sin(2 * np.pi * 30 / 30), "Expecting day feature starting from 0.")
        self.assertEqual(x.day[31], np.sin(2 * np.pi * 0 / 30), "Expecting day feature starting from 0.")
        self.assertEqual(x.weekday[0], np.sin(2 * np.pi * 5 / 6), "Expecting weekday feature starting from 0.")
        self.assertEqual(x.weekday[1], np.sin(2 * np.pi * 6 / 6), "Expecting weekday feature starting from 0.")
        self.assertEqual(x.weekday[2], np.sin(2 * np.pi * 0 / 6), "Expecting weekday feature starting from 0.")
        self.assertEqual(x.hour[0], np.sin(2 * np.pi * 0 / 23), "Expecting hour feature starting from 0.")

        # expect weekend, workday and holiday not encoded with sine because they are boolean
        self.assertEqual(x.weekend[0], 1, "Weekend not correct recognized (weekend should be one).")
        self.assertEqual(x.weekend[1], 1, "Weekend not correct recognized (weekend should be one).")
        self.assertEqual(x.weekend[2], 0, "Weekend not correct recognized (Monday shouldn't be one).")
        self.assertEqual(x.workday[0], 0, "Expect weekend isn't a workday.")
        self.assertEqual(x.workday[1], 0, "Expect weekend isn't a workday.")
        self.assertEqual(x.workday[2], 1, "Expect monday is a workday.")
        self.assertEqual(x.holiday[0], 1, "Expect 01.01.2000 is a holiday.")
        self.assertEqual(x.holiday[1], 0, "Expect sunday the 02.01.2000 isn't a holiday.")
