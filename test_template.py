import unittest

from pywatts.core.base import BaseTransformer, BaseEstimator

class TestTemplate(unittest.TestCase):
    """
    Template for testing an estimator.
    For testing a transformer delete the test_fit methods
    """

    def setUp(self):
        """
        This class should setUp all everything which is necessary to execute the tests.
        """

        #example
        # self.clock_shift = ClockShift(lag=10)

    def tearDown(self) -> None:
        """

        :return:
        """

    def test_get_params_default_parameter(self):
        """
        This class tests the method get_params and expects the default params

        Note, that it is often necessary to implement multiple tests for one method, for testing the edge cases.

        There exist multiple self.assert_* methods for checking the result against the exepected value.
        """

        # Example
        # params = self.clock_shift.get_params()
        # self.assertEqual(len(params), 2)
        # self.assertEqual(params, {"lag": 10,
        #                          "indeces": None})


    def test_get_params_non_default_parameter(self):
        """
        This class tests the method get_params and expects the non default params params
        """

        # Example
        # self.clock_shift = ClockShift(lag=10, index=["time", "test"])
        # params = self.clock_shift.get_params()
        # self.assertEqual(len(params), 2)
        # self.assertEqual(params, {"lag": 10,
        #                          "indeces": ["time", "test"]})



    def test_set_params(self):
        """
        Tests the set_params method
        """

    def test_fit(self):
        """
        Test the fit method
        """


    def test_transform(self):
        """
        Test the transform method.

        Often it is necessary to test, if methods of underlying libraries are called correctly. Therfore, you can use mocking. For an example see test_lineplotter.
        """
