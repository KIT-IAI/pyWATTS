import unittest

from pywatts.core.summary_object import SummaryObjectList, SummaryCategory


class TestSummaryObjectList(unittest.TestCase):

    def test_set_kv(self):
        summary_object = SummaryObjectList("name", category=SummaryCategory.FitTime,
                                           additional_information="additional_information")
        summary_object.set_kv("a", 1)
        summary_object.set_kv("b", 2)

        self.assertEqual("name", summary_object.name)
        self.assertEqual("additional_information", summary_object.additional_information)
        self.assertEqual(SummaryCategory.FitTime, summary_object.category)
        self.assertEqual({"a": 1, "b": 2}, summary_object.k_v)
