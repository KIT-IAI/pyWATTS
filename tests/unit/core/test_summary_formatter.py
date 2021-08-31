import unittest
from unittest.mock import MagicMock, patch, call

import pandas as pd
import numpy as np

from pywatts.core.summary_object import SummaryObjectList, SummaryCategory, SummaryObjectTable
from pywatts.core.summary_formatter import SummaryMarkdown, SummaryJSON


class TestSummaryMarkdown(unittest.TestCase):

    @patch("builtins.open")
    def test_create_summary(self, open_mock):
        summary_object = SummaryObjectList("name", category=SummaryCategory.FitTime,
                                           additional_information="additional_information")
        summary_object.set_kv("a", 1)
        summary_object.set_kv("b", 2)

        summary_object2 = SummaryObjectTable("name", category=SummaryCategory.Summary,
                                             additional_information="additional_information")
        summary_object2.set_kv("a", pd.DataFrame({"a": [1, 1]}))
        summary_object2.set_kv("b", pd.DataFrame({"b": [2, 2]}))

        fm_mock = MagicMock()

        summary_formatter = SummaryMarkdown()

        expected_result = '# Summary: \n## Summary\n### name\nadditional_information\n#### a\n ' \
                          '|   0 |   1 |\n|-----|-----|\n|   0 |   1 |\n|   1 |   1 |\n' \
                          '#### b\n |   0 |   1 |\n|-----|-----|\n|   0 |   2 |\n|   1 |   2 |\n' \
                          '## FitTime\n### name\nadditional_information\n* a : 1\n* b : 2\n## TransformTime\n'
        fm_mock.get_path.return_value = "file.md"

        summary_string = summary_formatter.create_summary([summary_object, summary_object2], fm_mock)

        fm_mock.get_path.assert_called_once_with("summary.md")
        open_mock.assert_has_calls([call("file.md", "w")], any_order=True)

        self.assertEqual(expected_result, summary_string)


class TestSummaryJson(unittest.TestCase):

    @patch("builtins.open")
    def test_create_summary(self, open_mock):
        summary_object = SummaryObjectList("name", category=SummaryCategory.FitTime,
                                           additional_information="additional_information")
        summary_object.set_kv("a", 1)
        summary_object.set_kv("b", 2)

        summary_object2 = SummaryObjectTable("name", category=SummaryCategory.Summary,
                                             additional_information="additional_information")
        summary_object2.set_kv("a", np.array([1, 1]))
        summary_object2.set_kv("b", np.array([2, 2]))

        fm_mock = MagicMock()

        summary_formatter = SummaryJSON()

        expected_result = {'Summary': {
            'name': {'additional_information': 'additional_information', 'results': {'a': [1, 1], 'b': [2, 2]}}},
                           'FitTime': {'name': {'additional_information': 'additional_information',
                                                'results': {'a': 1, 'b': 2}}}, 'TransformTime': {}}
        fm_mock.get_path.return_value = "file.md"

        summary_json = summary_formatter.create_summary([summary_object, summary_object2], fm_mock)

        fm_mock.get_path.assert_called_once_with("summary.json")
        open_mock.assert_has_calls([call("file.md", "w")], any_order=True)

        self.assertEqual(expected_result, summary_json)
