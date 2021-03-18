import unittest
from datetime import datetime
from unittest.mock import patch, call

from pywatts.core.exceptions.io_exceptions import IOException
from pywatts.core.filemanager import FileManager


class TestFilemanager(unittest.TestCase):

    @patch("pywatts.core.filemanager.datetime")
    @patch("pywatts.core.filemanager.os")
    def test_get_path(self, os_mock, datetime_mock):
        datetime_mock.now.return_value = datetime(2442, 12, 24, 1, 2, 3)
        os_mock.path.join.side_effect = ["testpath/2442_12_24_01_02_03",
                                         "testpath/2442_12_24_01_02_03/my_result",
                                         "testpath/2442_12_24_01_02_03/my_result/result.csv"]

        self.filemanager = FileManager("testpath")

        os_mock.path.isfile.return_value = False
        os_mock.path.split.return_value = ("", "result.csv")

        filepath = self.filemanager.get_path("result.csv", "my_result")

        join_calls = [call("testpath", "2442_12_24_01_02_03"),
                      call("testpath/2442_12_24_01_02_03", "my_result"),
                      call("testpath/2442_12_24_01_02_03/my_result", "result.csv")
                      ]
        os_mock.path.join.assert_has_calls(join_calls)
        os_mock.path.split.assert_called_once_with("result.csv")

        self.assertEqual(filepath, "testpath/2442_12_24_01_02_03/my_result/result.csv")

    @patch("pywatts.core.filemanager.datetime")
    @patch("pywatts.core.filemanager.os")
    def test_get_path_filename_with_path(self, os_mock, datetime_mock):
        datetime_mock.now.return_value = datetime(2442, 12, 24, 1, 2, 3)
        os_mock.path.join.side_effect = ["testpath/2442_12_24_01_02_03",
                                         "testpath/2442_12_24_01_02_03/my_result",
                                         "testpath/2442_12_24_01_02_03/my_result/result.csv"]

        self.filemanager = FileManager("testpath")

        os_mock.path.isfile.return_value = False
        os_mock.path.split.return_value = ("", "result.csv")

        filepath = self.filemanager.get_path("path/result.csv", "my_result")

        self.assertEqual(filepath, "testpath/2442_12_24_01_02_03/my_result/result.csv")

    @patch("pywatts.core.filemanager.datetime")
    @patch("pywatts.core.filemanager.os")
    def test_not_allowed_filetype(self, os_mock, datetime_mock):
        datetime_mock.now.return_value = datetime(2442, 12, 24, 1, 2, 3)
        os_mock.path.join.side_effect = ["testpath/2442_12_24_01_02_03"]

        self.filemanager = FileManager("testpath")

        os_mock.path.isfile.return_value = False
        os_mock.path.split.return_value = ("", "result.test")
        with self.assertRaises(IOException) as cm:
            self.filemanager.get_path("result.test")
        self.assertEqual(cm.exception.args,
                         ("test is not an allowed file type. Allowed types are ['png', 'csv', 'xlsx', "
                          "'pickle', 'tex', 'json', 'h5', 'pt', 'md'].",))

    @patch("pywatts.core.filemanager.logger")
    @patch("pywatts.core.filemanager.datetime")
    @patch("pywatts.core.filemanager.os")
    def test_duplicate_filename(self, os_mock, datetime_mock, logger_mock):
        datetime_mock.now.return_value = datetime(2442, 12, 24, 1, 2, 3)
        os_mock.path.join.side_effect = ["testpath/2442_12_24_01_02_03",
                                         "testpath/2442_12_24_01_02_03/my_result",
                                         "testpath/2442_12_24_01_02_03/my_result/result.csv"]

        os_mock.path.splitext.return_value = ("testpath/2442_12_24_01_02_03/my_result/result", "csv")

        self.filemanager = FileManager("testpath")

        os_mock.path.isfile.return_value = True
        os_mock.path.split.return_value = ("", "result.csv")
        result = self.filemanager.get_path("result.csv")
        self.assertEqual(result, 'testpath/2442_12_24_01_02_03/my_result/result_1.csv')
        logger_mock.info.assert_called_with('File %s already exists. We appended %s to the name',
                                            'testpath/2442_12_24_01_02_03/my_result', 1)
