import unittest

from pywatts.core.start_step import StartStep


class TestStartStep(unittest.TestCase):

    def test_load(self):
        params = {
            "index": "SomeIndex",
            "target_ids": {},
            "input_ids": {},
            "id": -1,
            'computation_mode': 4,
            "module": "pywatts.core.start_step",
            "class": "StartStep",
            "name": "StartStep",
            "last":False
        }
        step = StartStep(None).load(params, None, None, None, None)
        json = step.get_json("file")
        self.assertEqual(params, json)
