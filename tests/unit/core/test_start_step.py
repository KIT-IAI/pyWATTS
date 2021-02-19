import unittest

from pywatts.core.start_step import StartStep


class TestStartStep(unittest.TestCase):

    def test_load(self):
        params = {
            # TODO: Review @benheid: 
            # The index is passed to the load method as cls(params['index']).
            # What does it do and is that correct? Assume id for Dataframe/Dataset?
            "index": "SomeIndex",
            "target_ids": {},
            # TODO: Review @benheid:
            # Test fails because input_ids expected to by list type but is dict.
            # Also, target_ids is still list. Correct?
            "input_ids": {},
            "id": -1,
            'computation_mode': 4,
            "module": "pywatts.core.start_step",
            "class": "StartStep",
            "name": "StartStep",
            "last":False
        }
        # TODO Review @benheid:
        # StartStep awaits index and also needs to pass an index as params dict.
        # Is that correct? Just passing None for StartStep in test.
        step = StartStep(None).load(params, None, None, None, None)
        json = step.get_json("file")
        self.assertEqual(params, json)
