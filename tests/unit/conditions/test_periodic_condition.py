import unittest
from unittest.mock import MagicMock

import pytest

from pywatts.conditions.periodic_condition import PeriodicCondition
from pywatts.core.exceptions.step_creation_exception import StepCreationException


class TestPeriodicCondition(unittest.TestCase):
    def test_evaluate(self):
        periodic_condition = PeriodicCondition(2)
        self.assertFalse(periodic_condition.evaluate())
        self.assertTrue(periodic_condition.evaluate())
        self.assertFalse(periodic_condition.evaluate())

    def test_added_with_too_much_inputs(self):
        with pytest.raises(StepCreationException):
            PeriodicCondition()(x=MagicMock())
