from pywatts.core.base import ConditionObject
from pywatts.core.step import Step
from pywatts.core.step_information import StepInformation


class PeriodicCondition(ConditionObject):
    """
    This Condition is raised after each num_steps.
    :param num_steps: After num_steps the periodicCondition should be True.
    :type num_steps: int
    :param name: The name of the PeriodicCondition.
    :type name: str
    """

    def __init__(self, num_steps=10, name="PeriodicCondition"):
        super().__init__(name=name)
        self.num_steps = num_steps
        self.counter = 0

    def evaluate(self):
        """
        Returns True if it is num_steps times called else False.
        """
        self.counter += 1
        self.counter = self.counter % self.num_steps

        if self.counter == 0:
            return True
        else:
            return False
