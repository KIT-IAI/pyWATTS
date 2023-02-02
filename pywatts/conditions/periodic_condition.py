import pandas as pd

from pywatts.core.base_condition import BaseCondition


class PeriodicCondition(BaseCondition):
    """
    This Condition is raised after each num_steps.
    :param num_steps: After num_steps the periodicCondition should be True.
    :type num_steps: int
    :param name: The name of the PeriodicCondition.
    :type name: str
    """

    def __init__(self, num_steps=10, refit_batch: pd.Timedelta = pd.Timedelta(hours=10), refit_params: dict = None,
                 name="PeriodicCondition"):
        super().__init__(name=name, refit_batch=refit_batch, refit_params=refit_params)
        self.num_steps = num_steps
        self.counter = 0

    def evaluate(self, start, end):
        """
        Returns True if it is num_steps times called else False.
        :param start: start of the batch
        :type start: pd.Timestamp
        :param end: end of the batch
        :type end: pd.Timestamp
        """
        if not self._is_evaluated(end):
            self.counter += 1
        self.counter = self.counter % self.num_steps

        if self.counter == 0:
            print(f"{self.name}: refit with refit batch {self.refit_batch}")
            return True
        else:
            return False
