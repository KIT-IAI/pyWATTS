import logging
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd

from pywatts.summaries.metric_base import MetricBase
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


class F1Score(MetricBase):
    """
    Module to calculate the F1Score

    :param offset: Offset, which determines the number of ignored values in the beginning for calculating the MAE.
                   Default 0
    :type offset: int
    :param filter_method: Filter which should performed on the data before calculating the MAE.
    :type filter_method: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
    :param cuts: A list of Tuple of pd.Timestamps which specify intervals on which the metric should be calculated.
    :type cuts: List[Tuple[pd.Timestamp, pd.Timestamp]]
    :param average: The average param for the f1 score sklearn implementation
    :type average: str
    """

    def __init__(self, name: str = "F1Score", average="micro",  **kwargs):
        super().__init__(name=name, **kwargs)
        self.average = average


    def get_params(self) -> Dict[str, object]:
        """
        Returns a dict of parameters used in the Metric.

        :return: Parameters set for the Metric
        :rtype: Dict[str, object]
        """
        return {"offset": self.offset,
                "cuts": self.cuts,
                "average":self.average}

    def set_params(self, offset: Optional[int] = None, cuts=Optional[List[Tuple[pd.Timestamp, pd.Timestamp]]],
                   average:str=None):
        """
        Set parameters of the Metric.

        :param offset: Offset, which determines the number of ignored values in the beginning for calculating the Metric.
        :type offset: int
        :param cuts: The cutouts on which the metric should be additionally calculated.
        :type cuts: List[Tuple[pd.Timestamp, pd.Timestamp]]
        :param average: The average param for the f1 score sklearn implementation
        :type average: str
        """
        if offset is not None:
            self.offset = offset
        if cuts is not None:
            self.cuts = cuts
        if average is not None:
            self.average = average

    def _apply_metric(self, p, t):
        """
        Applies the metric.
        """
        return f1_score(p, t, average=self.average)
