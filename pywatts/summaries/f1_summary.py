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

    :param offset: Offset, which determines the number of ignored values in the beginning for calculating the F1 Score.
                   Default 0
    :type offset: int
    :param filter_method: Filter which should performed on the data before calculating the F1 Score.
    :type filter_method: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
    :param cuts: A list of Tuples of pd.Timestamps which specify intervals on which the metric should be calculated.
    :type cuts: List[Tuple[pd.Timestamp, pd.Timestamp]]
    :param average: The average param for the f1 score sklearn implementation. See
                    `SKLearn https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html`_
    :type average: str
    """

    def __init__(self, name: str = "F1Score", average="micro", cuts=[], offset=0, filter_method=None):
        super().__init__(name=name, cuts=cuts, offset=offset, filter_method=filter_method)
        self.average = average

    def _apply_metric(self, p, t):
        """
        Applies the metric.
        """
        return f1_score(p, t, average=self.average)
