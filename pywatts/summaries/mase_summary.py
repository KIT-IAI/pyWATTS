import logging
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

from pywatts.summaries.metric_base import MetricBase

logger = logging.getLogger(__name__)


class MASE(MetricBase):
    """
    Module to calculate the Mean Absolute Scaled Error (MAPE)

    :param offset: Offset, which determines the number of ignored values in the beginning for calculating the MAPE.
                   Default 0
    :type offset: int
    :param filter_method: Filter which should performed on the data before calculating the MAPE.
    :type filter_method: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
    :param lag: The lag determines the persistence forecast that is used for scaling the error.
    :type lag: int
    """

    def __init__(self, name: str = "MASE", filter_method=None, offset: int = 0,
                 cuts: Optional[List[Tuple[pd.Timestamp, pd.Timestamp]]] = None, lag: int = 1):
        super().__init__(name=name, filter_method=filter_method, offset=offset, cuts=cuts)
        self.lag = lag

    def _apply_metric(self, p, t):
        return np.mean(np.abs(p[self.lag:] - t[self.lag:])) / np.mean(np.abs(t[:-self.lag] - t[self.lag:]))
