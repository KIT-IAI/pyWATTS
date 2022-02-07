import logging

import numpy as np

from pywatts.summaries.metric_base import MetricBase

logger = logging.getLogger(__name__)


class MinErr(MetricBase):
    """
    Module to calculate the minimal absolute error

    :param offset: Offset, which determines the number of ignored values in the beginning for calculating the min.
                   Default 0
    :type offset: int
    :param filter_method: Filter which should performed on the data before calculating the min.
    :type filter_method: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
    """


    def _apply_metric(self, p, t):
        return np.min(np.abs((p - t)))

