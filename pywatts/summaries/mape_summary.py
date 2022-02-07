import logging

import numpy as np

from pywatts.summaries.metric_base import MetricBase

logger = logging.getLogger(__name__)


class MAPE(MetricBase):
    """
    Module to calculate the Mean Absolute Percentage Error (MAPE)

    :param offset: Offset, which determines the number of ignored values in the beginning for calculating the MAPE.
                   Default 0
    :type offset: int
    :param filter_method: Filter which should performed on the data before calculating the MAPE.
    :type filter_method: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
    """


    def _apply_metric(self, p, t):
        non_zero_indexes = t.nonzero()[0]
        return 100 * np.mean(np.abs((p[non_zero_indexes] - t[non_zero_indexes]) / t[non_zero_indexes]))
