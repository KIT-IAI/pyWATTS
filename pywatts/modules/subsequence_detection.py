from typing import Dict
import numpy as np

import xarray as xr

from pywatts.core.base import BaseTransformer


class SubsequenceDetection(BaseTransformer):
    """
    This module splits the timeseries into different subsequences.
    Therefore it uses a minimum search that can be parametrized with
    "minimum", "zero", "custom", "none".

    This module implements the first step of the Energy Times Series
    Motif Discovery using Symbolic Aggregated Approximation (eSAX) algorithm.
    Script to extract the "interesting" sequences
    Original code: Nicole Ludwig 2021

    :param name: Name of the Subsequence Detection
    :type name: str
    :param method: The method used for finding minima
    :type method: str
    :param measuring_interval: The measuring interval in seconds
    :type measuring_interval: int
    """

    def __init__(self, name: str = "SubsequenceDetection", method: str = "minimum",
                 measuring_interval: int = 60):
        super().__init__(name)
        self.method = method
        self.measuring_interval = measuring_interval

    def get_params(self) -> Dict[str, object]:
        """
        Returns a list of parameters used for the subsequence detection.

        :return: Parameters set for the subsequence detection
        :rtype: Dict
        """
        return {"method": self.method,
                "measuring_interval": self.measuring_interval}

    def set_params(self, method: str = None, measuring_interval: int = None):
        """
        Set the parameters for the subsequence detection

        :param method: The method used for subsequence detection (e.g. minimum)
        :type method: str
        :param measuring_interval: The measuring interval in seconds
        :type measuring_interval: int
        """
        if method is not None:
            self.method = method
        if measuring_interval is not None:
            self.measuring_interval = measuring_interval

    def transform(self, x=xr.DataArray) -> xr.DataArray:
        """
        Transforms the input.
        This method separates the time series in subsequences and calculates the Euclidean Cumulative Distribution
        Function ECFD.

        ASSUME: All measurements must have a timestamp and there should be no NaN values in it.

        :param data: Input xarray.DataArray
        :type data: xr.DataArray
        :return: The subsequences as a list of np arrays

        :return: Subsequences and ecdf
        :rtype: list of ndarrays and dataframe with the ecdf (x,y)
        """

        # create the subsequences with the day or subday patterns

        # calculate how many measuring intervalls fit in one day (in seconds)
        window = round(24 * ((60 * 60) / self.measuring_interval))

        # get sequences and store the startpoints and sequences separately to not have lists of lists
        sequences, _ = self._minimum_search(data=x, method=self.method, window=window)

        # Plot input (whole time-series) and output (sequences) data
        # NOTE: 'data' variable need unchanged!
        # plots.simple_plot(data, "CompleteTimeSeries.pdf")
        # plots.subsequences_plot(sequences, "All_Sequences.pdf")

        asc = list(range(0, len(sequences)))

        seq_dict = zip(asc, sequences)
        seq_dict = dict(seq_dict)

        print("Done")

        return seq_dict

    def _minimum_search(self, data: xr.DataArray, method: str, window: int, custom_event=0.06, window_size=100):
        """
        Function to find minima in the time series that could indicate process starts.

        :param window: custom window length in case 'event' equals 'none'
        :param data: the time series of interest
        :param method: (none, zero, minimum, custom) subsequences are either determined by a
        minimum search or thorugh the points where they are zero or another
        specified value. If none is selected the subsequences are predefined by
        the window length
        :param custom_event: the customized value for the event start (default = 0.06)
        :param window_size: indicates the window size for the minimum search
        :return:
        (dmin) list of nparrays containing the subsequences
        (localmin) list of startpoints     NOTE: The minima are not included in the subsequences
                                                        (thats why the points in localmin are always n+1)
        """
        dmin = []
        localmin = []

        # the subsequences in dmin always start with the minimum
        if method == "minimum":
            print("Searching for minima ...\n")
            # initialise __vector__ for minima

            # Loop that finds all minima occuring in each run
            w = window_size

            # find the minima in the window (use the first one found if more than one)

            for i in range(1, int(len(data) / w) + 1):
                k = i * w
                j = (i * w) - w
                vectorPart = data[j:k]
                localmin.append(np.where(vectorPart == min(vectorPart))[0][0] + ((i - 1) * w) + 1)

            print("Preparing list ...\n")

            dmin.append(data[0:localmin[0]])

            for i in range(0, len(localmin) - 1):
                if i == 0:
                    dmin.append(data[localmin[i] - 1:(localmin[i + 1])])
                else:
                    dmin.append(data[localmin[i]:(localmin[i + 1])])
            dmin.append(data[localmin[len(localmin) - 1]:len(data)])

        elif method == "zero":
            print("Searching for zeros ...\n")
            zeros = np.where(data == 0)[0]

            for i in range(0, len(zeros)):
                if data[zeros[i] + 1] != 0.0:
                    localmin.append(zeros[i] + 1)
                    # next point where it is zero again
                    if i + 1 < len(zeros):
                        localmin.append(zeros[i + 1])
                    else:
                        localmin.append(len(data) - 1)

            for i in range(0, len(localmin), 2):
                dmin.append(data[localmin[i]:localmin[i + 1]])

            print("Preparing list ...\n")

            for i in range(0, len(localmin) - 1, 2):
                dmin.append(data[localmin[i]:localmin[i + 1]])

        elif method == "custom":
            print("Searching for custom event ...\n")

            start = np.where(data == custom_event)[0]

            for i in range(0, len(start)):
                if data[start[i] + 1] != custom_event:
                    localmin.append(start[i] + 1)
                    # next point where it is custom again
                    if i + 1 < len(start):
                        localmin.append(start[i + 1])
                    else:
                        localmin.append(len(data) - 1)

            for i in range(1, len(localmin)):
                dmin.append(localmin[i] - localmin[i - 1])

            print("Preparing list ...\n")

            for i in range(0, len(localmin) - 1):
                dmin.append(data[localmin[i]:localmin[i + 1]])

        elif method == "none":
            print("Preparing subsequences ...\n")

            # store the subsequences of size window length for motif discovery in dmin

            for i in range(0, round(len(data) / window)):
                if ((i + 1) * window) < len(data):
                    dmin.append(data[(i * window):((i + 1) * window)])
                else:
                    dmin.append(data[(i * window):len(data) - 1])

            # save the startpoints(window length distance)
            for i in range(0, len(dmin)):
                localmin.append(i * window)

            print("Preparing list ...\n")

        return dmin, localmin


