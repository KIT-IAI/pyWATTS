import numpy as np

from pywatts.modules.generation.anomaly_generation_module import AnomalyGeneration


class EnergyAnomalyGeneration(AnomalyGeneration):
    """
    Module to define specific anomalies to be inserted into an energy time series.
    """

    def _anomaly_type1(self, target, indices, lengths):
        """
        Anomaly type 1 that sets the energy time series to a constant value of 0.
        """
        return self._anomaly_constant(target, indices, lengths, constant=0)

    def _anomaly_type2(self, target, indices, lengths, softstart=False):
        """
        Anomaly type 2 that decreases the gradient of the energy time series and potentially keeps it constant for
        several time steps.
        """
        for idx, length in zip(indices, lengths):
            if softstart:
                r = np.random.rand()  # random value
                target[idx] = r * target[idx] + (1 - r) * target[idx - 1]
                if length > 1:
                    target[idx:idx + length] = target[idx]
            else:
                target[idx:idx + length] = target[idx - 1]
        return target

    def _anomaly_type3(self, target, indices, lengths,
                       is_extreme=False, range_r=(0, 10)):
        """
        Anomaly type 3 that shifts the energy time series down.
        """
        for idx, length in zip(indices, lengths):
            if length > 1:
                raise Exception("Type 3 energy anomalies cannot be longer than 1.")
            else:
                if is_extreme:
                    shift = target[idx]
                else:
                    r = np.random.uniform(*range_r)
                    shift = r * np.abs(target[idx + 1] - target[idx])
                target[idx:] -= shift
        return target

    def _anomaly_type4(self, target, indices, lengths, range_r=(0, 10)):
        """
        Anomaly type 4 that shifts the energy time series up.
        """
        for idx, length in zip(indices, lengths):
            if length > 1:
                raise Exception("Type 4 energy anomalies cannot be longer than 1.")
            else:
                r = np.random.uniform(*range_r)
                shift = r * np.abs(target[idx + 1] - target[idx])
                target[idx:] += shift
        return target
