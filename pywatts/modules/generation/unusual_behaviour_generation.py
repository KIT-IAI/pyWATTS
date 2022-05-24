import numpy as np

from pywatts.modules.generation.anomaly_generation_module import AnomalyGeneration


class UnusualBehaviour(AnomalyGeneration):
    """
    Module to define specific anomalies to be inserted into a power time series.
    """

    def _anomaly_type1(self, target, indices, lengths, min_factor=0.3, max_factor=0.8):
        """
        Unusual behaviour type 1. The  power is smaller than normal.
        """
        for idx, length in zip(indices, lengths):
                change = np.random.uniform(min_factor, max_factor, 1) * min(target[idx : idx + length].values)
                target[idx : idx + length] -= change
        return target

    def _anomaly_type2(self, target, indices, lengths , min_factor=0.5, max_factor=1):
        """
        Unusual behaviour type 2. The  power is higher than normal.
        """
        for idx, length in zip(indices, lengths):
                factor = np.random.uniform(min_factor, max_factor, 1) * min(target[idx : idx + length].values)
                target[idx : idx + length] += factor
        return target


    def _anomaly_type3(self, target, indices, lengths, min_factor=0.3, max_factor=0.8, transition=0.1):
        """
        Unusual behaviour type 3. The  power is smaller than normal with transition
        """
        for idx, length in zip(indices, lengths):
            factor = np.random.uniform(min_factor, max_factor, 1) * min(target[idx : idx + length].values)
            softstart_zone = int(length * transition)
            target[idx: idx + softstart_zone] -= np.linspace(0, factor, softstart_zone).reshape((-1,))
            target[idx + softstart_zone: idx + length - softstart_zone] -= factor
            target[idx + length - softstart_zone: idx + length] -= np.linspace(factor, 0, softstart_zone).reshape((-1,))
        return target

    def _anomaly_type4(self, target, indices, lengths , min_factor=0.5, max_factor=1, transition=0.1):
        """
        Unusual behaviour type 4. The  power is higher than normal with transition
        """
        for idx, length in zip(indices, lengths):
            factor = np.random.uniform(min_factor, max_factor, 1) * min(target[idx : idx + length].values)
            softstart_zone = int(length * transition)
            target[idx: idx + softstart_zone] += np.linspace(0, factor, softstart_zone).reshape((-1,))
            target[idx + softstart_zone: idx + length - softstart_zone] += factor
            target[idx + length - softstart_zone: idx + length] += np.linspace(factor, 0, softstart_zone).reshape((-1,))
        return target
