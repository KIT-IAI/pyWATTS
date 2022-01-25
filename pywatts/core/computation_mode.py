from enum import IntEnum


class ComputationMode(IntEnum):
    """
    Enum which contains the different computation modes of step.
    """
    Transform = 1
    Train = 2
    FitTransform = 3
    Default = 4
    Refit = 5
