from enum import Enum


class KindOfTransform(Enum):
    """
    Enum for different types of transform
    """
    INVERSE_TRANSFORM = "inverse_transform"
    PROBABILISTIC_TRANSFORM = "prob_transform"
    PREDICT_TRANSFORM = "predict_transform"


class KindOfTransformDoesNotExistException(Exception):
    """
    Exception which indicates that the requested transform method does not exist
    Attributes:
        message -- explanation of the exception
        method -- method which does not exist
    """

    def __init__(self, message, method: KindOfTransform):
        self.message = message
        self.method = method
