# pylint: disable=W0223
# Pylint cannot handle abstract subclasses of abstract base classes

from abc import ABC

from pywatts.core.base import BaseEstimator


class BaseWrapper(BaseEstimator, ABC):
    """
    The base wrappers class, where all wrappers have to inherit from.

    :param name: Name of the module
    :type name: str
    """
    def __init__(self, name: str):
        super().__init__(name)
        self.is_wrapper = True
