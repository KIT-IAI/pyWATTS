from abc import ABC
from enum import IntEnum


class SummaryCategory(IntEnum):
    """
    Category for Summaries. Structures the final resulting summary file.
    """
    Summary = 1
    TransformTime = 2
    FitTime = 3


class SummaryObject(ABC):
    """
    TODO
    """
    def __init__(self, name, category: SummaryCategory = SummaryCategory.Summary,
                 additional_information=""):
        self.k_v = {}
        self.name = name
        self.category = category
        self.additional_information = additional_information

    def set_kv(self, key, value):
        """
        TODO
        """
        self.k_v[key] = value


class SummaryObjectList(SummaryObject):
    """
    TODO
    """
    pass


class SummaryObjectTable(SummaryObject):
    """
    TODO
    """
    pass


