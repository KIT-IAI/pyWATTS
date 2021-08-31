from abc import ABC
from enum import IntEnum


class SummaryCategory(IntEnum):
    """
    Category for Summaries.
    """
    Summary = 1
    TransformTime = 2
    FitTime = 3


class SummaryObject(ABC):
    """
    A SummaryObject contains the results of a Summary Module.

    :param name: Name of the summary object. Is also the headline in the resulting summary file.
    :type name: str
    :param category: The category of the SummaryObject. Determines the section in the final summary file of the
                     result of this summary object.
    :type category: SummaryCategory
    :param additional_information: A string containing additional information that should be stored in the summary.
    :type additional_information: str
    """

    def __init__(self, name: str, category: SummaryCategory = SummaryCategory.Summary,
                 additional_information: str = ""):
        self.k_v = {}
        self.name = name
        self.category = category
        self.additional_information = additional_information

    def set_kv(self, key, value):
        """
        This method sets a value in this summary information.
        :param key: the key of the value.
        :param value: the value
        """
        self.k_v[key] = value


class SummaryObjectList(SummaryObject):
    """
    A SummaryObjectList contains the results of a Summary Module. In the resulting file the summary of this object will
    be saved as list.

    :param name: Name of the summary object. Is also the headline in the resulting summary file.
    :type name: str
    :param category: The category of the SummaryObject. Determines the section in the final summary file of the
                     result of this summary object.
    :type category: SummaryCategory
    :param additional_information: A string containing additional information that should be stored in the summary.
    :type additional_information: str
    """


class SummaryObjectTable(SummaryObject):
    """
    A SummaryObjectList contains the results of a Summary Module. In the resulting file the summary of this object will
    be saved as table.

    :param name: Name of the summary object. Is also the headline in the resulting summary file.
    :type name: str
    :param category: The category of the SummaryObject. Determines the section in the final summary file of the
                     result of this summary object.
    :type category: SummaryCategory
    :param additional_information: A string containing additional information that should be stored in the summary.
    :type additional_information: str
    """

