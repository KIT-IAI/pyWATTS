from abc import ABC, abstractmethod
from enum import IntEnum
from typing import List

from pywatts.core.filemanager import FileManager
from tabulate import tabulate


class SummaryCategory(IntEnum):
    Summary = 1
    TransformTime = 2
    FitTime = 3


class SummaryObject(ABC):
    def __init__(self, name, category: SummaryCategory = SummaryCategory.Summary,
                 additional_information=""):
        self.k_v = {}
        self.name = name
        self.category = category
        self.additional_information = additional_information

    def set_kv(self, key, value):
        self.k_v[key] = value


class SummaryObjectList(SummaryObject):
    pass


class SummaryObjectTable(SummaryObject):
    pass


class SummaryFormatter(ABC):

    def create_summary(self, summaries: List[SummaryObject], fm: FileManager):
        pass

    @abstractmethod
    def _create_summary(self, summary: SummaryObject):
        pass

    @abstractmethod
    def _create_table_summary(self, summary: SummaryObject):
        pass


class SummaryMarkdown(SummaryFormatter):

    def create_summary(self, summaries: List[SummaryObject], fm: FileManager):
        summary_string = "# Summary: \n"
        for category in [SummaryCategory.Summary, SummaryCategory.FitTime, SummaryCategory.TransformTime]:
            summary_string += f"## {category.name}\n"
            for summary in filter(lambda s: s.category == category, summaries):
                if summary.additional_information != "" or len(summary.k_v) > 0:
                    if isinstance(summary, SummaryObjectList):
                        summary_string += self._create_summary(summary)
                    elif isinstance(summary, SummaryObjectTable):
                        summary_string += self._create_table_summary(summary)

        with open(fm.get_path("summary.md"), "w") as file:
            file.write(summary_string)
        return summary_string

    def _create_summary(self, summary: SummaryObject):
        return f"### {summary.name}\n" + f"{summary.additional_information}\n" + "".join(
            [f"* {key} : {value}\n" for key, value in summary.k_v.items()])

    def _create_table_summary(self, summary: SummaryObject):
        return f"### {summary.name}\n" + f"{summary.additional_information}\n" + "".join(
            [
                f"#### {key}\n {tabulate(value, headers=range(len(value)), showindex=range(len(value)), tablefmt='github')}\n"
                for key, value in summary.k_v.items()])
