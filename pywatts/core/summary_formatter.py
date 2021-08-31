import json
from abc import ABC, abstractmethod
from typing import List

from pywatts.core.filemanager import FileManager
from pywatts.core.summary_object import SummaryObject, SummaryCategory, SummaryObjectList, SummaryObjectTable
from tabulate import tabulate


class SummaryFormatter(ABC):
    """
    SummaryFormatter is responsible for formatting the SummaryObjects according to the output file.
    """

    def create_summary(self, summaries: List[SummaryObject], fm: FileManager):
        """
        This method is responsible for creating and storing the summaries according to the output file.
        :param summaries: The summaries that should be stored.
        :type summaries: List[SummaryObject]
        :param fm: The pyWATTS filemanager.
        :type fm: FileManager
        """

    @abstractmethod
    def _create_summary(self, summary: SummaryObject):
        pass

    @abstractmethod
    def _create_table_summary(self, summary: SummaryObject):
        pass


class SummaryMarkdown(SummaryFormatter):
    """
    SummaryMarkdown is a SummaryFormatter that save the SummaryObjects according as markdown file.
    """

    def create_summary(self, summaries: List[SummaryObject], fm: FileManager):
        """
        This method is responsible for creating and storing the summaries as markdown file.
        :param summaries: The summaries that should be stored.
        :type summaries: List[SummaryObject]
        :param fm: The pyWATTS filemanager.
        :type fm: FileManager
        """
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


class SummaryJSON(SummaryFormatter):
    """
    SummaryJSON is a SummaryFormatter that save the SummaryObjects according as json file.
    """

    def create_summary(self, summaries: List[SummaryObject], fm: FileManager):
        """
        This method is responsible for creating and storing the summaries as json file.
        :param summaries: The summaries that should be stored.
        :type summaries: List[SummaryObject]
        :param fm: The pyWATTS filemanager.
        :type fm: FileManager
        """
        summary_dict = {}
        for category in [SummaryCategory.Summary, SummaryCategory.FitTime, SummaryCategory.TransformTime]:
            category_dict = {}
            for summary in filter(lambda s: s.category == category, summaries):
                if summary.additional_information != "" or len(summary.k_v) > 0:
                    if isinstance(summary, SummaryObjectList):
                        category_dict.update(self._create_summary(summary))
                    elif isinstance(summary, SummaryObjectTable):
                        category_dict.update(self._create_table_summary(summary))

            summary_dict.update({category.name: category_dict})
        with open(fm.get_path("summary.json"), "w") as file:
            json.dump(summary_dict, file)
        return summary_dict

    def _create_summary(self, summary: SummaryObject):
        result_dict = {
            key: value for key, value in summary.k_v.items()
        }
        return {
            summary.name: {
                "additional_information": summary.additional_information,
                "results": result_dict
            }
        }

    def _create_table_summary(self, summary: SummaryObject):
        result_dict = {
            key: value.tolist() for key, value in summary.k_v.items()
        }
        return {
            summary.name: {
                "additional_information": summary.additional_information,
                "results": result_dict
            }
        }
