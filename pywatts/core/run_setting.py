from typing import Dict

from pywatts.core.computation_mode import ComputationMode
from pywatts.core.summary_formatter import SummaryFormatter, SummaryMarkdown


class RunSetting:
    """
    The RunSetting contains the setting which is specific to one run.

    :param computation_mode: The computation mode for the specific run.
    :type computation_mode: ComputationMode
    :param summary_formatter: The formatter which formats the summary data.
    :type summary_formatter: SummaryFormatter
    """

    def __init__(self, computation_mode: ComputationMode, summary_formatter: SummaryFormatter = SummaryMarkdown(),
                 online_start=None, return_summary=False):
        self.computation_mode = computation_mode
        self.summary_formatter = summary_formatter
        self.online_start = online_start
        self.return_summary = return_summary

    def update(self, run_setting: 'RunSetting') -> 'RunSetting':
        """
        Updates and returns a new run_setting. **Note** The existing run_settings stay unchanged.

        :param run_setting: The run_setting which should be combined with this RunSetting.
        :type run_setting: RunSetting
        :return: The new RunSetting
        :rtype: RunSetting
        """
        setting = self.clone()
        if setting.computation_mode == ComputationMode.Default:
            setting.computation_mode = run_setting.computation_mode
        setting.summary_formatter = run_setting.summary_formatter
        setting.online_start = run_setting.online_start
        setting.return_summary = run_setting.return_summary
        return setting

    def clone(self) -> 'RunSetting':
        """
        Clones the current RunSetting.
        :return: The cloned RunSetting.
        :rtype: RunSetting
        """
        return RunSetting(
            computation_mode=self.computation_mode,
            summary_formatter=self.summary_formatter,
            online_start=self.online_start,
            return_summary=self.return_summary
        )

    def save(self) -> Dict:
        """
        Saves the RunSetting as JSON.
        :return: A dict which contains all information needed for restoring the RunSetting
        :rtype: Dict
        """
        return {
            "computation_mode": int(self.computation_mode)
        }

    @staticmethod
    def load(load_information):
        return RunSetting(computation_mode=load_information["computation_mode"],
                          summary_formatter=SummaryMarkdown())

    def load(load_information: Dict):
        """
        Create a RunSetting from a Dict.
        :return: The loaded RunSetting
        :rtype: RunSetting
        """
        return RunSetting(**load_information)
