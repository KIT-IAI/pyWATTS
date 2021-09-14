from typing import Dict

from pywatts.core.computation_mode import ComputationMode


class RunSetting:
    """
    The RunSetting contains the setting which is specific to one run.

    :param computation_mode: The computation mode for the specific run.
    :type computation_mode: ComputationMode
    """

    def __init__(self, computation_mode: ComputationMode):
        self.computation_mode = computation_mode

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
        return setting

    def clone(self) -> 'RunSetting':
        """
        Clones the current RunSetting.
        :return: The cloned RunSetting.
        :rtype: RunSetting
        """
        return RunSetting(
            computation_mode=self.computation_mode
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
    def load(load_information: Dict):
        """
        Create a RunSetting from a Dict.
        :return: The loaded RunSetting
        :rtype: RunSetting
        """
        return RunSetting(**load_information)
