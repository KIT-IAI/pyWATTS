from pywatts.core.computation_mode import ComputationMode


class RunSetting:

    def __init__(self, computation_mode: ComputationMode):
        self.computation_mode = computation_mode

    def update(self, run_setting):
        setting = self.clone()
        if setting.computation_mode == ComputationMode.Default:
            setting.computation_mode = run_setting.computation_mode
        return setting

    def clone(self):
        return RunSetting(
            computation_mode=self.computation_mode
        )

    def save(self):
        return {
            "computation_mode": int(self.computation_mode)
        }

    @staticmethod
    def load(load_information):
        return RunSetting(**load_information)