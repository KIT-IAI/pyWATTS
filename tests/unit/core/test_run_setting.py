import unittest

from pywatts.core.computation_mode import ComputationMode
from pywatts.core.run_setting import RunSetting


class TestRunSetting(unittest.TestCase):

    def setUp(self) -> None:
        self.run_setting = RunSetting(computation_mode=ComputationMode.Default)

    def tearDown(self) -> None:
        self.run_setting = None

    def test_update(self):
        run_setting = self.run_setting.update(RunSetting(computation_mode=ComputationMode.Train))
        self.assertEqual(run_setting.computation_mode, ComputationMode.Train)

    def test_update_computation_mode_not_updatable(self):
        run_setting = RunSetting(computation_mode=ComputationMode.Train)
        run_setting.update(RunSetting(computation_mode=ComputationMode.Transform))
        self.assertEqual(run_setting.computation_mode, ComputationMode.Train)

        run_setting = RunSetting(computation_mode=ComputationMode.Transform)
        run_setting.update(RunSetting(computation_mode=ComputationMode.Train))
        self.assertEqual(run_setting.computation_mode, ComputationMode.Transform)

    def test_clone(self):
        run_setting = self.run_setting.clone()

        self.assertEqual(run_setting.computation_mode, self.run_setting.computation_mode)

    def test_save(self):
        json = self.run_setting.save()

        self.assertEqual(json, {
            "computation_mode": 4
        })

    def test_load(self):
        run_setting = RunSetting.load({
            "computation_mode": 4
        })
        self.assertEqual(run_setting.computation_mode, ComputationMode.Default)
