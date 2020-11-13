import os
import unittest
from unittest.mock import mock_open, patch, call, MagicMock

import pandas as pd
import xarray as xr
from networkx.tests.test_convert_pandas import pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from pywatts.core.computation_mode import ComputationMode
from pywatts.core.pipeline import Pipeline
from pywatts.core.step import Step
from pywatts.modules.missing_value_detection import MissingValueDetector
from pywatts.modules.whitelister import WhiteLister
from pywatts.wrapper.sklearn_wrapper import SKLearnWrapper

pipeline_json = {'id': 1,
                 'name': 'Pipeline',
                 'modules': [{'class': 'WhiteLister',
                              'module': 'pywatts.modules.whitelister',
                              'name': 'WhiteLister',
                              'params': {'target': 'name'}},
                             {'class': 'SKLearnWrapper',
                              'is_fitted': False,
                              'module': 'pywatts.wrapper.sklearn_wrapper',
                              'name': 'StandardScaler',
                              'params': {'copy': True, 'with_mean': True, 'with_std': True},
                              'sklearn_module': os.path.join('test_pipeline', 'StandardScaler.pickle')},
                             {'class': 'SKLearnWrapper',
                              'is_fitted': False,
                              'module': 'pywatts.wrapper.sklearn_wrapper',
                              'name': 'LinearRegression',
                              'params': {'copy_X': True,
                                         'fit_intercept': True,
                                         'n_jobs': None,
                                         'normalize': False},
                              'sklearn_module': os.path.join('test_pipeline', 'LinearRegression.pickle')}],
                 'steps': [{'class': 'StartStep',
                            'computation_mode': 4,
                            'id': 1,
                            'input_ids': [],
                            'last': False,
                            'module': 'pywatts.core.start_step',
                            'name': 'StartStep',
                            'target_ids': []},
                           {'class': 'Step',
                            'computation_mode': 4,
                            'condition': None,
                            'id': 2,
                            'input_ids': [1],
                            'last': False,
                            'module': 'pywatts.core.step',
                            'module_id': 0,
                            'name': 'WhiteLister',
                            'plot': False,
                            'target_ids': [],
                            'to_csv': False,
                            'train_if': None},
                           {'class': 'Step',
                            'computation_mode': 4,
                            'condition': None,
                            'id': 3,
                            'input_ids': [2],
                            'last': False,
                            'module': 'pywatts.core.step',
                            'module_id': 1,
                            'name': 'StandardScaler',
                            'plot': False,
                            'target_ids': [],
                            'to_csv': False,
                            'train_if': None},
                           {'class': 'CollectStep',
                            'computation_mode': 4,
                            'id': 4,
                            'input_ids': [2, 3],
                            'last': False,
                            'module': 'pywatts.core.collect_step',
                            'name': 'Collect',
                            'target_ids': []},
                           {'class': 'Step',
                            'computation_mode': 4,
                            'condition': None,
                            'id': 5,
                            'input_ids': [4],
                            'last': True,
                            'module': 'pywatts.core.step',
                            'module_id': 2,
                            'name': 'LinearRegression',
                            'plot': False,
                            'target_ids': [],
                            'to_csv': False,
                            'train_if': None}],
                 'version': 1}


class TestPipeline(unittest.TestCase):

    @patch("pywatts.core.pipeline.FileManager")
    def setUp(self, fm_mock) -> None:
        self.pipeline = Pipeline()

    def tearDown(self) -> None:
        self.pipeline = None

    def test_add_only_module(self):
        SKLearnWrapper(LinearRegression())(self.pipeline)
        # nodes 1 plus startstep
        self.assertEqual(self.pipeline.computational_graph.number_of_nodes(), 2)
        self.assertEqual(self.pipeline.target_graph.number_of_nodes(), 2)

    def test_add_one_module_with_pipeline_in_a_list(self):
        SKLearnWrapper(LinearRegression())([self.pipeline])
        # nodes 1 plus startstep
        self.assertEqual(self.pipeline.computational_graph.number_of_nodes(), 2)
        self.assertEqual(self.pipeline.target_graph.number_of_nodes(), 2)

    def test_add_module_which_is_not_in_a_list(self):
        wrapper = SKLearnWrapper(LinearRegression())([self.pipeline])
        SKLearnWrapper(LinearRegression())(wrapper)
        # nodes 1 plus startstep
        self.assertEqual(self.pipeline.computational_graph.number_of_nodes(), 3)
        self.assertEqual(self.pipeline.target_graph.number_of_nodes(), 3)

    def test_add_module_with_inputs(self):
        whitelister = WhiteLister(target="name")(self.pipeline)
        scaler = SKLearnWrapper(StandardScaler())([whitelister])
        SKLearnWrapper(LinearRegression())([whitelister, scaler])

        # Three modules plus start step and one collect step
        self.assertEqual(5, self.pipeline.computational_graph.number_of_nodes())
        self.assertEqual(5, self.pipeline.target_graph.number_of_nodes())
        self.assertEqual(5, self.pipeline.computational_graph.number_of_edges())
        self.assertEqual(0, self.pipeline.target_graph.number_of_edges())

    def test_add_module_with_one_input_without_a_list(self):
        whitelister = WhiteLister(target="name")(self.pipeline)
        scaler = SKLearnWrapper(StandardScaler())(whitelister)

        # Three modules plus start step and one collect step
        self.assertEqual(3, self.pipeline.computational_graph.number_of_nodes())
        self.assertEqual(3, self.pipeline.target_graph.number_of_nodes())
        self.assertEqual(2, self.pipeline.computational_graph.number_of_edges())
        self.assertEqual(0, self.pipeline.target_graph.number_of_edges())

    @patch('pywatts.core.pipeline.FileManager')
    @patch('pywatts.core.pipeline.json')
    @patch("builtins.open", new_callable=mock_open)
    def test_to_folder(self, mock_file, json_mock, fm_mock):
        whitelister = WhiteLister(target="name")(self.pipeline)
        scaler = SKLearnWrapper(StandardScaler())([whitelister])
        SKLearnWrapper(LinearRegression())([whitelister, scaler])
        fm_mock_object = MagicMock()
        fm_mock.return_value = fm_mock_object
        fm_mock_object.get_path.side_effect = [
            os.path.join('test_pipeline', 'StandardScaler.pickle'),
            os.path.join('test_pipeline', 'LinearRegression.pickle'),
            os.path.join('test_pipeline', 'pipeline.json'),
        ]

        self.pipeline.to_folder("test_pipeline")

        calls_open = [call(os.path.join('test_pipeline', 'StandardScaler.pickle'), 'wb'),
                      call(os.path.join('test_pipeline', 'LinearRegression.pickle'), 'wb'),
                      call(os.path.join('test_pipeline', 'pipeline.json'), 'w')]
        mock_file.assert_has_calls(calls_open, any_order=True)
        args, kwargs = json_mock.dump.call_args
        assert kwargs["obj"]["id"] == pipeline_json["id"]
        assert kwargs["obj"]["name"] == pipeline_json["name"]
        assert kwargs["obj"]["modules"] == pipeline_json["modules"]
        assert kwargs["obj"]["steps"] == pipeline_json["steps"]

    @patch('pywatts.core.pipeline.FileManager')
    @patch('pywatts.wrapper.sklearn_wrapper.pickle')
    @patch('pywatts.core.pipeline.json')
    @patch("builtins.open", new_callable=mock_open)
    @patch('pywatts.core.pipeline.os.path.isdir')
    def test_from_folder(self, isdir_mock, mock_file, json_mock, pickle_mock, fm_mock):
        scaler = StandardScaler()
        linear_regression = LinearRegression()

        isdir_mock.return_value = True
        json_mock.load.return_value = pipeline_json

        pickle_mock.load.side_effect = [scaler, linear_regression]

        self.pipeline.from_folder("test_pipeline")
        print(mock_file.method_calls)
        calls_open = [call(os.path.join("test_pipeline", "StandardScaler.pickle"), "rb"),
                      call(os.path.join("test_pipeline", "LinearRegression.pickle"), "rb"),
                      call(os.path.join("test_pipeline", "pipeline.json"), "r")]

        mock_file.assert_has_calls(calls_open, any_order=True)

        json_mock.load.assert_called_once()
        assert pickle_mock.load.call_count == 2

        isdir_mock.assert_called_once()
        self.assertEqual(self.pipeline.computational_graph.number_of_nodes(), 5)
        self.assertEqual(self.pipeline.target_graph.number_of_nodes(), 5)
        self.assertEqual(self.pipeline.computational_graph.number_of_edges(), 5)
        self.assertEqual(self.pipeline.target_graph.number_of_edges(), 0)

    def test_naming_conflict(self):
        whitelister = WhiteLister(target="test")(self.pipeline)
        whitelister2 = WhiteLister(target="test")(self.pipeline)
        MissingValueDetector()([whitelister, whitelister2])
        self.pipeline.train(pd.DataFrame({"test": [1, 2, 2, 3, 4], "test2": [2, 2, 2, 2, 2]},
                                         index=pd.DatetimeIndex(pd.date_range('2000-01-01', freq='24H', periods=5))))

    def test_add_with_target(self):
        whitelister = WhiteLister(target="name")(self.pipeline)
        SKLearnWrapper(LinearRegression())([whitelister], targets=[whitelister])
        self.assertEqual(3, self.pipeline.computational_graph.number_of_nodes())
        self.assertEqual(3, self.pipeline.target_graph.number_of_nodes())
        self.assertEqual(2, self.pipeline.computational_graph.number_of_edges())
        self.assertEqual(1, self.pipeline.target_graph.number_of_edges())

    def test_multiple_same_module(self):
        whitelister_one = WhiteLister(target="test")(self.pipeline)
        whitelister_two = WhiteLister(target="test2")(self.pipeline)
        detector = MissingValueDetector()
        detector([whitelister_one])
        detector([whitelister_two])

        self.assertEqual(5, self.pipeline.computational_graph.number_of_nodes())
        modules = []
        for element in self.pipeline.id_to_step.values():
            if isinstance(element, Step) and not element.module in modules:
                modules.append(element.module)
        self.assertEqual(3, len(modules))
        self.assertEqual(5, self.pipeline.target_graph.number_of_nodes())
        self.assertEqual(4, self.pipeline.computational_graph.number_of_edges())
        self.assertEqual(0, self.pipeline.target_graph.number_of_edges())

        self.pipeline.train(pd.DataFrame({"test": [1, 2, 2, 3, 4], "test2": [2, 2, 2, 2, 2]},
                                         index=pd.DatetimeIndex(pd.date_range('2000-01-01', freq='24H', periods=5))))

    @patch('pywatts.core.pipeline.FileManager')
    def test_add_pipeline_to_pipeline_and_train(self, fm_mock):
        whitelister = WhiteLister(target="test")(self.pipeline)

        sub_pipeline = Pipeline()

        detector = MissingValueDetector()
        detector(sub_pipeline)

        sub_pipeline(whitelister)

        self.pipeline.train(pd.DataFrame({"test": [24, 24]}, index=pd.to_datetime(
            ['2015-06-03 00:00:00', '2015-06-03 01:00:00'])))

        for step in self.pipeline.id_to_step.values():
            assert step.computation_mode == ComputationMode.FitTransform

    @patch('pywatts.core.pipeline.FileManager')
    def test_add_pipeline_to_pipeline_and_test(self, fm_mock):
        # Add some steps to the pipeline

        # Assert that the computation is set to fit_transform if the ComputationMode was default

        step = MagicMock()
        step.computation_mode = ComputationMode.Default
        step.finished = False
        time = pd.date_range('2000-01-01', freq='24H', periods=7)

        ds = xr.Dataset({'foo': ('time', [2, 3, 4, 5, 6, 7, 8]), 'time': time})

        subpipeline = Pipeline()
        subpipeline.add(module=step)

        subpipeline(self.pipeline)

        self.pipeline.test(ds)

        step.set_computation_mode.assert_called_once_with(ComputationMode.Transform)

        step.reset.assert_called_once()

    @patch("pywatts.core.pipeline.FileManager")
    @patch('pywatts.core.pipeline.json')
    @patch("builtins.open", new_callable=mock_open)
    def test_add_pipeline_to_pipeline_and_save(self, open_mock, json_mock, fm_mock):
        whitelister = WhiteLister(target="test")(self.pipeline)

        sub_pipeline = Pipeline()

        detector = MissingValueDetector()
        detector(sub_pipeline)

        sub_pipeline(whitelister)

        self.pipeline.to_folder(path="path")

        self.assertEqual(json_mock.dump.call_count, 2)

    @patch('pywatts.core.pipeline.FileManager')
    def test__collect_batch_results_naming_conflict(self, fm_mock):
        time = pd.date_range('2000-01-01', freq='24H', periods=7)

        ds_one = xr.Dataset({'foo': ('time', [2, 3, 4, 5, 6, 7, 8]), 'time': time})
        ds_two = xr.Dataset({'foo': ('time', [2, 2, 2, 2, 2, 2, 2]), 'time': time})
        ds_result = xr.Dataset({'foo': ('time', [2, 3, 4, 5, 6, 7, 8]),
                                'foo_0': ('time', [2, 2, 2, 2, 2, 2, 2]),
                                'time': time})
        step_one = MagicMock()
        step_two = MagicMock()
        step_one.get_result.return_value = ds_one
        step_two.get_result.return_value = ds_two

        result = self.pipeline._collect_results([step_one, step_two])

        # Two calls, since get_result is called by further_elements and _transform
        xr.testing.assert_equal(result, ds_result)

    @patch("pywatts.core.pipeline.FileManager")
    def test_get_params(self, fm_mock):
        result = Pipeline(batch=pd.Timedelta("1h")).get_params()
        self.assertEqual(result, {
            "batch": pd.Timedelta("1h")
        })

    def test_set_params(self):
        self.pipeline.set_params(batch=pd.Timedelta("2h"))
        self.assertEqual(self.pipeline.get_params(),
                         {
                             "batch": pd.Timedelta("2h")
                         })

    def test__collect_batch_results(self):
        step_one = MagicMock()
        step_two = MagicMock()
        result_step_one = MagicMock()
        result_step_two = MagicMock()
        merged_result = MagicMock()

        step_one.get_result.return_value = result_step_one
        step_two.get_result.return_value = result_step_two
        result_step_one.merge.return_value = merged_result

        result = self.pipeline._collect_results([step_one, step_two])

        # Assert that steps are correclty called.
        step_one.get_result.assert_called_once_with(None, None)
        step_two.get_result.assert_called_once_with(None, None)
        result_step_one.merge.assert_called_once_with(result_step_two)

        # Assert return value is correct
        self.assertEqual(merged_result, result)

    @patch("pywatts.core.pipeline.FileManager")
    @patch("pywatts.core.pipeline.xr.concat")
    def test_batched_pipeline(self, concat_mock, fm_mock):
        # Add some steps to the pipeline

        # Assert that the computation is set to fit_transform if the ComputationMode was default
        first_step = MagicMock()
        first_step.computation_mode = ComputationMode.Default
        first_step.finished = False

        self.pipeline.set_params(pd.Timedelta("24h"))
        self.pipeline.add(module=first_step)

        self.pipeline.test(pd.DataFrame({"test": [1, 2, 2, 3, 4], "test2": [2, 2, 2, 2, 2]},
                                        index=pd.DatetimeIndex(pd.date_range('2000-01-01', freq='24H', periods=5))))

        first_step.set_computation_mode.assert_called_once_with(ComputationMode.Transform)
        calls = [
            call(pd.Timestamp('2000-01-01 00:00:00', freq='24H'), pd.Timestamp('2000-01-02 00:00:00', freq='24H')),
            call(pd.Timestamp('2000-01-02 00:00:00', freq='24H'), pd.Timestamp('2000-01-03 00:00:00', freq='24H')),
            call(pd.Timestamp('2000-01-03 00:00:00', freq='24H'), pd.Timestamp('2000-01-04 00:00:00', freq='24H')),
            call(pd.Timestamp('2000-01-04 00:00:00', freq='24H'), pd.Timestamp('2000-01-05 00:00:00', freq='24H')),
            call(pd.Timestamp('2000-01-05 00:00:00', freq='24H'), pd.Timestamp('2000-01-06 00:00:00', freq='24H')),
        ]
        first_step.get_result.assert_has_calls(calls, any_order=True)
        self.assertEqual(concat_mock.call_count, 4)


    @patch("pywatts.core.pipeline.FileManager")
    @patch("pywatts.core.pipeline.xr.concat")
    def test_batch_2H_transform(self, concat_mock, fm_mock):
        time = pd.date_range('2000-01-01', freq='1H', periods=7)
        ds = xr.Dataset({'foo': ('time', [2, 3, 4, 3, 3, 1, 2]), 'time': time})
        pipeline = Pipeline(batch=pd.Timedelta("2h"))
        step_one = MagicMock()
        result_mock = MagicMock()
        concat_mock.return_value = result_mock
        pipeline.start_step.last = False
        step_one.further_elements.side_effect = [True, True, True, True, False]
        pipeline.add(module=step_one, input_ids=[1])

        result = pipeline.transform(ds)

        self.assertEqual(concat_mock.call_count, 3)
        self.assertEqual(step_one.get_result.call_count, 4)
        self.assertEqual(step_one.further_elements.call_count, 5)
        self.assertEqual(result_mock, result)

    @patch('pywatts.core.pipeline.FileManager')
    @patch("pywatts.core.pipeline._get_time_indeces", return_value=["time"])
    def test_transform_pipeline(self, get_time_indeces_mock, fm_mock):
        input_mock = MagicMock()
        input_mock.indexes = {"time": ["20.12.2020"]}
        step_two = MagicMock()
        result_mock = MagicMock()
        step_two.get_result.return_value = result_mock
        self.pipeline.add(module=step_two, input_ids=[1])
        self.pipeline.start_step.last = False

        result = self.pipeline.transform(input_mock)

        step_two.get_result.assert_called_once_with("20.12.2020", None)
        get_time_indeces_mock.assert_called_once_with(input_mock)
        self.assertEqual(result_mock, result)


    @patch("pywatts.core.pipeline.FileManager")
    @patch("pywatts.core.pipeline.Pipeline.from_folder")
    def test_load(self, from_folder_mock, fm_mock):
        created_pipeline = MagicMock()
        from_folder_mock.return_value = created_pipeline
        pipeline = Pipeline.load({'name': 'Pipeline',
                                  'class': 'Pipeline',
                                  'module': 'pywatts.core.pipeline',
                                  'pipeline_path': 'save_path'})

        from_folder_mock.assert_called_once_with("save_path")
        self.assertEqual(created_pipeline, pipeline)


    @patch("pywatts.core.pipeline.FileManager")
    @patch("pywatts.core.pipeline.Pipeline.to_folder")
    @patch("pywatts.core.pipeline.os")
    def test_save(self, os_mock, to_folder_mock, fm_mock):
        os_mock.path.join.return_value = "save_path"
        sub_pipeline = Pipeline(batch=pd.Timedelta("1h"))
        detector = MissingValueDetector()
        detector(sub_pipeline)
        fm_mock = MagicMock()
        fm_mock.basic_path = "path_to_save"
        result = sub_pipeline.save(fm_mock)

        to_folder_mock.assert_called_once_with("save_path")
        os_mock.path.join.assert_called_once_with("path_to_save", "Pipeline")
        self.assertEqual({'name': 'Pipeline',
                          'class': 'Pipeline',
                          'module': 'pywatts.core.pipeline',
                          'params': {'batch': '0 days 01:00:00'},
                          'pipeline_path': 'save_path'}, result)


    @patch("pywatts.core.pipeline.FileManager")
    @patch("pywatts.core.pipeline.xr.concat")
    def test_batch_1_transform(self, concat_mock, fm_mock):
        time = pd.date_range('2000-01-01', freq='1H', periods=7)
        ds = xr.Dataset({'foo': ('time', [2, 3, 4, 3, 3, 1, 2]), 'time': time})
        pipeline = Pipeline(batch=pd.Timedelta("1h"))
        step_one = MagicMock()
        result_mock = MagicMock()
        concat_mock.return_value = result_mock
        pipeline.start_step.last = False
        step_one.further_elements.side_effect = [True, True, True, True, True, True, True, False]
        pipeline.add(module=step_one, input_ids=[1])

        result = pipeline.transform(ds)

        self.assertEqual(concat_mock.call_count, 6)
        self.assertEqual(step_one.get_result.call_count, 7)
        self.assertEqual(step_one.further_elements.call_count, 8)
        self.assertEqual(result_mock, result)

    @patch('pywatts.core.pipeline.FileManager')
    def test_test(self, fm_mock):
        # Add some steps to the pipeline

        # Assert that the computation is set to fit_transform if the ComputationMode was default
        first_step = MagicMock()
        first_step.computation_mode = ComputationMode.Default
        first_step.finished = False

        second_step = MagicMock()
        second_step.computation_mode = ComputationMode.Train
        second_step.finished = False

        self.pipeline.add(module=first_step)
        self.pipeline.add(module=second_step)

        self.pipeline.test(pd.DataFrame({"test": [1, 2, 2, 3, 4], "test2": [2, 2, 2, 2, 2]},
                                        index=pd.DatetimeIndex(pd.date_range('2000-01-01', freq='24H', periods=5))))

        first_step.get_result.assert_called_once_with(pd.Timestamp('2000-01-01 00:00:00', freq='24H'), None)
        second_step.get_result.assert_called_once_with(pd.Timestamp('2000-01-01 00:00:00', freq='24H'), None)

        first_step.set_computation_mode.assert_called_once_with(ComputationMode.Transform)
        second_step.set_computation_mode.assert_called_once_with(ComputationMode.Transform)

        first_step.reset.assert_called_once()
        second_step.reset.assert_called_once()

    @patch('pywatts.core.pipeline.FileManager')
    def test_train(self, fmmock):
        # Add some steps to the pipeline

        # Assert that the computation is set to fit_transform if the ComputationMode was default
        first_step = MagicMock()
        first_step.computation_mode = ComputationMode.Default
        first_step.finished = False

        second_step = MagicMock()
        second_step.computation_mode = ComputationMode.Train
        second_step.finished = False

        self.pipeline.add(module=first_step)
        self.pipeline.add(module=second_step)

        self.pipeline.train(pd.DataFrame({"test": [1, 2, 2, 3, 4], "test2": [2, 2, 2, 2, 2]},
                                         index=pd.DatetimeIndex(pd.date_range('2000-01-01', freq='24H', periods=5))))

        first_step.set_computation_mode.assert_called_once_with(ComputationMode.FitTransform)
        second_step.set_computation_mode.assert_called_once_with(ComputationMode.FitTransform)
        first_step.get_result.assert_called_once_with(pd.Timestamp('2000-01-01 00:00:00', freq='24H'), None)
        second_step.get_result.assert_called_once_with(pd.Timestamp('2000-01-01 00:00:00', freq='24H'), None)

        first_step.reset.assert_called_once()
        second_step.reset.assert_called_once()