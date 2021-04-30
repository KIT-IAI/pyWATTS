import os
import unittest
from unittest.mock import mock_open, patch, call, MagicMock

import pandas as pd
import xarray as xr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from pywatts.core.computation_mode import ComputationMode
from pywatts.core.pipeline import Pipeline
from pywatts.core.start_step import StartStep
from pywatts.core.step import Step
from pywatts.modules.missing_value_detection import MissingValueDetector
from pywatts.modules.root_mean_squared_error import RmseCalculator
from pywatts.wrapper.sklearn_wrapper import SKLearnWrapper

pipeline_json = {'id': 1,
                 'name': 'Pipeline',
                 'modules': [{'class': 'SKLearnWrapper',
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
                                         'positive': False,
                                         'normalize': False},
                              'sklearn_module': os.path.join('test_pipeline', 'LinearRegression.pickle')}],
                 'steps': [{'class': 'StartStep',
                            'computation_mode': 4,
                            'id': 1,
                            'index': 'input',
                            'input_ids': {},
                            'last': False,
                            'module': 'pywatts.core.start_step',
                            'name': 'input',
                            'target_ids': {}},
                           {'batch_size': None,
                            'callbacks': [],
                            'class': 'Step',
                            'computation_mode': 4,
                            'condition': None,
                            'id': 2,
                            'input_ids': {1: 'input'},
                            'last': False,
                            'module': 'pywatts.core.step',
                            'module_id': 0,
                            'name': 'StandardScaler',
                            'target_ids': {},
                            'train_if': None},
                           {'batch_size': None,
                            'callbacks': [],
                            'class': 'Step',
                            'computation_mode': 4,
                            'condition': None,
                            'id': 3,
                            'input_ids': {2: 'x'},
                            'last': True,
                            'module': 'pywatts.core.step',
                            'module_id': 1,
                            'name': 'LinearRegression',
                            'target_ids': {},
                            'train_if': None}],
                 'version': 1}


class TestPipeline(unittest.TestCase):

    @patch("pywatts.core.pipeline.FileManager")
    def setUp(self, fm_mock) -> None:
        self.pipeline = Pipeline()

    def tearDown(self) -> None:
        self.pipeline = None

    def test_add_input_as_positional(self):
        # Should fail with an better error message
        SKLearnWrapper(LinearRegression())(x=self.pipeline["input"])

    def test_add_only_module(self):
        SKLearnWrapper(LinearRegression())(x=self.pipeline["input"])
        # nodes 1 plus startstep
        self.assertEqual(len(self.pipeline.id_to_step), 2)

    def test_add_module_which_is_not_in_a_list(self):
        wrapper = SKLearnWrapper(LinearRegression())(input=self.pipeline["input"])
        SKLearnWrapper(LinearRegression())(x=wrapper)
        # nodes 1 plus startstep
        self.assertEqual(len(self.pipeline.id_to_step), 3)

    def test_add_pipeline_without_index(self):
        # This should raise an exception since pipeline might get multiple columns in the input dataframe
        with self.assertRaises(Exception) as context:
            SKLearnWrapper(StandardScaler())(x=self.pipeline)  # This should fail
        self.assertEqual(
            "Adding a pipeline as input might be ambigious. Specifiy the desired column of your dataset by using pipeline[<column_name>]",
            str(context.exception))

    def test_add_module_with_inputs(self):
        scaler1 = SKLearnWrapper(StandardScaler())(x=self.pipeline["x"])
        scaler2 = SKLearnWrapper(StandardScaler())(x=self.pipeline["test1"])
        SKLearnWrapper(LinearRegression())(input_1=scaler1, input_2=scaler2)

        # Three modules plus start step and one collect step
        self.assertEqual(5, len(self.pipeline.id_to_step))

    def test_add_module_with_one_input_without_a_list(self):
        scaler = SKLearnWrapper(StandardScaler())(input=self.pipeline["test"])
        SKLearnWrapper(LinearRegression())(input=scaler)

        # Three modules plus start step and one collect step
        self.assertEqual(3, len(self.pipeline.id_to_step))

    @patch('pywatts.core.pipeline.FileManager')
    @patch('pywatts.core.pipeline.json')
    @patch("builtins.open", new_callable=mock_open)
    def test_to_folder(self, mock_file, json_mock, fm_mock):
        scaler = SKLearnWrapper(StandardScaler())(input=self.pipeline["input"])
        SKLearnWrapper(LinearRegression())(x=scaler)
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

        pipeline = Pipeline.from_folder("test_pipeline")
        calls_open = [call(os.path.join("test_pipeline", "StandardScaler.pickle"), "rb"),
                      call(os.path.join("test_pipeline", "LinearRegression.pickle"), "rb"),
                      call(os.path.join("test_pipeline", "pipeline.json"), "r")]

        mock_file.assert_has_calls(calls_open, any_order=True)

        json_mock.load.assert_called_once()
        assert pickle_mock.load.call_count == 2

        isdir_mock.assert_called_once()
        self.assertEqual(3, len(pipeline.id_to_step))

    def test_module_naming_conflict(self):
        # This test should check, that modules with the same name do not lead to an error
        # What should this test?        
        # self.fail()
        pass

    def test_add_with_target(self):
        SKLearnWrapper(LinearRegression())(input=self.pipeline["input"], target=self.pipeline["target"])
        self.assertEqual(3, len(self.pipeline.id_to_step))

    def test_multiple_same_module(self):
        reg_module = SKLearnWrapper(module=LinearRegression())
        reg_one = reg_module(x=self.pipeline["test"], target=self.pipeline["target"])
        reg_two = reg_module(x=self.pipeline["test2"], target=self.pipeline["target"])
        detector = MissingValueDetector()
        detector(dataset=reg_one)
        detector(dataset=reg_two)

        # Three start steps (test, test2, target), two regressors two detectors
        self.assertEqual(7, len(self.pipeline.id_to_step))
        modules = []
        for element in self.pipeline.id_to_step.values():
            if isinstance(element, Step) and not element.module in modules:
                modules.append(element.module)
        # One sklearn wrappers, one missing value detector
        self.assertEqual(2, len(modules))

        self.pipeline.train(
            pd.DataFrame({"test": [1, 2, 2, 3, 4], "test2": [2, 2, 2, 2, 2], "target": [2, 2, 4, 4, -5]},
                         index=pd.DatetimeIndex(pd.date_range('2000-01-01', freq='24H', periods=5))))

    @patch('pywatts.core.pipeline.FileManager')
    def test_add_pipeline_to_pipeline_and_train(self, fm_mock):
        sub_pipeline = Pipeline()

        detector = MissingValueDetector()

        detector(dataset=sub_pipeline["regression"])

        regressor = SKLearnWrapper(LinearRegression(), name="regression")(x=self.pipeline["test"],
                                                                          target=self.pipeline["target"])
        sub_pipeline(regression=regressor)

        self.pipeline.train(pd.DataFrame({"test": [24, 24], "target": [12, 24]}, index=pd.to_datetime(
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

        # BUG: In step_factory.py -> create_step the file_manager of the pipeline is accessed
        # and the pipeline is None... 
        # subpipeline(self.pipeline)

        # self.pipeline.test(ds)

        # step.set_computation_mode.assert_called_once_with(ComputationMode.Transform)

        # step.reset.assert_called_once()

    @patch("pywatts.core.pipeline.FileManager")
    @patch('pywatts.core.pipeline.json')
    @patch("builtins.open", new_callable=mock_open)
    def test_add_pipeline_to_pipeline_and_save(self, open_mock, json_mock, fm_mock):
        sub_pipeline = Pipeline()

        detector = MissingValueDetector()
        detector(dataset=sub_pipeline["regressor"])

        regressor = SKLearnWrapper(LinearRegression())(x=self.pipeline["test"])
        sub_pipeline(regression=regressor)

        self.pipeline.to_folder(path="path")

        self.assertEqual(json_mock.dump.call_count, 2)

    @patch('pywatts.core.pipeline.FileManager')
    def test__collect_batch_results_naming_conflict(self, fm_mock):
        step_one = MagicMock()
        step_one.name = "step"
        step_two = MagicMock()
        step_two.name = "step"
        result_step_one = MagicMock()
        result_step_two = MagicMock()
        merged_result = {
            "step": result_step_one,
            "step_1": result_step_two
        }

        step_one.get_result.return_value = {"step": result_step_one}
        step_two.get_result.return_value = {"step_1": result_step_two}

        result = self.pipeline._collect_results([step_one, step_two])

        # Assert that steps are correclty called.
        step_one.get_result.assert_called_once_with(None, None, return_all=True)
        step_two.get_result.assert_called_once_with(None, None, return_all=True)

        # Assert return value is correct
        self.assertEqual(merged_result, result)

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
        step_one.name = "step_one"
        step_two = MagicMock()
        step_two.name = "step_two"
        result_step_one = MagicMock()
        result_step_two = MagicMock()
        merged_result = {
            "step_one": result_step_one,
            "step_two": result_step_two
        }

        step_one.get_result.return_value = {"step_one": result_step_one}
        step_two.get_result.return_value = {"step_two": result_step_two}

        result = self.pipeline._collect_results([step_one, step_two])

        # Assert that steps are correclty called.
        step_one.get_result.assert_called_once_with(None, None, return_all=True)
        step_two.get_result.assert_called_once_with(None, None, return_all=True)

        # Assert return value is correct
        self.assertEqual(merged_result, result)

    @patch("pywatts.core.pipeline.FileManager")
    @patch("pywatts.core.pipeline.xr.concat")
    def test_batched_pipeline(self, concat_mock, fm_mock):
        # Add some steps to the pipeline

        time = pd.date_range('2000-01-01', freq='1H', periods=7)
        da = xr.DataArray([2, 3, 4, 3, 3, 1, 2], dims=["time"], coords={'time': time})

        # Assert that the computation is set to fit_transform if the ComputationMode was default
        first_step = MagicMock()
        first_step.computation_mode = ComputationMode.Default
        first_step.finished = False
        first_step.further_elements.side_effect = [True, True, True, True, False]

        first_step.get_result.return_value = {"one": da}
        self.pipeline.set_params(pd.Timedelta("24h"))
        self.pipeline.add(module=first_step)

        self.pipeline.test(pd.DataFrame({"test": [1, 2, 2, 3], "test2": [2, 2, 2, 2]},
                                        index=pd.DatetimeIndex(pd.date_range('2000-01-01', freq='24H', periods=4))))

        first_step.set_computation_mode.assert_called_once_with(ComputationMode.Transform)
        calls = [
            call(pd.Timestamp('2000-01-01 00:00:00', freq='24H'), pd.Timestamp('2000-01-02 00:00:00', freq='24H'),
                 return_all=True),
            call(pd.Timestamp('2000-01-02 00:00:00', freq='24H'), pd.Timestamp('2000-01-03 00:00:00', freq='24H'),
                 return_all=True),
            call(pd.Timestamp('2000-01-03 00:00:00', freq='24H'), pd.Timestamp('2000-01-04 00:00:00', freq='24H'),
                 return_all=True),
            call(pd.Timestamp('2000-01-04 00:00:00', freq='24H'), pd.Timestamp('2000-01-05 00:00:00', freq='24H'),
                 return_all=True),
        ]
        first_step.get_result.assert_has_calls(calls, any_order=True)
        self.assertEqual(concat_mock.call_count, 3)

    @patch("pywatts.core.pipeline.FileManager")
    @patch("pywatts.core.pipeline.xr.concat")
    def test_batch_2H_transform(self, concat_mock, fm_mock):
        time = pd.date_range('2000-01-01', freq='1H', periods=7)
        da = xr.DataArray([2, 3, 4, 3, 3, 1, 2], dims=["time"], coords={'time': time})
        pipeline = Pipeline(batch=pd.Timedelta("2h"))
        step_one = MagicMock()
        step_one.get_result.return_value = {"step": da}
        step_one.name = "step"
        result_mock = MagicMock()
        concat_mock.return_value = result_mock
        pipeline.start_steps["foo"] = StartStep("foo"), None
        pipeline.start_steps["foo"][0].last = False
        step_one.further_elements.side_effect = [True, True, True, True, False]
        pipeline.add(module=step_one, input_ids=[1])

        result = pipeline.transform(foo=da)

        self.assertEqual(concat_mock.call_count, 3)
        self.assertEqual(step_one.get_result.call_count, 4)
        self.assertEqual(step_one.further_elements.call_count, 5)
        self.assertEqual({"step": result_mock}, result)

    @patch('pywatts.core.pipeline.FileManager')
    @patch("pywatts.core.pipeline._get_time_indeces", return_value=["time"])
    def test_transform_pipeline(self, get_time_indeces_mock, fm_mock):
        input_mock = MagicMock()
        input_mock.indexes = {"time": ["20.12.2020"]}
        step_two = MagicMock()
        result_mock = MagicMock()
        step_two.name = "mock"
        step_two.get_result.return_value = {"mock": result_mock}
        self.pipeline.add(module=step_two, input_ids=[1])

        result = self.pipeline.transform(x=input_mock)

        step_two.get_result.assert_called_once_with("20.12.2020", None, return_all=True)
        get_time_indeces_mock.assert_called_once_with({"x": input_mock})
        self.assertEqual({"mock": result_mock}, result)

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
        os_mock.path.isdir.return_value = False
        sub_pipeline = Pipeline(batch=pd.Timedelta("1h"))
        detector = MissingValueDetector()
        detector(dataset=sub_pipeline["test"])
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
        da = xr.DataArray([2, 3, 4, 3, 3, 1, 2], dims=["time"], coords={'time': time})
        pipeline = Pipeline(batch=pd.Timedelta("1h"))
        step_one = MagicMock()
        step_one.get_result.return_value = {"step": da}
        step_one.name = "step"
        result_mock = MagicMock()
        concat_mock.return_value = result_mock
        pipeline.start_steps["foo"] = StartStep("foo"), None
        pipeline.start_steps["foo"][0].last = False
        step_one.further_elements.side_effect = [True, True, True, True, True, True, True, False]
        pipeline.add(module=step_one, input_ids=[1])

        result = pipeline.transform(foo=da)

        self.assertEqual(concat_mock.call_count, 6)
        self.assertEqual(step_one.get_result.call_count, 7)
        self.assertEqual(step_one.further_elements.call_count, 8)
        self.assertEqual({"step": result_mock}, result)

    @patch('pywatts.core.pipeline.FileManager')
    def test_test(self, fm_mock):
        # Add some steps to the pipeline

        # Assert that the computation is set to fit_transform if the ComputationMode was default
        first_step = MagicMock()
        first_step.computation_mode = ComputationMode.Default
        first_step.finished = False
        time = pd.date_range('2000-01-01', freq='1H', periods=7)

        da = xr.DataArray([2, 3, 4, 3, 3, 1, 2], dims=["time"], coords={'time': time})

        first_step.get_result.return_value = {"first": da}
        second_step = MagicMock()
        second_step.computation_mode = ComputationMode.Train
        second_step.finished = False
        second_step.get_result.return_value = {"Second": da}

        self.pipeline.add(module=first_step)
        self.pipeline.add(module=second_step)

        self.pipeline.test(pd.DataFrame({"test": [1, 2, 2, 3, 4], "test2": [2, 2, 2, 2, 2]},
                                        index=pd.DatetimeIndex(pd.date_range('2000-01-01', freq='24H', periods=5))))

        first_step.get_result.assert_called_once_with(pd.Timestamp('2000-01-01 00:00:00', freq='24H'), None,
                                                      return_all=True)
        second_step.get_result.assert_called_once_with(pd.Timestamp('2000-01-01 00:00:00', freq='24H'), None,
                                                       return_all=True)

        first_step.set_computation_mode.assert_called_once_with(ComputationMode.Transform)
        second_step.set_computation_mode.assert_called_once_with(ComputationMode.Transform)

        first_step.reset.assert_called_once()
        second_step.reset.assert_called_once()

    @patch('pywatts.core.pipeline.FileManager')
    def test_train(self, fmmock):
        # Add some steps to the pipeline
        time = pd.date_range('2000-01-01', freq='1H', periods=7)

        da = xr.DataArray([2, 3, 4, 3, 3, 1, 2], dims=["time"], coords={'time': time})

        # Assert that the computation is set to fit_transform if the ComputationMode was default
        first_step = MagicMock()
        first_step.computation_mode = ComputationMode.Default
        first_step.finished = False
        first_step.get_result.return_value = {"first": da}

        second_step = MagicMock()
        second_step.computation_mode = ComputationMode.Train
        second_step.finished = False
        second_step.get_result.return_value = {"second": da}

        self.pipeline.add(module=first_step)
        self.pipeline.add(module=second_step)

        self.pipeline.train(pd.DataFrame({"test": [1, 2, 2, 3, 4], "test2": [2, 2, 2, 2, 2]},
                                         index=pd.DatetimeIndex(pd.date_range('2000-01-01', freq='24H', periods=5))))

        first_step.set_computation_mode.assert_called_once_with(ComputationMode.FitTransform)
        second_step.set_computation_mode.assert_called_once_with(ComputationMode.FitTransform)
        first_step.get_result.assert_called_once_with(pd.Timestamp('2000-01-01 00:00:00', freq='24H'), None,
                                                      return_all=True)
        second_step.get_result.assert_called_once_with(pd.Timestamp('2000-01-01 00:00:00', freq='24H'), None,
                                                       return_all=True)

        first_step.reset.assert_called_once()
        second_step.reset.assert_called_once()

    def test_horizon_greater_one_regression(self):
        lin_reg = LinearRegression()

        multi_regressor = SKLearnWrapper(lin_reg)(foo=self.pipeline["foo"], target=self.pipeline["target"],
                                                  target2=self.pipeline["target2"])
        RmseCalculator()(y=self.pipeline["target"], prediction=multi_regressor["target"])

        time = pd.date_range('2000-01-01', freq='24H', periods=5)

        foo = xr.DataArray([1, 2, 3, 4, 5], dims=["time"], coords={'time': time})
        target = xr.DataArray([[2, 3], [2, 4], [2, 5], [2, 6], [2, 7]], dims=["time", "horizon"],
                              coords={'time': time, "horizon": [1, 2]})
        target2 = xr.DataArray([3, 3, 3, 3, 3], dims=["time"], coords={'time': time})

        ds = xr.Dataset({'foo': foo, "target": target, "target2": target2})

        result, summary = self.pipeline.train(ds)
        self.assertAlmostEqual(result["RmseCalculator"].values[0, 0], 0.0)
