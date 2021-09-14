from pywatts.modules.wrappers.sklearn_wrapper import SKLearnWrapper
from pywatts.modules.wrappers.keras_wrapper import KerasWrapper
try:
    from pywatts.modules.wrappers.pytorch_wrapper import PyTorchWrapper
except ModuleNotFoundError:
    pass

from pywatts.modules.wrappers.sm_time_series_model_wrapper import SmTimeSeriesModelWrapper
from pywatts.modules.wrappers.function_module import FunctionModule
