from typing import Dict

import torch
import xarray as xr
import cloudpickle
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader

from pywatts.core.filemanager import FileManager
from pywatts.utils._split_kwargs import split_kwargs
from pywatts.utils._xarray_time_series_utils import numpy_to_xarray, xarray_to_numpy
from pywatts.modules.wrappers.dl_wrapper import DlWrapper


class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return torch.from_numpy(self.x[index, :]).float(), torch.from_numpy(self.y[index, :]).float()


class PyTorchWrapper(DlWrapper):
    """
    Wrapper for PyTorich Models.

    :param model: The pytorch model
    :type model: torch.nn.Module,
    :param name: The name of the wrappers
    :type name: str
    :param fit_kwargs: Key word arguments used for fitting the model.
    :type fit_kwargs: Optional[Dict]
    :param loss_fn: The loss function of the model.
    :type loss_fn: Callable
    :param optimizer: The optimizer of the model.
    :type optimizer: Object
    """

    def __init__(self, model: torch.nn.Module, optimizer, loss_fn, name: str = "PyTorchWrapper", fit_kwargs=None):
        super().__init__(model, name, fit_kwargs)
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def fit(self, **kwargs: xr.DataArray):
        """
        Calls the compile and the fit method of the wrapped pytorch module.
        """
        x, y = split_kwargs(kwargs)

        # check if gpu is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        batch_size_str = self.fit_kwargs['batch_size']
        epochs_str = self.fit_kwargs['epochs']

        x_np = xarray_to_numpy(x)
        y_np = xarray_to_numpy(y)

        dataset = TimeSeriesDataset(x_np, y_np)
        train_loader = DataLoader(dataset=dataset, batch_size=int(batch_size_str), shuffle=True)

        learning_rate = 1e-4
        scheduler = StepLR(self.optimizer, step_size=1)

        for epoch in range(1, int(epochs_str) + 1):
            self.model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                # put data to computing device (gpu)
                data, target = data.to(device), target.to(device)

                # Before the backward pass, use the optimizer object to zero all of the
                # gradients for the variables it will update (which are the learnable
                # weights of the model). This is because by default, gradients are
                # accumulated in buffers( i.e, not overwritten) whenever .backward()
                # is called. Checkout docs of torch.autograd.backward for more details.
                self.optimizer.zero_grad()

                # Forward pass: compute predicted y by passing x to the model.
                y_pred = self.model(data)

                # Compute loss
                loss = self.loss_fn(y_pred, target)

                # Backward pass: compute gradient of the loss with respect to model
                # parameters
                loss.backward()

                # Calling the step function on an Optimizer makes an update to its
                # parameters
                self.optimizer.step()

                # maybe do some printing and loss output

            # test routine
            self.model.eval()

            scheduler.step()

        self.model.to("cpu")
        self.is_fitted = True

    def transform(self, **kwargs: xr.DataArray) -> xr.DataArray:
        """
        Calls predict of the underlying PyTorch model.

        :param x: The dataset for which a prediction should be performed
        :return:  The prediction. Each output of the PyTorch model is a separate data variable in the returned xarray.
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        x_np = xarray_to_numpy(kwargs)
        x_dl = torch.from_numpy(x_np).float()

        self.model.eval()
        output = []
        with torch.no_grad():
            x_dl = x_dl.to(device)
            output = self.model(x_dl)

        pred = output.to("cpu").numpy()

        self.model.to("cpu")

        ret = numpy_to_xarray(pred, list(kwargs.values())[0])
        return ret

    def save(self, fm: FileManager):
        """
        Saves the pytorch wrappers and the containing model
        :param fm: Filemanager for getting the path
        :type fm: FileManager
        :return: Dictionary with additional information
        :rtype: Dict
        """

        json = super().save(fm)
        file_path = fm.get_path(f'{self.name}.pt')
        loss_fn_path = fm.get_path(f"loss_{self.name}.pickle")
        with open(loss_fn_path, "wb")as file:
            cloudpickle.dump(self.loss_fn, file)
        optimizer_path = fm.get_path(f"optimizer_{self.name}.pickle")
        with open(optimizer_path, "wb")as file:
            cloudpickle.dump(self.optimizer, file)
        torch.save(self.model, file_path)
        json.update({"pytorch_module": file_path, "optimizer": optimizer_path, "loss_fn": loss_fn_path})
        return json

    @classmethod
    def load(cls, load_information) -> 'PyTorchWrapper':
        """
        Method for restoring a pytorch Wrapper.

        :param load_information: Dict which contains the information for restoring the wrappers
        :type load_information: Dict
        :return: The restored wrappers
        :rtype: PyTorchWrapper
        """
        name = load_information["name"]
        model = torch.load(load_information["pytorch_module"])
        with open(load_information["loss_fn"], "rb") as file:
            loss_fn = cloudpickle.load(file)
        with open(load_information["optimizer"], "rb") as file:
            optimizer = cloudpickle.load(file)
        module = cls(model=model, name=name, fit_kwargs=load_information["params"]["fit_kwargs"],
                     loss_fn=loss_fn, optimizer=optimizer)
        module.is_fitted = load_information["is_fitted"]

        return module

    def get_params(self) -> Dict[str, object]:
        """
        Returns the parameters of deep learning frameworks.
        :return: A dict containing the fit keyword arguments and the compile keyword arguments
        """
        return {
            "fit_kwargs": self.fit_kwargs
        }

    def set_params(self, fit_kwargs=None, loss_fn=None, optimizer=None):
        """
        Set the parameters of the deep learning wrappers
        :param fit_kwargs: keyword arguments for the fit method.
        :param compile_kwargs: keyword arguments for the compile methods.
        """
        if fit_kwargs:
            self.fit_kwargs = fit_kwargs
        if loss_fn:
            self.loss_fn = loss_fn
        if optimizer:
            self.optimizer = optimizer
