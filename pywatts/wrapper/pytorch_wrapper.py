import torch
import xarray as xr
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader

from pywatts.core.filemanager import FileManager
from pywatts.utils._xarray_time_series_utils import numpy_to_xarray, xarray_to_numpy
from pywatts.wrapper.dl_wrapper import DlWrapper


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
    :param name: The name of the wrapper
    :type name: str
    :param fit_kwargs: Key word arguments used for fitting the model.
    :type fit_kwargs: Optional[Dict]
    :param compile_kwargs: Key word arguments used for compiling the model.
    :type compile_kwargs: Optional[Dict]
    """
    def __init__(self, model: torch.nn.Module, name: str = "PyTorchWrapper", fit_kwargs=None, compile_kwargs=None):
        super().__init__(model, name, fit_kwargs, compile_kwargs)

    def fit(self, x: xr.Dataset, y: xr.Dataset):
        """
        Calls the compile and the fit method of the wrapped keras module.

        :param x: The input data
        :param y: The target data
        """

        # check if gpu is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        # loss_str = self.compile_kwargs['loss']
        # optimizer_str = self.compile_kwargs['optimizer']
        # metrics_str = self.compile_kwargs['metrics']

        batch_size_str = self.fit_kwargs['batch_size']
        epochs_str = self.fit_kwargs['epochs']

        x_np = xarray_to_numpy(x)
        y_np = xarray_to_numpy(y)

        dataset = TimeSeriesDataset(x_np, y_np)
        train_loader = DataLoader(dataset=dataset, batch_size=int(batch_size_str), shuffle=True)

        loss_fn = torch.nn.MSELoss(reduction='sum')

        learning_rate = 1e-4
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=1)

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
                optimizer.zero_grad()

                # Forward pass: compute predicted y by passing x to the model.
                y_pred = self.model(data)

                # Compute loss
                loss = loss_fn(y_pred, target)

                # Backward pass: compute gradient of the loss with respect to model
                # parameters
                loss.backward()

                # Calling the step function on an Optimizer makes an update to its
                # parameters
                optimizer.step()

                # maybe do some printing and loss output

            # test routine
            self.model.eval()

            scheduler.step()

        self.model.to("cpu")
        self.is_fitted = True

    def transform(self, x: xr.Dataset) -> xr.Dataset:
        """
        Calls predict of the underlying PyTorch model.

        :param x: The dataset for which a prediction should be performed
        :return:  The prediction. Each output of the PyTorch model is a separate data variable in the returned xarray.
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        x_np = xarray_to_numpy(x)
        x_dl = torch.from_numpy(x_np).float()

        self.model.eval()
        output = []
        with torch.no_grad():
            x_dl = x_dl.to(device)
            output = self.model(x_dl)

        pred = output.to("cpu").numpy()

        self.model.to("cpu")

        '''
        coords = (
            # first dimension is number of batches. We assume that this is the time.
            ("time", list(x.coords.values())[0].to_dataframe().index.array),
            *[(f"dim_{j}", list(range(size))) for j, size in enumerate(pred.shape[1:])])
        data = {"prediction": (tuple(map(lambda x: x[0], coords)), pred),
                "time": list(x.coords.values())[0].to_dataframe().index.array}
        result = xr.Dataset(data)
        '''
        ret = numpy_to_xarray(pred, x, self.name)
        return ret

    def save(self, fm: FileManager):
        """
        Saves the pytorch wrapper and the containing model
        :param fm: Filemanager for getting the path
        :type fm: FileManager
        :return: Dictionary with additional information
        :rtype: Dict
        """

        json = super().save(fm)
        file_path = fm.get_path(f'{self.name}.pt')
        torch.save(self.model, file_path)
        json.update({"pytorch_module": file_path})

        # sm = torch.jit.script(self.model)

        return json

    @classmethod
    def load(cls, load_information) -> 'PyTorchWrapper':
        """
        Method for restoring a pytorch Wrapper.

        :param load_information: Dict which contains the information for restoring the wrapper
        :type load_information: Dict
        :return: The restored wrapper
        :rtype: PyTorchWrapper
        """
        name = load_information["name"]
        model = torch.load(load_information["pytorch_module"])
        module = cls(model=model, name=name, fit_kwargs=load_information["params"]["fit_kwargs"],
                     compile_kwargs=load_information["params"]["compile_kwargs"])
        module.is_fitted = load_information["is_fitted"]

        return module
