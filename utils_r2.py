import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torchmetrics.functional import mean_squared_log_error

class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


# RBF Layer

class RBF(nn.Module):
    """
    Transforms incoming data using a given radial basis function:
    u_{i} = rbf(||x - c_{i}|| / s_{i})
    Arguments:
        in_features: size of each input sample
        out_features: size of each output sample
    Shape:
        - Input: (N, in_features) where N is an arbitrary batch size
        - Output: (N, out_features) where N is an arbitrary batch size
    Attributes:
        centres: the learnable centres of shape (out_features, in_features).
            The values are initialised from a standard normal distribution.
            Normalising inputs to have mean 0 and standard deviation 1 is
            recommended.
        
        log_sigmas: logarithm of the learnable scaling factors of shape (out_features).
        
        basis_func: the radial basis function used to transform the scaled
            distances.
    """

    def __init__(self, in_features, out_features, basis_func):
        super(RBF, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centres = nn.Parameter(torch.Tensor(out_features, in_features))
        self.log_sigmas = nn.Parameter(torch.Tensor(out_features))
        self.basis_func = basis_func
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.centres, 0, 1)
        nn.init.constant_(self.log_sigmas, 0)

    def forward(self, input):
        size = (input.size(0), self.out_features, self.in_features)
        x = input.unsqueeze(1).expand(size)
        c = self.centres.unsqueeze(0).expand(size)
        distances = (x - c).pow(2).sum(-1).pow(0.5) / torch.exp(self.log_sigmas).unsqueeze(0)
        return self.basis_func(distances)


# RBFs

def gaussian(alpha):
    phi = torch.exp(-1*alpha.pow(2))
    return phi

def linear(alpha):
    phi = alpha
    return phi

def quadratic(alpha):
    phi = alpha.pow(2)
    return phi

def inverse_quadratic(alpha):
    phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2))
    return phi

def multiquadric(alpha):
    phi = (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi

def inverse_multiquadric(alpha):
    phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi

def spline(alpha):
    phi = (alpha.pow(2) * torch.log(alpha + torch.ones_like(alpha)))
    return phi

def poisson_one(alpha):
    phi = (alpha - torch.ones_like(alpha)) * torch.exp(-alpha)
    return phi

def poisson_two(alpha):
    phi = ((alpha - 2*torch.ones_like(alpha)) / 2*torch.ones_like(alpha)) \
    * alpha * torch.exp(-alpha)
    return phi

def matern32(alpha):
    phi = (torch.ones_like(alpha) + 3**0.5*alpha)*torch.exp(-3**0.5*alpha)
    return phi

def matern52(alpha):
    phi = (torch.ones_like(alpha) + 5**0.5*alpha + (5/3) \
    * alpha.pow(2))*torch.exp(-5**0.5*alpha)
    return phi

def haversine_distance(df, lat1, long1, lat2, long2):
    r = 6371  # average radius of Earth in kilometers
       
    phi1 = np.radians(df[lat1])
    phi2 = np.radians(df[lat2])
    delta_phi = np.radians(df[lat2]-df[lat1])
    delta_lambda = np.radians(df[long2]-df[long1])
     
    a = np.sin(delta_phi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = (r * c) # in kilometers

    return d

class TaxiDataModule(pl.LightningDataModule):
    def __init__(self, X_train, y_train, X_val, y_val, bs):
        super().__init__()

        self.save_hyperparameters()
    def setup(self, stage = None):
        pass
    def train_dataloader(self):
        return FastTensorDataLoader(self.hparams.X_train, self.hparams.y_train, batch_size=self.hparams.bs, shuffle=True)
    def val_dataloader(self):
        return FastTensorDataLoader(self.hparams.X_val, self.hparams.y_val, batch_size=self.hparams.bs, shuffle=False)


class RadialBasisNeuralNet(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size, rbf_func = gaussian):
        super().__init__()
        self.rbf = RBF(input_size, hidden_size, rbf_func)
        self.fc = nn.Linear(hidden_size, output_size)
        self.criterion = nn.MSELoss()
        self.save_hyperparameters()
    
    def forward(self, x):
        x = self.rbf(x)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        rmse = torch.sqrt(loss)
        rmsle = torch.sqrt(mean_squared_log_error(y_hat.squeeze(-1), y))

        self.log_dict({"train_loss": loss, "train_rmse": rmse, "train_rmsle": rmsle}, prog_bar=True, logger=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        rmse = torch.sqrt(loss)
        rmsle = torch.sqrt(mean_squared_log_error(y_hat.squeeze(-1), y))
        self.log_dict({"val_loss": loss, "val_rmse": rmse, "val_rmsle": rmsle}, prog_bar=True, logger=False)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)




class MLP(pl.LightningModule):
    def __init__(self, hidden_size = 64):
        super().__init__()

        self.net = nn.Sequential(
            #nn.Linear(44, 128),
            nn.LazyLinear(hidden_size),
            nn.ReLU(),
            #nn.Tanh(),
            # nn.Linear(64, 32),
            # nn.ReLU(),
            # nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        self.criterion = nn.MSELoss()
    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch#["features"], batch["labels"]

        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, logger=False)
        return loss#{"loss": loss, "predictions": y_hat, "labels": batch["labels"]}

    def validation_step(self, batch, batch_idx):
        x, y = batch#["features"], batch["labels"]

        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, prog_bar=True, logger=False)
        return loss#{"loss": loss, "predictions": y_hat, "labels": batch["labels"]}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
