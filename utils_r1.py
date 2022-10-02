import torch
import pytorch_lightning as pl
import torch.nn as nn

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


class CoverTypeDataModule(pl.LightningDataModule):
    def __init__(self, X_train, y_train, X_val, y_val, batch_size = 64, split_seed = 42):
        super().__init__()
        self.save_hyperparameters()
    def setup(self, stage = None):
        pass
    def train_dataloader(self):
        return FastTensorDataLoader(self.hparams.X_train, self.hparams.y_train, batch_size = self.hparams.batch_size, shuffle = True)

    def val_dataloader(self):
        return FastTensorDataLoader(self.hparams.X_val, self.hparams.y_val, batch_size = self.hparams.batch_size, shuffle = False)



class MLP(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            #nn.Linear(44, 128),
            nn.LazyLinear(64),
            nn.ReLU(),
            # nn.Tanh(),
            nn.Linear(64, 32),
            nn.ReLU(),
            # nn.Tanh(),
            nn.Linear(32, 7)
        )

        self.criterion = nn.CrossEntropyLoss()
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
        return torch.optim.SGD(self.parameters(), lr=0.001)