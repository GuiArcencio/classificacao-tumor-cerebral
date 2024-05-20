import numpy as np

import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.utils.data import random_split, DataLoader
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

import lightning as L

class BadzaNetwork(L.LightningModule):
    def __init__(
        self,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
            ),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=2,
            ),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
            ),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
            ),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),

            nn.Flatten(start_dim=-3),
            nn.Linear(
                in_features=4608,
                out_features=1024
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=1024,
                out_features=3
            ),
            nn.Softmax(dim=-1)
        )

        self.learning_rate = 0.0004

    def configure_optimizers(self):
        return Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.004
        )
    
    def forward(self, input):
        return self.model.forward(input)
    
    def training_step(self, batch):
        input_i, label_i = batch
        output_i = self.forward(input_i)
        loss = cross_entropy(output_i, label_i)

        return loss
    
    def validation_step(self, batch):
        input_i, label_i = batch
        output_i = self.forward(input_i)
        loss = cross_entropy(output_i, label_i)

        self.log("val_loss", loss)

        return loss

    def predict_step(self, batch):
        input_i, y_true = batch

        y_pred = torch.argmax(self.forward(input_i), 1)

        return (y_true, y_pred)
    
class BadzaCNN:
    def __init__(
        self,
        rng=None
    ):
        if type(rng) == np.random.Generator:
            self._rng = rng
        else:
            self._rng = np.random.default_rng(rng)

    def fit(self, dataset):
        self._model = BadzaNetwork()
        split_seed = int(self._rng.integers(2**32 - 1))

        train_dataset, val_dataset = random_split(
            dataset, [0.8, 0.2],
            generator=torch.Generator().manual_seed(split_seed)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=16,
            num_workers=2,
            pin_memory=True,
            pin_memory_device="cuda",
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=16,
            num_workers=2,
            pin_memory=True,
            pin_memory_device="cuda",
            shuffle=False,
        )

        self._trainer = L.Trainer(
            max_epochs=200,
            accelerator="gpu",
            callbacks=[
                EarlyStopping(
                    monitor="val_loss",
                    patience=11,
                    verbose=True,
                    strict=True,
                    mode="min",
                )
            ]
        )

        self._trainer.fit(self._model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    def predict(self, dataset, return_true_labels=False):
        test_loader = DataLoader(
            dataset,
            batch_size=16,
            num_workers=2,
            pin_memory=True,
            pin_memory_device="cuda",
            shuffle=False,
        )

        self._model.eval()
        y_true = []
        y_pred = []
        predictions = self._trainer.predict(self._model, test_loader)
        for (yi_true, yi_pred) in predictions:
            yi_true = yi_true.numpy(force=True)
            yi_pred = yi_pred.numpy(force=True)

            y_true.append(yi_true)
            y_pred.append(yi_pred)

        y_true = np.hstack(y_true)
        y_pred = np.hstack(y_pred)

        if return_true_labels:
            return y_true, y_pred
        else:
            return y_pred