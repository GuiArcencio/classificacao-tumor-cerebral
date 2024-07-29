import numpy as np

from sklearn.linear_model import RidgeClassifierCV

import torch
from torch import tensor
from torch.nn import Conv2d, Parameter
from torch.utils.data import DataLoader

class Rocket:
    def __init__(
        self,
        n_filters=10000,
        batch_size=128,
        device="cpu",
        random_state=None
    ):
        self._n_filters = n_filters
        self._device = device
        self._classifier = RidgeClassifierCV()
        self._batch_size = batch_size
        self._kernels = []
        if type(random_state) == np.random.Generator:
            self._rng = random_state
        else:
            self._rng = np.random.default_rng(random_state)

    def fit(self, dataset):
        _, height, width = dataset[0][0].shape
        assert height == width, "Input images must be square"

        for _ in range(self._n_filters):
            length = self._rng.choice([7, 9, 11])

            weights = np.empty((1, 1, length, length), dtype=np.float32)
            weights[0,0,:,:] = self._rng.normal(size=(length, length))
            weights[0,0,:,:] = weights[0,0,:,:] - np.mean(weights[0,0,:,:])
            weights = tensor(weights, dtype=torch.float32, device=self._device)
            weights = Parameter(weights, requires_grad=False)

            bias = self._rng.uniform(-1, 1, size=(1,))
            bias = tensor(bias, dtype=torch.float32, device=self._device)
            bias = Parameter(bias, requires_grad=False)

            max_exponent = np.log2((height - 1) / (length - 1))
            dilation = np.floor(2**self._rng.uniform(0, max_exponent)).astype(int)

            padding = self._rng.choice(["valid", "same"])

            kernel = Conv2d(
                1, 1, length,
                padding=padding,
                dilation=dilation,
                device=self._device
            )
            kernel.weight = weights
            kernel.bias = bias

            self._kernels.append(kernel)

        transformed_dataset, labels = self.transform(dataset)

        self._classifier.fit(transformed_dataset, labels)

        return self
            
    def predict(self, dataset, return_true_labels=False):
        transformed_dataset, real_labels = self.transform(dataset)

        if return_true_labels:
            return real_labels, self._classifier.predict(transformed_dataset)
        
        return self._classifier.predict(transformed_dataset)

    def transform(self, dataset):
        loader = DataLoader(
            dataset,
            batch_size=self._batch_size,
            num_workers=2,
            pin_memory=True,
            pin_memory_device=self._device,
            shuffle=False,
        )
        transformed_dataset = np.empty((len(dataset), self._n_filters * 2))
        transformed_labels = np.empty(len(dataset), dtype=int)

        for batch_idx, (imgs, labels) in enumerate(loader):
            imgs = imgs.to(torch.device(self._device), non_blocking=True)

            idx_start = batch_idx * self._batch_size
            idx_end = idx_start + labels.shape[0]
            data_idx = np.arange(idx_start, idx_end)

            for i, kernel in enumerate(self._kernels):
                convolutions = kernel(imgs)
                _, _, conv_h, conv_w = convolutions.shape

                max_pool = convolutions.amax(dim=(1,2,3))

                ppv_pool = convolutions.gt(0).sum(dim=(1,2,3)) / (conv_h * conv_w)

                transformed_dataset[data_idx, 2*i] = max_pool.numpy(force=True)
                transformed_dataset[data_idx, 2*i + 1] = ppv_pool.numpy(force=True)

            transformed_labels[data_idx] = labels.numpy(force=True)

        return transformed_dataset, transformed_labels