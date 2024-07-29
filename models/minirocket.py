# TODO: check license stuff
# Adapted from https://github.com/angus924/minirocket

import numpy as np

from sklearn.linear_model import RidgeClassifierCV

import torch
from torch import tensor
from torch.nn import Conv2d, Parameter
from torch.utils.data import DataLoader

from itertools import combinations


class MiniRocket:
    def __init__(
        self,
        n_features=10000,
        max_dilations_per_kernel=32,
        batch_size=128,
        device="cpu",
        random_state=None
    ):
        self._n_features = n_features
        self._max_dilations_per_kernel = max_dilations_per_kernel
        self._device = device
        self._classifier = RidgeClassifierCV()
        self._batch_size = batch_size
        if type(random_state) == np.random.Generator:
            self._rng = random_state
        else:
            self._rng = np.random.default_rng(random_state)

    def _build_kernels(self, dataset, input_size, dilations, n_features_per_dilation, quantiles):
        n_examples = len(dataset)
        indices = np.array([_ for _ in combinations(np.arange(9), 3)], dtype=np.int32)

        n_kernels = len(indices)
        n_dilations = len(dilations)
        self._n_features = n_kernels * np.sum(n_features_per_dilation)

        paddings = ["valid", "same"]
        padding_i = 0
        feature_index_start = 0
        for dilation_index in range(n_dilations):
            dilation = dilations[dilation_index]
            n_features_this_dilation = n_features_per_dilation[dilation_index]

            for kernel_index in range(n_kernels):
                feature_index_end = feature_index_start + n_features_this_dilation

                weights = np.repeat(-1., 9).astype(np.float32)
                weights[indices[kernel_index]] = -2
                weights = weights.reshape((1, 1, 3, 3))
                weights = tensor(weights, dtype=torch.float32, device=self._device)
                weights = Parameter(weights, requires_grad=False)

                kernel = Conv2d(
                    in_channels=1,
                    out_channels=1,
                    kernel_size=3,
                    padding=paddings[padding_i],
                    dilation=dilation,
                    bias=False,
                    device=self._device
                )
                kernel.weight = weights

                random_example, _ = dataset[self._rng.integers(n_examples)]
                random_conv = kernel(random_example.to(torch.device(self._device)))

                this_quantiles = quantiles[feature_index_start:feature_index_end]
                this_quantiles = tensor(this_quantiles, dtype=torch.float32, device=self._device)
                bias = torch.quantile(random_conv, this_quantiles)

                self._kernels.append(kernel)
                self._biases.append(bias)

                feature_index_start = feature_index_end
                padding_i = (padding_i + 1) % 2             

    def fit(self, dataset):
        _, height, width = dataset[0][0].shape
        assert height == width, "Input images must be square"

        n_kernels = 84
        n_features_per_kernel = self._n_features // n_kernels
        true_max_dilations_per_kernel = min(n_features_per_kernel, self._max_dilations_per_kernel)
        multiplier = n_features_per_kernel / true_max_dilations_per_kernel

        max_exponent = np.log2((height - 1) / (9 - 1))
        dilations, n_features_per_dilation = \
        np.unique(np.logspace(0, max_exponent, true_max_dilations_per_kernel, base = 2).astype(np.int32), return_counts = True)
        n_features_per_dilation = (n_features_per_dilation * multiplier).astype(np.int32) # this is a vector

        remainder = n_features_per_kernel - np.sum(n_features_per_dilation)
        i = 0
        while remainder > 0:
            n_features_per_dilation[i] += 1
            remainder -= 1
            i = (i + 1) % len(n_features_per_dilation)

        n_features_per_kernel = np.sum(n_features_per_dilation)

        n_quantiles = n_kernels * n_features_per_kernel
        quantiles = np.array([(_ * ((np.sqrt(5) + 1) / 2)) % 1 for _ in range(1, n_quantiles + 1)], dtype = np.float32)

        self._kernels = []
        self._biases = []
        self._build_kernels(dataset, height, dilations, n_features_per_dilation, quantiles)

        transformed_dataset, real_labels = self.transform(dataset)

        self._classifier.fit(transformed_dataset, real_labels)

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
        transformed_dataset = np.empty((len(dataset), self._n_features))
        transformed_labels = np.empty(len(dataset), dtype=int)

        for batch_idx, (imgs, labels) in enumerate(loader):
            imgs = imgs.to(torch.device(self._device), non_blocking=True)

            idx_start = batch_idx * self._batch_size
            idx_end = idx_start + labels.shape[0]
            data_idx = np.arange(idx_start, idx_end)
            feature_idx = 0

            for kernel, biases in zip(self._kernels, self._biases):
                convolutions = kernel(imgs)
                _, _, conv_h, conv_w = convolutions.shape
                for i in range(biases.shape[0]):
                    results = convolutions - biases[i]

                    ppv_pool = results.gt(0).sum(dim=(1,2,3)) / (conv_h * conv_w)

                    transformed_dataset[data_idx, feature_idx] = ppv_pool.numpy(force=True)

                    feature_idx += 1

            transformed_labels[data_idx] = labels.numpy(force=True)

        return transformed_dataset, transformed_labels