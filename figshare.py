import numpy as np

from os import listdir
from gzip import GzipFile

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Resize

class FigshareDataset(Dataset):
    label_to_string = [
        "meningioma",
        "glioma",
        "pituitary_tumor"
    ]

    def __init__(
        self,
        folder="datasets/figshare",
    ):
        self._folder = folder
        self._length = \
            len(listdir(f"{folder}/meningioma")) + \
            len(listdir(f"{folder}/glioma")) + \
            len(listdir(f"{folder}/pituitary_tumor"))
        self._labels = np.empty(self._length, dtype=int)
        self._resizer = Resize((256, 256))

        for img in listdir(f"{folder}/meningioma"):
            idx = int(img.split(".")[0])
            self._labels[idx-1] = 0
        for img in listdir(f"{folder}/glioma"):
            idx = int(img.split(".")[0])
            self._labels[idx-1] = 1
        for img in listdir(f"{folder}/pituitary_tumor"):
            idx = int(img.split(".")[0])
            self._labels[idx-1] = 2

    def __len__(self):
        return self._length
    
    def __getitem__(self, idx):
        label = self._labels[idx]
        tumor_type = self.label_to_string[label]

        with GzipFile(f"{self._folder}/{tumor_type}/{idx+1}.npy.gz") as f:
            data = np.load(f)

        data = data / np.max(data)
        data = data.astype(np.float32)
        data = torch.tensor(data, dtype=torch.float32)
        data = data.reshape((1, *data.shape))
        data = self._resizer(data)

        return data, label