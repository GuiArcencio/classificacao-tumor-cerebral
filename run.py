from time import perf_counter

from sklearn.metrics import accuracy_score

import torch
from torch.utils.data import random_split

from figshare import FigshareDataset
from rocket import Rocket

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')

    dataset = FigshareDataset(device="cuda")
    train_dataset, test_dataset = random_split(dataset, [0.7, 0.3])

    start = perf_counter()
    model = Rocket(n_filters=100, device="cuda", seed=None)
    model.fit(train_dataset)
    y_true, y_pred = model.predict(test_dataset, return_true_labels=True)
    end = perf_counter()

    print(end - start)
    print(accuracy_score(y_true, y_pred))
