import numpy as np
import pandas as pd
from os import makedirs
from time import perf_counter

from sklearn.model_selection import StratifiedKFold

from torch.utils.data import Subset

from figshare import FigshareDataset
from rocket import Rocket

if __name__ == "__main__":
    rng = np.random.default_rng(0x852fe139)

    makedirs("results/figshare/rocket", exist_ok=True)

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=rng.integers(2**32 - 1))
    dataset = FigshareDataset()

    for i, (train_idx, test_idx) in enumerate(cv.split(dataset, dataset._labels)):
        train_dataset = Subset(dataset, train_idx)
        test_dataset = Subset(dataset, test_idx)

        for n_filters in [10]:
            model = Rocket(n_filters=n_filters, device="cuda", random_state=rng)

            start = perf_counter()
            model.fit(train_dataset)
            y_true, y_pred = model.predict(test_dataset, return_true_labels=True)
            end = perf_counter()

            labels_df = {
                "true": y_true,
                "pred": y_pred,
            }
            labels_df = pd.DataFrame(labels_df)
            labels_df.to_csv(f"results/figshare/rocket/{n_filters}_kernels_fold_{i}.csv", index=None)

            with open(f"results/figshare/rocket/{n_filters}_kernels_fold_{i}_runtime.txt", "w") as f:
                f.write(f"{end - start:4f}")
            


            






"""
    dataset = FigshareDataset()
    train_dataset, test_dataset = random_split(dataset, [0.7, 0.3])

    start = perf_counter()
    model = Rocket(n_filters=1000, device="cuda", seed=None)
    model.fit(train_dataset)
    y_true, y_pred = model.predict(test_dataset, return_true_labels=True)
    end = perf_counter()

    print(end - start)
    print(accuracy_score(y_true, y_pred))
"""