import numpy as np
import pandas as pd
from os import makedirs
from time import perf_counter

from sklearn.model_selection import StratifiedKFold

from torch.utils.data import Subset

from figshare import FigshareDataset
from models.minirocket import MiniRocket

N_REPETITIONS = 5
N_FOLDS = 10

if __name__ == "__main__":
    rng = np.random.default_rng(0x852fe139)

    makedirs("results/figshare/minirocket", exist_ok=True)

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=rng.integers(2**32 - 1))
    dataset = FigshareDataset()

    for n_features in [1000, 4000, 7000, 10000]:
        print(f"Testing MiniRocket with {n_features} features")

        for iteration in range(N_REPETITIONS):
            print(f"\tIteration {iteration}...")
            y_true = dataset._labels
            y_pred = np.empty_like(y_true)
            runtime = 0.

            for train_idx, test_idx in cv.split(dataset, y_true):
                train_dataset = Subset(dataset, train_idx)
                test_dataset = Subset(dataset, test_idx)
                model = MiniRocket(n_features=n_features, device="cuda", random_state=rng)

                start = perf_counter()
                model.fit(train_dataset)
                iter_y_pred = model.predict(test_dataset)
                end = perf_counter()

                y_pred[test_idx] = iter_y_pred
                runtime += end - start

            labels_df = {
                "true": y_true,
                "pred": y_pred,
            }
            labels_df = pd.DataFrame(labels_df)
            labels_df.to_csv(f"results/figshare/minirocket/{n_features}_features_run_{iteration}.csv", index=None)

            with open(f"results/figshare/minirocket/{n_features}_features_run_{iteration}_runtime.txt", "w") as f:
                f.write(f"{runtime / N_FOLDS:4f}")