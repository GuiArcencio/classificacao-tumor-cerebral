import numpy as np
import pandas as pd
from os import makedirs
from time import perf_counter

from sklearn.model_selection import StratifiedKFold

from torch.utils.data import Subset

from figshare import FigshareDataset
from models.rocket import Rocket
from models.minirocket import MiniRocket
from models.cnn import BadzaCNN

N_REPETITIONS = 5
N_FOLDS = 5

if __name__ == "__main__":
    # ROCKET
    rng = np.random.default_rng(0x852fe139)

    makedirs("results/figshare-patients/rocket", exist_ok=True)

    dataset = FigshareDataset()

    for n_filters in [10, 100, 1000, 10000]:
        print(f"Testing Rocket with {n_filters} kernels")

        for iteration in range(N_REPETITIONS):
            print(f"\tIteration {iteration}...")
            y_true = dataset._labels
            y_pred = np.empty_like(y_true)
            runtime = 0.

            for cv in [1,2,3,4,5]:
                train_idx = np.load(f"cvidx/fold{cv}_train_idx.npy")
                test_idx = np.load(f"cvidx/fold{cv}_test_idx.npy")

                train_dataset = Subset(dataset, train_idx)
                test_dataset = Subset(dataset, test_idx)
                model = Rocket(n_filters=n_filters, device="cuda", random_state=rng)

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
            labels_df.to_csv(f"results/figshare-patients/rocket/{n_filters}_kernels_run_{iteration}.csv", index=None)

            with open(f"results/figshare-patients/rocket/{n_filters}_kernels_run_{iteration}_runtime.txt", "w") as f:
                f.write(f"{runtime / N_FOLDS:4f}")

    # MiniRocket
    rng = np.random.default_rng(0x852fe139)

    makedirs("results/figshare-patients/minirocket", exist_ok=True)

    dataset = FigshareDataset()

    for n_features in [1000, 4000, 7000, 10000]:
        print(f"Testing MiniRocket with {n_features} features")

        for iteration in range(N_REPETITIONS):
            print(f"\tIteration {iteration}...")
            y_true = dataset._labels
            y_pred = np.empty_like(y_true)
            runtime = 0.

            for cv in [1,2,3,4,5]:
                train_idx = np.load(f"cvidx/fold{cv}_train_idx.npy")
                test_idx = np.load(f"cvidx/fold{cv}_test_idx.npy")
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
            labels_df.to_csv(f"results/figshare-patients/minirocket/{n_features}_features_run_{iteration}.csv", index=None)

            with open(f"results/figshare-patients/minirocket/{n_features}_features_run_{iteration}_runtime.txt", "w") as f:
                f.write(f"{runtime / N_FOLDS:4f}")

    # CNN
    rng = np.random.default_rng(0x852fe139)

    makedirs("results/figshare-patients/cnn", exist_ok=True)

    dataset = FigshareDataset()

    print(f"Testing CNN")

    for iteration in range(N_REPETITIONS):
        print(f"\tIteration {iteration}...")
        y_true = dataset._labels
        y_pred = np.empty_like(y_true)
        runtime = 0.

        for cv in [1,2,3,4,5]:
            train_idx = np.load(f"cvidx/fold{cv}_train_idx.npy")
            test_idx = np.load(f"cvidx/fold{cv}_test_idx.npy")
            train_dataset = Subset(dataset, train_idx)
            test_dataset = Subset(dataset, test_idx)
            model = BadzaCNN(rng=rng)

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
        labels_df.to_csv(f"results/figshare-patients/cnn/run_{iteration}.csv", index=None)

        with open(f"results/figshare-patients/cnn/run_{iteration}_runtime.txt", "w") as f:
            f.write(f"{runtime / N_FOLDS:4f}")