import h5py
import os
import numpy as np

from gzip import GzipFile

label_to_string = [
    "meningioma",
    "glioma",
    "pituitary_tumor"
]

os.makedirs("datasets/figshare/meningioma", exist_ok=True)
os.makedirs("datasets/figshare/glioma", exist_ok=True)
os.makedirs("datasets/figshare/pituitary_tumor", exist_ok=True)

for i in range(3064):
    with h5py.File(f'figshare/data/{i+1}.mat', 'r') as f:
        data = f['cjdata']
        label = int(data['label'][0,0]) - 1
        image = np.array(data['image'])

        folder = f"datasets/figshare/{label_to_string[label]}"

        with GzipFile(f"{folder}/{i+1}.npy.gz", "wb") as f:
            np.save(f, image, allow_pickle=False)
