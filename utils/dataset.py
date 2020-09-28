import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from skimage import io
from torch.utils.data import Dataset

import raw_data_manager as dm

CUSTOM_JSON_LOC = "data/custom_json"
IMAGE_LOC = "data/images"
DEFINITIVE_JSON_LOC = CUSTOM_JSON_LOC + "definitive.json"


class BallDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.df.iloc[idx, 0])
        image = io.imread(img_name)
        bbox = self.df.iloc[idx, 1:]
        bbox = np.array([bbox])
        bbox = bbox.astype("float").reshape(4)
        print(bbox)
        sample = {"image": image, "bbox": bbox}

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == "__main__":
    dm.assert_data(CUSTOM_JSON_LOC, IMAGE_LOC)

    bd = BallDataset(
        csv_file="data/true_annotations/annotations.csv", root_dir="data/images"
    )

    fig, ax = plt.subplots(1)
    sample = bd[3650]
    plt.imshow(sample["image"])
    rect = patches.Rectangle(
        (sample["bbox"][0], sample["bbox"][1]),
        sample["bbox"][2],
        sample["bbox"][3],
        linewidth=1,
        edgecolor="r",
        facecolor="none",
    )
    ax.add_patch(rect)
    plt.show()
