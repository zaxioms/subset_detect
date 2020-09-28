import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from torch.utils.data import Dataset


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
        img_name = os.path.join(self.root_dir, self.df.iloc[idx, 0])
        image = cv2.imread(img_name, cv2.IMREAD_COLOR)
        bbox = self.df.iloc[idx, 1:]
        bbox = np.array([bbox])
        bbox = bbox.astype("float").reshape(4)

        if self.transform:
            image = self.transform(image)

        return (image, bbox)


if __name__ == "__main__":
    bd = BallDataset(
        csv_file="data/true_annotations/resized_annotations.csv",
        root_dir="data/resized_images",
    )

    fig, ax = plt.subplots(1)
    sample = bd[4320]
    plt.imshow(sample[0])
    rect = patches.Rectangle(
        (sample[1][0], sample[1][1]),
        sample[1][2],
        sample[1][3],
        linewidth=1,
        edgecolor="r",
        facecolor="none",
    )
    ax.add_patch(rect)
    plt.show()
