import numpy as np
import torch
import torch.optim as optim
import cv2
from tqdm import tqdm

import models.bbox
import utils.data_loader


EPOCHS = 10

for epoch in tqdm(range(EPOCHS)):
    for data in trainset
