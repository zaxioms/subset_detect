import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class bbox(nn.Module):
    def __init__(self):
        super(bbox, self).__init__()
        resnet = models.resnet50(pretrained=False)
        layers = list(resnet.children())[:8]
        self.features1 = nn.Sequential(*layers[:6])
        self.features2 = nn.Sequential(*layers[6:])
        self.bb = nn.Sequential(nn.BatchNorm1d(2048), nn.Linear(2048, 4))

    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = F.relu(x)
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = x.view(x.shape[0], -1)
        return self.bb(x)
