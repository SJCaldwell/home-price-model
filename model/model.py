import torch
import torch.nn as nn
from torchvision.models import resnet18

import torch.nn.functional as F

'''

'''

# frontal features1 128x128
# bedroom features2
# bathroom features3
# kitchen features4
# numerical features - bed, bath, area, zip
simple_feature_extractor = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(12),
    nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(2),
    nn.ReLU(),
)


class HousePriceModel(nn.Module):
    def __init__(self, dropout=0.0):
        super(HousePriceModel, self).__init__()
        self.backend = resnet18(pretrained=True)
        for param in self.backend.parameters():
            param.requires_grad = False
        self.feature_extractor = nn.Sequential(*list(self.backend.children())[:-1])

        self.features1 = self.feature_extractor
        self.features2 = self.feature_extractor
        self.features3 = self.feature_extractor
        self.features4 = self.feature_extractor
        self.features5 = nn.Sequential(
            nn.Linear(10, 4),
            nn.ReLU(),
        )

        self.input_layer = nn.Linear(2052, 64)
        self.fc1 = nn.Linear(64, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(p=dropout)
        self.regressor = nn.Linear(64, 1)  # 4 images smoothed to 8 x 512 and then our 2 Numbers.

    def forward(self, x1, x2, x3, x4, x5):
        x1 = self.features1(x1)
        x2 = self.features2(x2)
        x3 = self.features3(x3)
        x4 = self.features4(x4)
        x5 = self.features5(x5)

        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x3 = x3.view(x3.size(0), -1)
        x4 = x4.view(x4.size(0), -1)
        x5 = x5.view(x5.size(0), -1)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.input_layer(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.regressor(x)
        return x


class HouseMLP(nn.Module):
    def __init__(self, dropout=0.0):
        super(HouseMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(10, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.Relu(),
            nn.Linear(4, 1))

    def forward(self, x):
        x = self.mlp(x)
        return x
