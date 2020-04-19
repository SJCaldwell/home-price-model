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
            nn.ReLU(),
             nn.Linear(4, 1))

    def forward(self, x):
        x = self.mlp(x)
        return x

# Let's define a module meant to make up a smaller portion of our CNN
class CNN_Unit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNN_Unit, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, kernel_size=3, out_channels=out_channels)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet,self).__init__()

        self.unit1 = CNN_Unit(in_channels=3, out_channels=32)
        self.unit2 = CNN_Unit(in_channels=32,out_channels=32)
        self.unit3 = CNN_Unit(in_channels=32,out_channels=32)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.unit4 = CNN_Unit(in_channels=32, out_channels=64)
        self.unit5 = CNN_Unit(in_channels=64, out_channels=64)
        self.unit6 = CNN_Unit(in_channels=64, out_channels=64)
        self.unit7 = CNN_Unit(in_channels=64, out_channels=64)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.unit8 = CNN_Unit(in_channels=64, out_channels=128)
        self.unit9 = CNN_Unit(in_channels=128, out_channels=128)
        self.unit10 = CNN_Unit(in_channels=128, out_channels=128)
        self.unit11 = CNN_Unit(in_channels=128, out_channels=128)

        self.pool3 =  nn.MaxPool2d(kernel_size=2)

        self.unit12 = CNN_Unit(in_channels=128, out_channels=128)

        self.avgpool = nn.AvgPool2d(kernel_size=2)

        self.net = nn.Sequential(self.unit1, self.unit2, self.unit3, self.pool1,
                                 self.unit4, self.unit5, self.unit6, self.unit7, self.pool2,
                                 self.unit8, self.unit9, self.unit10, self.unit11, self.pool3,
                                 self.unit12, self.avgpool)
    def forward(self, x):
        x = self.net(x)
        return x

class HouseMixed(nn.Module):
    def __init__(self, dropout=0.0):
        super(HouseMixed, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(10, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU())

        self.cnn = SimpleNet()
        self.fc1 = nn.Linear(1156, 4)
        self.relu1 = nn.ReLU()
        self.regr_head = nn.Linear(4, 1)

    def forward(self, x_img, x_regr):
        x1 = self.cnn(x_img)
        x2 = self.mlp(x_regr)
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x = torch.cat((x1, x2), dim=1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.regr_head(x)
        return x
