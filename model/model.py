import torch
import torch.nn as nn
import torch.nn.functional as F
'''
I’ve 2 images (1st is 256x256 and the second 64x64) and some data (list of 10 floats) as an input and I’d like to classify the data in 4 classes (for now).
4 images
'''

# frontal features1 128x128
# bedroom features2
# bathroom features3
# kitchen features4
# numerical features - bed, bath, area, zip

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.features1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.features2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.features3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.features4 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.features5 = nn.Sequential(
            nn.Linear(4, 2),
            nn.ReLU(),
        )
        self.regressor = nn.Linear((64 * 64 * 12) * 4 + 2, 1) # 4 images smoothed to 64 x 64 x 12, and then our 2 Numbers.

    def forward(self, x1, x2, x3, x4, x5):
        x1 = self.features1(x1)
        x2 = self.features2(x2)
        x3 = self.features3(x3)
        x4 = self.features(x4)
        x5 = self.features(x5)

        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x3 = x3.view(x3.size(0), -1)
        x4 = x4.view(x4.size(0), -1)
        x5 = x5.view(x5.size(0), -1)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.regressor(x)
        return x
