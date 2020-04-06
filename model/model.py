import torch
import torch.nn as nn

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
    nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(2),
    nn.ReLU(),
)


class HousePriceModel(nn.Module):
    def __init__(self):
        super(HousePriceModel, self).__init__()
        self.features1 = simple_feature_extractor
        self.features2 = simple_feature_extractor
        self.features3 = simple_feature_extractor
        self.features4 = simple_feature_extractor
        self.features5 = nn.Sequential(
            nn.Linear(4, 2),
            nn.ReLU(),
        )
        self.regressor = nn.Linear((64 * 64 * 12) * 4 + 2,
                                   1)  # 4 images smoothed to 64 x 64 x 12, and then our 2 Numbers.

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
        x = self.regressor(x)
        return x
