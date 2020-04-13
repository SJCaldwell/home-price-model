import torch
import torch.nn as nn
from torchvision.models import resnet18

class HousePriceModel(nn.Module):
    def __init__(self, dropout=0.0):
        super(HousePriceModel, self).__init__()
        self.backend = resnet18(pretrained=True)
        for param in self.backend.parameters():
            param.requires_grad = False
        self.feature_extractor = nn.Sequential(*list(self.backend.children())[:-1])

        self.features1 = self.feature_extractor
        self.features1_post = nn.Sequential(
            nn.MaxPool1d(kernel_size=32),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU()
        )
        self.max_pool1 = nn.MaxPool1d(kernel_size=32)
        self.fc_rep1_1 = nn.Linear(16, 8)
        self.fc_rep2_1 = nn.Linear(8, 4)

        self.features2 = self.feature_extractor
        self.max_pool2 = nn.MaxPool1d(kernel_size=32)
        self.fc_rep1_2 = nn.Linear(16, 8)
        self.fc_rep2_2 = nn.Linear(8, 4)

        self.features3 = self.feature_extractor
        self.max_pool3 = nn.MaxPool1d(kernel_size=32)
        self.fc_rep1_3 = nn.Linear(16, 8)
        self.fc_rep2_3 = nn.Linear(8, 4)

        self.features4 = self.feature_extractor
        self.max_pool4 = nn.MaxPool1d(kernel_size=32)
        self.fc_rep1_4 = nn.Linear(16, 8)
        self.fc_rep2_4 = nn.Linear(8, 4)

        self.features5 = nn.Sequential(
            nn.Linear(10, 4),
            nn.ReLU(),
        )
        self.input_layer = nn.Linear(20, 20)
        #self.fc1 = nn.Linear(64, 64)
        self.bn1 = nn.BatchNorm1d(20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 10)
        self.bn2 = nn.BatchNorm1d(10)
        self.dropout = nn.Dropout(p=dropout)
        self.regressor = nn.Linear(10, 1)  # 4 images smoothed to 8 x 512 and then our 2 Numbers.

    def forward(self, x1, x2, x3, x4, x5):
        x1 = self.features1(x1)
        x1 = x1.squeeze(3) # reduce hanging dimension
        x1 = x1.transpose(1, 2) # the 512 dimension needs to be last for max pooling to work
        x1 = self.max_pool1(x1)
        x1 = x1.transpose(1, 2) # and slap it back
        x1 = x1.unsqueeze(3) # add hanging dimension back
        x1 = x1.view(x1.size(0), -1)
        x1 = self.fc_rep1_1(x1)
        x1 = self.fc_rep2_1(x1)
        
        x2 = self.features2(x2)
        x2 = x2.squeeze(3)
        x2 = x2.transpose(1, 2)
        x2 = self.max_pool2(x2)
        x2 = x2.transpose(1, 2)
        x2 = x2.unsqueeze(3)  # add hanging dimension back
        x2 = x2.view(x2.size(0), -1)
        x2 = self.fc_rep1_2(x2)
        x2 = self.fc_rep2_2(x2)

        x3 = self.features3(x3)
        x3 = x3.squeeze(3)
        x3 = x3.transpose(1, 2)
        x3 = self.max_pool3(x3)
        x3 = x3.transpose(1, 2)
        x3 = x3.unsqueeze(3)  # add hanging dimension back
        x3 = x3.view(x3.size(0), -1)
        x3 = self.fc_rep1_3(x3)
        x3 = self.fc_rep2_3(x3)

        x4 = self.features4(x4)
        x4 = x4.squeeze(3)
        x4 = x4.transpose(1, 2)
        x4 = self.max_pool4(x4)
        x4 = x4.transpose(1, 2)
        x4 = x4.unsqueeze(3)  # add hanging dimension back
        x4 = x4.view(x4.size(0), -1)
        x4 = self.fc_rep1_4(x4)
        x4 = self.fc_rep2_4(x4)

        x5 = self.features5(x5)

        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x3 = x3.view(x3.size(0), -1)
        x4 = x4.view(x4.size(0), -1)
        x5 = x5.view(x5.size(0), -1)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.input_layer(x)
        #x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.regressor(x)
        return x
