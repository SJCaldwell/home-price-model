"""
Run MAPE against a given model. Try on both the train and test set. Get average and std deviation home price
"""

from comet_ml import Experiment
import torch
import torchvision
from torch import nn, optim
import numpy as np
from torchvision.transforms import Compose, CenterCrop, Resize, RandomHorizontalFlip, ToTensor, Normalize
from dataset.dataset import HousePriceDataset
from model.model import HousePriceModel
from loss.loss import MAPELoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.util import pct_accuracy

np.random.seed(666)
BATCH_SIZE=1

train_transforms = Compose([Resize((128, 128)), RandomHorizontalFlip(), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
full_dataset = HousePriceDataset(root_dir='../data/Houses-dataset/Houses Dataset/', csv_file='HousesInfo.txt', transform=train_transforms)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True) # not divisible by batch size, so batch norm layer fails.
val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

print("TRAIN MAPE:")

if torch.cuda.is_available():
    device = torch.device('cuda')

model = HousePriceModel(dropout=0.0)
checkpoint = torch.load("model_checkpoints/model_d25a2d25ca9043b1892fe937db62f518_checkpoint.pt", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
opt = optim.Adam(model.parameters(), lr=.0001, weight_decay=1e-3/200)
crit = MAPELoss

with torch.no_grad():
    model.eval()
    val_loss = 0
    for bathroom, bedroom, frontal, kitchen, regr_vals, price in val_loader:
        bathroom = bathroom.to(device)
        bedroom = bedroom.to(device)
        frontal = frontal.to(device)
        kitchen = kitchen.to(device)
        regr_vals = regr_vals.to(device)
        price = price.to(device)
        est_price = model(bathroom, bedroom, frontal, kitchen, regr_vals)

        loss = crit(price.squeeze(), est_price.squeeze())
        val_loss += loss

print(val_loss/len(test_dataset))