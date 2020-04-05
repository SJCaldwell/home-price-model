import torch
import torchvision
from torch import nn, optim
import numpy as np
from torchvision.transforms import Compose, CenterCrop, Resize, RandomHorizontalFlip, ToTensor
from dataset.dataset import HousePriceDataset
from model.model import HousePriceModel
from torch.utils.data import DataLoader
from tqdm import tqdm

# Set Seed
np.random.seed(1)

BATCH_SIZE = 8
EPOCHS = 3

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using GPU')
else:
    device = torch.device('cpu')
    print('Using CPU')

### Define data once
train_transforms = Compose([Resize((128, 128)), RandomHorizontalFlip(), ToTensor()])
train_set = HousePriceDataset(root_dir='../data/Houses-dataset/Houses Dataset/', csv_file='HousesInfo.txt', transform=train_transforms)
train_set, ignore = torch.utils.data.random_split(train_set, [200, len(train_set)-200])

val_transforms = Compose([Resize((128, 128)), ToTensor()])
val_set = HousePriceDataset(root_dir='../data/Houses-dataset/Houses Dataset/', csv_file='HousesInfo.txt', transform=train_transforms)
val_set, ignore = torch.utils.data.random_split(val_set, [200, len(val_set)-200])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)

model = HousePriceModel()
model.to(device)
opt = optim.Adam(model.parameters(), lr=.001)
crit = nn.MSELoss()

for e in tqdm(range(EPOCHS)):
    # Training phase
    model.train()
    train_loss = 0
    for bathroom, bedroom, frontal, kitchen, regr_vals, price in train_loader:
        bathroom = bathroom.to(device)
        bedroom = bedroom.to(device)
        frontal = frontal.to(device)
        kitchen = kitchen.to(device)
        regr_vals = kitchen.to(regr_vals)
        price = price.to(device)
        est_price = model(bathroom, bedroom, frontal, kitchen, regr_vals)

        loss = crit(price, est_price)
        loss.backward()
        opt.step()
        train_loss += loss

    train_loss /= BATCH_SIZE
    #experiment.log_metric('epoch_loss', train_loss.detach().cpu().numpy(), step=e)

# Validation phase
    with torch.no_grad():
        model.eval()
        val_loss = 0
        for bathroom, bedroom, frontal, kitchen, regr_vals, price in train_loader:
            bathroom = bathroom.to(device)
            bedroom = bedroom.to(device)
            frontal = frontal.to(device)
            kitchen = kitchen.to(device)
            regr_vals = kitchen.to(regr_vals)
            price = price.to(device)
            est_price = model(bathroom, bedroom, frontal, kitchen, regr_vals)

            loss = crit(price, est_price)
            val_loss += loss

        val_loss /= BATCH_SIZE
        #experiment.log_metric('epoch_loss', val_loss.detach().cpu().numpy(), step=e)
        if e == EPOCHS - 1:
            print('recording final loss')
            #experiment.log_metric('final_loss', val_loss.detach().cpu().numpy())

    if e % 10 == 0:
        torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss},
            f'model_checkpoints/model_{e}_checkpoint.pt')
print('Training Complete')
del model
torch.cuda.empty_cache()  # empty model cache
