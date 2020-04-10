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

# Add the following code anywhere in your machine learning file
experiment = Experiment(api_key="ueodw9bjrtM4LGohzeyY0zNLG",
                        project_name="house-prices", workspace="sjcaldwell")

# Set Seed
np.random.seed(666)

BATCH_SIZE = 4
EPOCHS = 200

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using GPU')
else:
    device = torch.device('cpu')
    print('Using CPU')

### Define data once
train_transforms = Compose([Resize((128, 128)), RandomHorizontalFlip(), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
full_dataset = HousePriceDataset(root_dir='../data/Houses-dataset/Houses Dataset/', csv_file='HousesInfo.txt', transform=train_transforms)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True) # not divisible by batch size, so batch norm layer fails.
val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = HousePriceModel(dropout=0.2)
model.to(device)
opt = optim.Adam(model.parameters(), lr=.0001, weight_decay=1e-3/200)
crit = nn.MSELoss()

for e in tqdm(range(EPOCHS)):
    with experiment.train():
        # Training phase
        model.train()
        train_loss = 0
        for bathroom, bedroom, frontal, kitchen, regr_vals, price in train_loader:
            bathroom = bathroom.to(device)
            bedroom = bedroom.to(device)
            frontal = frontal.to(device)
            kitchen = kitchen.to(device)
            regr_vals = regr_vals.to(device)
            price = price.to(device)
            est_price = model(bathroom, bedroom, frontal, kitchen, regr_vals)

            loss = crit(est_price.squeeze(), price.squeeze())
            loss.backward()
            opt.step()
            train_loss += loss

        train_loss /= BATCH_SIZE
        experiment.log_metric('epoch_loss', train_loss.detach().cpu().numpy(), step=e)

# Validation phase
    with experiment.validate():
        with torch.no_grad():
            model.eval()
            val_loss = 0
            for bathroom, bedroom, frontal, kitchen, regr_vals, price in train_loader:
                bathroom = bathroom.to(device)
                bedroom = bedroom.to(device)
                frontal = frontal.to(device)
                kitchen = kitchen.to(device)
                regr_vals = regr_vals.to(device)
                price = price.to(device)
                est_price = model(bathroom, bedroom, frontal, kitchen, regr_vals)

                loss = crit(price.squeeze(), est_price.squeeze())
                val_loss += loss

            val_loss /= BATCH_SIZE
            experiment.log_metric('epoch_loss', val_loss.detach().cpu().numpy(), step=e)
        if e == EPOCHS - 1:
            print('recording final loss')
            experiment.log_metric('final_loss', val_loss.detach().cpu().numpy())
            single_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
            pct_accurate = pct_accuracy(single_loader, model, device)
            experiment.log_metric('accuracy', pct_accurate)

    if e % 10 == 0:
        torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss},
            f'model_checkpoints/model_{experiment.get_key()}_checkpoint.pt')
print('Training Complete')
experiment.end()
del model
torch.cuda.empty_cache()  # empty model cache
