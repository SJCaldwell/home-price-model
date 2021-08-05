from comet_ml import Experiment
import torch
import torchvision
from torch import nn, optim
import numpy as np
from torchvision.transforms import Compose, CenterCrop, Resize, RandomHorizontalFlip, ToTensor, Normalize
from dataset.dataset import  HouseNumericalDataset
from model.model import HouseMLP
from loss.loss import MAPELoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.util import pct_accuracy

# Add the following code anywhere in your machine learning file
experiment = Experiment(api_key=None,
                        project_name="house-prices-mlp", workspace="sjcaldwell")

# Set Seed
np.random.seed(8)

BATCH_SIZE = 16
EPOCHS = 300

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using GPU')
else:
    device = torch.device('cpu')
    print('Using CPU')

### Define data once
full_dataset = HouseNumericalDataset(root_dir='houses_dataset/houses_dataset', csv_file='HousesInfo.txt')
train_size = int(0.70 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True) # not divisible by batch size, so batch norm layer fails.
val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = HouseMLP()
model.to(device)
opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3/200)
crit = MAPELoss

for e in tqdm(range(EPOCHS)):
    with experiment.train():
        # Training phase
        model.train()
        train_loss = 0
        for regr_vals, price in train_loader:
            regr_vals = regr_vals.to(device)
            price = price.to(device)
            est_price = model(regr_vals)
            loss = crit(price.squeeze(), est_price.squeeze())
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
            for regr_vals, price in val_loader:
                regr_vals = regr_vals.to(device)
                price = price.to(device)
                est_price = model(regr_vals)
                loss = crit(price.squeeze(), est_price.squeeze())
                val_loss += loss

            val_loss /= BATCH_SIZE
            experiment.log_metric('epoch_loss', val_loss.detach().cpu().numpy(), step=e)
        if e == EPOCHS - 1:
            print('recording MAPE')
            eval_crit = MAPELoss
            mape_loss = 0
            mape_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
            for regr_vals, price in mape_loader:
                regr_vals = regr_vals.to(device)
                price = price.to(device)
                est_price = model(regr_vals)
                loss = eval_crit(price.squeeze(), est_price.squeeze())
                mape_loss += loss
            final_mape = mape_loss/len(test_dataset)
            final_mape = final_mape.detach().cpu()
            experiment.log_metric('MAPE', final_mape)

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
