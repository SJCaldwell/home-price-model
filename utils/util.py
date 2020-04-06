import numpy as np
import torch

def pct_accuracy(val_loader, model, device, pct_close=0.25):
    with torch.no_grad():
        model.eval()
        n_correct = 0
        n_wrong = 0
        for bathroom, bedroom, frontal, kitchen, regr_vals, price in val_loader:
            bathroom = bathroom.to(device)
            bedroom = bedroom.to(device)
            frontal = frontal.to(device)
            kitchen = kitchen.to(device)
            regr_vals = regr_vals.to(device)
            price = price.to(device)
            est_price = model(bathroom, bedroom, frontal, kitchen, regr_vals).squeeze()
            price = price.squeeze()
            est_price = est_price.cpu()
            price = price.cpu()
            print("Actual price: {}. Estimated price: {}".format(price, est_price))
            if np.abs(est_price - price) < np.abs(pct_close * price):
                n_correct += 1
            else:
                n_wrong += 1
        return (n_correct * 100.0) / (n_correct + n_wrong)