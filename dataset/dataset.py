import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np # linear algebra
import torchvision
from torchvision import transforms


class HousePriceDataset(Dataset):

    def __init__(self, root_dir, csv_file, transform=None):
        '''
        :param root_dir:
        :param csv_file:
        :param transform:
        '''
        self.root_dir = root_dir
        self.annotations = pd.read_csv(root_dir + '/' + csv_file, delim_whitespace=True)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_prefix = self.root_dir + str(idx+1) + "_" # indexing scheme different
        bathroom_img = Image.open(img_prefix + 'bathroom.jpg')
        bedroom_img = Image.open(img_prefix +'bedroom.jpg')
        frontal_img = Image.open(img_prefix +'frontal.jpg')
        kitchen_img = Image.open(img_prefix +'kitchen.jpg')
        num_bedrooms = self.annotations.iloc[idx, 0]
        num_bath = self.annotations.iloc[idx, 1]
        area = self.annotations.iloc[idx, 2]
        zip_code = self.annotations.iloc[idx, 3]
        price = self.annotations.iloc[idx, 4]

        if self.transform:
            bathroom_img = self.transform(bathroom_img)
            bedroom_img = self.transform(bedroom_img)
            frontal_img = self.transform(frontal_img)
            kitchen_img = self.transform(kitchen_img)

        return bathroom_img, bedroom_img, frontal_img, kitchen_img, torch.FloatTensor([num_bedrooms, num_bath, area, zip_code]), torch.tensor(price)