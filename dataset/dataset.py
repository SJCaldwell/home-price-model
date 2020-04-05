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
        self.annotations.index += 1 #starting at 1 makes reading our csv easier
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_prefix = self.root_dir + str(idx) + "_"
        bathroom_img = Image.open(img_prefix + 'bathroom.jpg')
        bedroom_img = Image.open(img_prefix +'bedroom.jpg')
        frontal_img = Image.open(img_prefix +'frontal.jpg')
        kitchen_img = Image.open(img_prefix +'kitchen.jpg')
        num_bedrooms = self.annotations.iloc(idx - 1).num_bed
        num_bath = self.annotations.iloc(idx-1).num_bath
        area = self.annotations.iloc(idx-1).area
        zip_code = self.annotations.iloc(idx-1).zip
        price = self.annotations.iloc(idx-1).price

        return bathroom_img, bedroom_img, frontal_img, kitchen_img, [num_bedrooms, num_bath, area, zip_code], price