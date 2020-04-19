import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
import numpy as np # linear algebra
import torchvision
from torchvision import transforms

def clean_zips(df):
    """
    Some zipcodes are so infrequent that they're not likely to be useful as features. We should drop
    houses from these zip codes to improve our prediction. This could be extended to other categorical variables
    :param df: The pandas dataframe
    :return:
    """
    zip_codes = df["zip_code"].value_counts().keys().tolist()
    counts = df['zip_code'].value_counts().tolist()
    for (zip_code, count) in zip(zip_codes, counts):
        if count < 25:
            idxs = df[df["zip_code"] == zip_code].index
            df.drop(idxs, inplace=True)

def preprocess_continous(df, cont_var_names):
    cs = MinMaxScaler(feature_range=(1,2)) # prevent MAPE from dividing by 0
    df[cont_var_names] = cs.fit_transform(df[cont_var_names])

def preprocess_categorical(df, categorical_var_names):
    zip_binarizer = LabelBinarizer().fit(df[categorical_var_names])
    vals = zip_binarizer.fit_transform(df[categorical_var_names])
    df[categorical_var_names] = vals
class HouseMixedDataset(Dataset):

    def __init__(self, root_dir, csv_file, transform=None):
        '''
        :param root_dir:
        :param csv_file:
        :param transform:
        '''
        self.root_dir = root_dir
        self.annotations = pd.read_csv(root_dir + '/' + csv_file, delim_whitespace=True)
        clean_zips(self.annotations)
        self.zip_binarizer = LabelBinarizer().fit(self.annotations['zip_code'])
        preprocess_continous(self.annotations, ['num_bed', 'num_bath', 'area', 'price'])
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_prefix = self.root_dir + str(idx+1) + "_" # indexing scheme different
        #bathroom_img = Image.open(img_prefix + 'bathroom.jpg')
        #bedroom_img = Image.open(img_prefix +'bedroom.jpg')
        #frontal_img = Image.open(img_prefix +'frontal.jpg')
        kitchen_img = Image.open(img_prefix +'kitchen.jpg')
        num_bedrooms = self.annotations.iloc[idx, 0]
        num_bath = self.annotations.iloc[idx, 1]
        area = self.annotations.iloc[idx, 2]
        zip_code = self.zip_binarizer.transform([self.annotations.iloc[idx, 3]])
        price = self.annotations.iloc[idx, 4]

        if self.transform:
            #bathroom_img = self.transform(bathroom_img)
            #bedroom_img = self.transform(bedroom_img)
            #frontal_img = self.transform(frontal_img)
            kitchen_img = self.transform(kitchen_img)

        regr_independent_vars = np.hstack([num_bedrooms, num_bath, area, zip_code[0]])
        return kitchen_img, torch.Tensor(regr_independent_vars), torch.tensor(price)

class HouseNumericalDataset(Dataset):

    def __init__(self, root_dir, csv_file):
        '''
        :param root_dir:
        :param csv_file:
        '''
        self.root_dir = root_dir
        self.annotations = pd.read_csv(root_dir + '/' + csv_file, delim_whitespace = True)
        clean_zips(self.annotations)
        self.zip_binarizer = LabelBinarizer().fit(self.annotations['zip_code'])
        preprocess_continous(self.annotations, ['num_bed', 'num_bath', 'area', 'price'])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        num_bedrooms = self.annotations.iloc[idx, 0]
        num_bath = self.annotations.iloc[idx, 1]
        area = self.annotations.iloc[idx, 2]
        zip_code = self.zip_binarizer.transform([self.annotations.iloc[idx, 3]])
        price = self.annotations.iloc[idx, 4]
        # Place them in a single tensor
        regr_independent_vars = np.hstack([num_bedrooms, num_bath, area, zip_code[0]])
        return torch.Tensor(regr_independent_vars), torch.tensor(price)
