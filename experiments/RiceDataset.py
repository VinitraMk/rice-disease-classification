import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from constants.types.model_enums import Model
import re
import cv2

#custom imports
from helper.utils import get_model_params, get_target_cols

class RiceDataset(Dataset):
    data_list = []
    transform = False
    class_to_idx = dict()

    def __init__(self, data_list, class_to_idx, transform = False):
        self.root_dir = os.environ["ROOT_DIR"]
        self.data_list = data_list
        self.transform = transform
        self.class_to_idx = class_to_idx
        

    def __len__(self):
        return len(self.data_list) 

    def __getitem__(self, idx):
        image_path = self.data_list[idx]['data_path']
        image = cv2.imread(image_path)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = -1
        if 'label' in self.data_list[idx]:
            label = self.data_list[idx]['label']
        #label = self.class_to_idx[label]
        if self.transform is not None:
            image = self.transform(image = image)['image']
        return image, label, self.data_list[idx]['Image_id']