import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import string
import re
import numpy as np
import os
import random

#custom imports
from helper.utils import get_config, get_model_params, get_preproc_params, get_target_cols, save_tensor, save_model_supports
from constants.types.label_encoding import LabelEncoding

class Preprocessor:
    preproc_args = None
    config = None
    class_to_idx = dict()
    idx_to_class = dict()
    train = []
    test = []
    class_count = 0

    def __init__(self):
        self.config = get_config()
        self.model_args = get_model_params()
        self.preproc_args = get_preproc_params()
        train_csv_path = f"{self.config['input_path']}\\train.csv"
        test_csv_path = f"{self.config['input_path']}\\test.csv"
        data = pd.read_csv(train_csv_path)
        data = data.loc[data['Image_id'].str.endswith('rgn.jpg')]
        data_list = []
        for i, row in data.iterrows():
            data_path = f"{self.config['image_input_path']}\\{row['Image_id']}"
            label = self.__get_label(row)
            a = { 'data_path': data_path, 'label': label, 'Image_id': row['Image_id'] }
            data_list.append(a)
        random.shuffle(data_list)
        ei = int(len(data_list) * self.preproc_args['train_validation_split'])
        self.train = data_list[:ei]
        self.valid = data_list[ei:]
        test = pd.read_csv(test_csv_path)
        test = test.loc[test['Image_id'].str.endswith('rgn.jpg')]
        for i, row in test.iterrows():
            data_path = f"{self.config['image_input_path']}\\{row['Image_id']}"
            a = { 'data_path': data_path, 'Image_id': row['Image_id'] }
            self.test.append(a)
        print('\tTrain size:', len(self.train))
        print('\tValid size:', len(self.valid))
        print('\tTest size:', len(self.test))
        print('\tClasses:', list(self.class_to_idx.keys()))

    def get_class_mappings(self):
        return self.class_to_idx, self.idx_to_class

    def get_data_paths(self):
        return self.train, self.test, self.valid

    def __get_label(self, row):
        if (self.preproc_args['encoding_type'] == LabelEncoding.LABEL_ENCODING):
            label = row['Label']
            if label not in self.class_to_idx:
                self.class_to_idx[label] = self.class_count
                self.idx_to_class[self.class_count] = label
                self.class_count+=1
            return self.class_to_idx[label]
        elif (self.preproc_args['encoding_type'] == LabelEncoding.ONEHOT_ENCODING):
            label = row['Label']
            labelarr = [0] * self.model_args
            if label == 'blast':
                labelarr[0] = 1
            elif label == 'brown':
                labelarr[1] = 1
            else:
                labelarr[2] = 1
            return torch.Tensor(labelarr)
    