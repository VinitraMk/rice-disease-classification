import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import string
import re
import numpy as np
import os
import random
from matplotlib import pyplot as plt

#custom imports
from helper.utils import get_config, get_model_params, get_preproc_params, get_target_cols, save_tensor, save_model_supports, save_fig, find_value
from constants.types.label_encoding import LabelEncoding

class Preprocessor:
    preproc_args = None
    config = None
    class_to_idx = dict()
    idx_to_class = dict()
    train = []
    train_rgn = []
    valid = []
    valid_rgn = []
    test = []
    test_rgn = []
    class_count = 0

    def __plot_histogram(self, data, data_type):
        blast_list_filter = filter(lambda x: torch.equal(x['label'], torch.Tensor([1.0, 0.0, 0.0])), data)
        blast_list_len =  len(list(blast_list_filter))
        brown_list_filter = filter(lambda x: torch.equal(x['label'], torch.Tensor([0.0, 1.0, 0.0])), data)
        brown_list_len = len(list(brown_list_filter))
        healthy_list_filter = filter(lambda x: torch.equal(x['label'], torch.Tensor([0.0, 0.0, 1.0])), data)
        healthy_list_len = len(list(healthy_list_filter))
        plt.bar(x = ['blast', 'brown', 'healthy'], height = [blast_list_len, brown_list_len, healthy_list_len], width=0.5)
        plt.ylabel('No of data points with classes')
        save_fig(f'{data_type}_class_distribution', plt)
        plt.clf()

    def __init__(self):
        self.config = get_config()
        self.model_args = get_model_params()
        self.preproc_args = get_preproc_params()
        train_csv_path = f"{self.config['input_path']}\\train.csv"
        test_csv_path = f"{self.config['input_path']}\\test.csv"
        all_data = pd.read_csv(train_csv_path)
        rgn_selector = (all_data['Image_id'].str.endswith('rgn.jpg'))
        data = all_data.loc[~rgn_selector]
        data_rgn = all_data.loc[rgn_selector]
        data_list = []
        data_rgn_list = []
        for i, row in data.iterrows():
            data_path = f"{self.config['image_input_path']}\\{row['Image_id']}"
            label = self.__get_label(row)
            a = { 'data_path': data_path, 'label': label, 'Image_id': row['Image_id'], 'image_type': 'path' }
            data_list.append(a)
        for i, row in data_rgn.iterrows():
            data_path = f"{self.config['image_input_path']}\\{row['Image_id']}"
            label = self.__get_label(row)
            a = { 'data_path': data_path, 'label': label, 'Image_id': row['Image_id'], 'image_type': 'path' }
            data_rgn_list.append(a)
        random.shuffle(data_list)
        ei = int(len(data_list) * self.preproc_args['train_validation_split'])
        self.__plot_histogram(data_list, 'all_data')
        self.train = data_list[:ei]
        self.valid = data_list[ei:]
        self.train_rgn = []
        self.valid_rgn = []
        for data_el in self.train:
            val = data_el['Image_id'].replace('.jpg', '_rgn.jpg')
            el = find_value(data_rgn_list, 'Image_id', val)
            if el != None:
                self.train_rgn.append(el)
        for data_el in self.valid:
            val = data_el['Image_id'].replace('.jpg', '_rgn.jpg')
            el = find_value(data_rgn_list, 'Image_id', val)
            if el != None:
                self.valid_rgn.append(el)
        self.__plot_histogram(self.train, 'train_data')
        self.__plot_histogram(self.valid, 'valid_data')
        all_test = pd.read_csv(test_csv_path)
        rgn_selector = (all_test['Image_id'].str.endswith('rgn.jpg'))
        test = all_test.loc[~rgn_selector]
        test_rgn = all_test.loc[rgn_selector]
        for i, row in test.iterrows():
            data_path = f"{self.config['image_input_path']}\\{row['Image_id']}"
            a = { 'data_path': data_path, 'Image_id': row['Image_id'], 'image_type': 'path' }
            self.test.append(a)
        for i, row in test_rgn.iterrows():
            data_path = f"{self.config['image_input_path']}\\{row['Image_id']}"
            a = { 'data_path': data_path, 'Image_id': row['Image_id'], 'image_type': 'path' }
            self.test_rgn.append(a)
        print('\tClasses:', list(self.class_to_idx.keys()))

    def get_class_mappings(self):
        return self.class_to_idx, self.idx_to_class

    def get_data_paths(self):
        return self.train, self.train_rgn, self.test, self.test_rgn, self.valid, self.valid_rgn
    
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
            self.class_to_idx['blast'] = 0
            self.class_to_idx['brown'] = 1
            self.class_to_idx['healthy'] = 2
            labelarr = [0] * self.model_args['num_classes']
            if label == 'blast':
                labelarr[0] = 1
            elif label == 'brown':
                labelarr[1] = 1
            else:
                labelarr[2] = 1
            return torch.Tensor(labelarr)
    