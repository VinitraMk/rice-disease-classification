from helper.utils import get_config, get_preproc_params
import pandas as pd
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms

class Transformer:
    preproc_args = None
    train_X = None
    y = None
    train_texts = None
    train_transforms = None
    test_transforms = None

    def __init__(self):
        self.preproc_args = get_preproc_params()
        self.train_transforms = A.Compose(
            [
                A.Resize(self.preproc_args['resize_len'], self.preproc_args['resize_len']),
                A.SmallestMaxSize(max_size=self.preproc_args['resize_len']),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
        self.test_transforms = A.Compose(
            [
                A.Resize(self.preproc_args['resize_len'], self.preproc_args['resize_len']),
                A.SmallestMaxSize(max_size=self.preproc_args['resize_len']),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    def get_transforms(self):
        return self.train_transforms, self.test_transforms