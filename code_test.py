import torch
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
import string
import re
import numpy as np
import os

import re
from azure.storage.blob import BlobClient, BlobServiceClient
from helper.utils import get_config

config = get_config()
azure_config = config["resource_config_path"]
STORAGE_ACCOUNT_URL = 'https://mlintro1651836008.blob.core.windows.net/'
print('getting blob service')
blob_service = BlobServiceClient(account_url=STORAGE_ACCOUNT_URL,
credential=os.environ["AZURE_STORAGE_CONNECTIONKEY"])
print(blob_service)
container = 'azureml-blobstore-31aeaa24-564c-4aa8-bdf4-fc4b5707bd1b/input/input'
root_container = 'azureml-blobstore-31aeaa24-564c-4aa8-bdf4-fc4b5707bd1b'
blob = 'tensor_train_batch_0_labels.pt'
print('getting client service')
blob_client = blob_service.get_blob_client(container, blob, snapshot=None)
print('getting container client')
blob_container_client = blob_service.get_container_client(root_container)
print('getting list of all blobs')
all_blobs = blob_container_client.list_blobs(name_starts_with="input/input/tensor")
print('print tensor blobs')
for b in all_blobs:
    print(b.name)
print(blob_client.exists())
