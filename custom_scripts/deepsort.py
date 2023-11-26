import sys
import os

current_script_path = __file__
current_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
parent_parent = os.path.dirname(parent_dir)
data_dir_ = parent_parent+'/dance_data'

import torch
import torch.nn.functional as F
import numpy as np

import dgl
import copy
import gc


#ScDeepSort Imports
from dance.modules.single_modality.cell_type_annotation.scdeepsort import ScDeepSort
from dance.utils import set_seed

os.environ["DGLBACKEND"] = "pytorch"
from pprint import pprint
from dance.datasets.singlemodality import ScDeepSortDataset

import scanpy as sc
from dance.transforms import AnnDataTransform, FilterGenesPercentile
from dance.transforms import Compose, SetConfig
from dance.transforms.graph import PCACellFeatureGraph, CellFeatureGraph
from dance.typing import LogLevel, Optional

import matplotlib.pyplot as plt

train_losses = []
train_accuracies = []


in_channels = 400
hidden_channels = 400
out_channels = 100
num_classes = 21


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(42)

#ScDeepSort

model = ScDeepSort(dim_in=in_channels, dim_hid=hidden_channels, num_layers=1, species='mouse', tissue='Kidney', device=device)
preprocessing_pipeline = Compose(
    AnnDataTransform(sc.pp.normalize_total, target_sum=1e4),
    AnnDataTransform(sc.pp.log1p),
    FilterGenesPercentile(min_val=1, max_val=99, mode="sum"),
)
def train_pipeline(n_components: int = 400, log_level: LogLevel = "INFO"):
    return Compose(
        PCACellFeatureGraph(n_components=n_components, split_name="train"),
        SetConfig({"label_channel": "cell_type"}),
        log_level=log_level,
    )
dataset = ScDeepSortDataset(species="mouse", tissue="Brain",
                            train_dataset=["753", "3285"], test_dataset=["2695"], data_dir = data_dir_)
data = dataset.load_data()
preprocessing_pipeline(data)
train_pipeline()(data)
data.to('cuda')
model.to('cuda')
y_train = data.get_train_data(return_type="torch")[1]
y_test = data.get_test_data(return_type="torch")[1]
y_train = torch.cat([y_train, y_test], dim=0)
y_train = torch.argmax(y_train, 1)
y_test = torch.argmax(y_test, 1)
print(y_train)
print(y_test)
print(data.data.uns['CellFeatureGraph'])
model.fit(graph=data.data.uns["CellFeatureGraph"], labels=y_train)
train_losses = model.train_losses
train_accuracies = model.train_accuracies

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, 300 + 1), train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

# Plotting the training accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, 300 + 1), train_accuracies, label='Train Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

result = model.predict_proba(graph=data.data.uns["CellFeatureGraph"])

result = result[4682:]
result = torch.tensor(result)
predicted = torch.argmax(result, 1)
#torch.set_printoptions(profile="full")
print(predicted)
correct = (predicted == y_test).sum().item()
total = y_test.numel()
accuracy = correct / total
print('accuracy: ', accuracy)