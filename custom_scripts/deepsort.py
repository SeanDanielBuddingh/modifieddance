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
import pandas as pd

import dgl
import copy
import gc

from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix

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
from WordSage import WordSAGE

train_losses = []
train_accuracies = []


in_channels = 400
hidden_channels = 400
out_channels = 100
num_classes = 21


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(42)
'''
trl = pd.read_csv(data_dir_+'/train/mouse/mouse_Brain999_celltype.csv', header=0)
tel = pd.read_csv(data_dir_+'/test/mouse/mouse_Brain9999_celltype.csv', header=0)
trl['Cell'] = "C_" + trl.index.astype(str)
tel['Cell'] = "C_" + tel.index.astype(str)
trl.to_csv(data_dir_+'/train/mouse/mouse_Brain999_celltype.csv', index=True)
tel.to_csv(data_dir_+'/test/mouse/mouse_Brain9999_celltype.csv', index=True)
'''
#ScDeepSort

model = ScDeepSort(dim_in=in_channels, dim_hid=hidden_channels, num_layers=1, species='mouse', tissue='Kidney', device=torch.device('cuda'))

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
dataset = ScDeepSortDataset(species="mouse", tissue="Kidney",
                            train_dataset=["4682"], test_dataset=["203"], data_dir = data_dir_)
data = dataset.load_data()
#preprocessing_pipeline(data)
#data = [train_inputs, test_inputs, 0, 0, 0]
train_pipeline()(data)

y_train = data.get_train_data(return_type="torch")[1]
y_test = data.get_test_data(return_type="torch")[1]
y_train = torch.cat([y_train, y_test], dim=0)
y_train = torch.argmax(y_train, 1)
y_test = torch.argmax(y_test, 1)
#print(y_train)
#print(y_test)
#print(data.data.uns['CellFeatureGraph'])
model.fit(graph=data.data.uns["CellFeatureGraph"], labels=y_train)
train_losses = model.train_losses
train_accuracies = model.train_accuracies
'''
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
'''
with torch.no_grad():
    result = model.predict_proba(graph=data.data.uns["CellFeatureGraph"])

result = result[-len(y_test):]
result = torch.tensor(result)
predicted = torch.argmax(result, 1)
#torch.set_printoptions(profile="full")
#print(predicted)
correct = (predicted == y_test).sum().item()
total = y_test.numel()
accuracy = correct / total

macro_auc = roc_auc_score(y_test.cpu(), result.cpu().detach(), multi_class='ovo', average='macro')
f1 = f1_score(y_test.cpu(), predicted.cpu(), average='macro')
precision = precision_score(y_test.cpu(), predicted.cpu(), average='macro')
recall = recall_score(y_test.cpu(), predicted.cpu(), average='macro')

# For specificity, calculate the confusion matrix and derive specificity
cm = confusion_matrix(y_test.cpu(), predicted.cpu())
specificity = np.sum(np.diag(cm)) / np.sum(cm)

print(f"ACC: {accuracy}")
print(f"Macro AUC: {macro_auc}")
print(f"F1: {f1}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Specificity: {specificity}")

