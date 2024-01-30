import torch
import torch.nn.functional as F
import numpy as np

import dgl
import copy
import gc

#ScDeepSort Imports
from dance.modules.single_modality.cell_type_annotation.scdeepsort import ScDeepSort
from dance.utils import set_seed

import os
os.environ["DGLBACKEND"] = "pytorch"
from pprint import pprint
from dance.datasets.singlemodality import ScDeepSortDataset

import scanpy as sc
from dance.transforms import AnnDataTransform, FilterGenesPercentile
from dance.transforms import Compose, SetConfig
from dance.transforms.graph import PCACellFeatureGraph, CellFeatureGraph
from dance.typing import LogLevel, Optional

from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import accuracy_score

from WordSage import WordSAGE

in_channels = 2500
hidden_channels = 2500
out_channels = 100
num_classes = 21
WordSage = WordSAGE(in_channels, hidden_channels, out_channels, num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 42
set_seed(42)
train_graph, train_targets, test_graph, test_targets, train_nodes, test_nodes = WordSage.read_data(seed)
train_targets = torch.tensor(train_targets[0].values, dtype=torch.long).to(device)
test_targets = torch.tensor(test_targets[0].values, dtype=torch.long).to(device)
print(train_graph, train_targets, test_graph, test_targets, train_nodes, test_nodes)

#ScDeepSort

model = ScDeepSort(dim_in=in_channels, dim_hid=hidden_channels, num_layers=1, species='mouse', tissue='Kidney', device=device)

model.fit(graph=train_graph, labels=train_targets)
prob = model.predict_proba(graph=test_graph)
result = torch.tensor(prob)
pred = torch.argmax(result, 1)

acc = accuracy_score(test_targets.cpu(), pred.cpu())

macro_auc = roc_auc_score(F.one_hot(test_targets, num_classes=num_classes).cpu(), prob, multi_class='ovo', average='macro')
f1 = f1_score(test_targets.cpu(), pred.cpu(), average='macro')
precision = precision_score(test_targets.cpu(), pred.cpu(), average='macro')
recall = recall_score(test_targets.cpu(), pred.cpu(), average='macro')

# For specificity, calculate the confusion matrix and derive specificity
cm = confusion_matrix(test_targets.cpu(), pred.cpu())
specificity = np.sum(np.diag(cm)) / np.sum(cm)

print(f"ACC: {acc}")
print(f"Macro AUC: {macro_auc}")
print(f"F1: {f1}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Specificity: {specificity}")