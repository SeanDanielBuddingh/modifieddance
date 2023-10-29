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

from WordSage import WordSAGE

in_channels = 100
hidden_channels = 100
out_channels = 100
num_classes = 21
WordSage = WordSAGE(in_channels, hidden_channels, out_channels, num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 42
set_seed(42)
train_graph, train_targets, test_graph, test_targets, train_nodes, test_nodes = WordSage.read_data(seed)
train_targets = torch.tensor(train_targets, dtype=torch.long)
test_targets = torch.tensor(test_targets, dtype=torch.long)
print(train_graph, train_targets, test_graph, test_targets, train_nodes, test_nodes)

#ScDeepSort

model = ScDeepSort(dim_in=in_channels, dim_hid=hidden_channels, num_layers=1, species='mouse', tissue='Kidney', device=device)

model.fit(graph=train_graph, labels=train_targets)
result = model.predict_proba(graph=test_graph)
result = torch.tensor(result)
predicted = torch.argmax(result, 1)
print(predicted)
correct = (predicted == test_targets).sum().item()
total = test_targets.numel()
accuracy = correct / total
print('accuracy: ', accuracy)
