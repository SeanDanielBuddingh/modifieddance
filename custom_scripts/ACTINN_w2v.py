import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import SAGEConv
import networkx as nx
from data_pre  import data_pre
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score
import gc
from scipy.sparse import csr_matrix
import dgl
import copy

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

#ACTINN
from dance.modules.single_modality.cell_type_annotation.actinn import ACTINN

#Celltypist
from dance.modules.single_modality.cell_type_annotation.celltypist import Celltypist

from WordSage import WordSAGE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 42
set_seed(42)
in_channels = 100
hidden_channels = 100
out_channels = 100
num_classes = 16
model = WordSAGE(in_channels, hidden_channels, out_channels, num_classes).to(device)
train_inputs, train_targets, test_inputs, test_targets = WordSAGE.read_data(self=model, seed=seed)
train_targets = torch.as_tensor(list(train_targets[0]), dtype=torch.long).to(device)#.reshape(-1, 1)
test_targets = torch.as_tensor(list(test_targets[0]), dtype=torch.long).to(device)#.reshape(-1, 1)

train_inputs = torch.as_tensor(train_inputs.to_numpy(), dtype=torch.float32).to(device)
test_inputs = torch.as_tensor(test_inputs.to_numpy(), dtype=torch.float32).to(device)
print(train_inputs.size())
print(train_targets.size())

#ACTINN
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

model = ACTINN(hidden_dims=[100, 100], lambd=0.01, device='cuda')

model.fit(train_inputs, train_targets, lr=0.001, num_epochs=300,
          batch_size=1000, print_cost=True)
pred = model.predict(test_inputs)
acc = accuracy_score(test_targets.cpu(), pred.cpu())
print(f"ACC: {acc}")
#print(model.model)