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

in_channels = 100
hidden_channels = 100
out_channels = 100
num_classes = 21
WordSage = WordSAGE(in_channels, hidden_channels, out_channels, num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 42
set_seed(42)
data = data_pre()
inputs, targets, genes, normalized_raw_data, test, y_test = data.read_w2v()

WordSage = WordSAGE(in_channels, hidden_channels, out_channels, num_classes)

encoding = np.hstack([targets, y_test])
label_encoder = LabelEncoder().fit(encoding)
targets_encoded = label_encoder.transform(targets)
targets_encoded = torch.tensor(targets_encoded, dtype=torch.long)
num_classes = max(targets_encoded)+1
#targets_encoded = F.one_hot(targets_encoded, num_classes=num_classes)
test_encoded = label_encoder.transform(y_test)
test_encoded = torch.tensor(test_encoded, dtype=torch.long)
#test_encoded = F.one_hot(test_encoded, num_classes=num_classes)
#train_inputs, train_targets, test_inputs, test_targets = mix_data(seed, inputs, targets_encoded)
train_inputs, train_targets = WordSage.mix_data(seed, inputs, targets_encoded)
test_inputs, test_targets = WordSage.mix_data(seed, test, test_encoded)
train_inputs = torch.tensor(train_inputs, dtype=torch.float32).numpy()
test_inputs = torch.tensor(test_inputs, dtype=torch.float32).numpy() 


#Celltypist
model = Celltypist(majority_voting = False)
model.fit(indata = train_inputs, labels=train_targets)
result = model.predict(test_inputs)
result = torch.tensor(result)
predicted = result#torch.argmax(result, 1)
print(predicted)
y_test = test_targets
correct = (predicted == y_test).sum().item()
total = y_test.numel()
accuracy = correct / total
print('accuracy: ', accuracy)

