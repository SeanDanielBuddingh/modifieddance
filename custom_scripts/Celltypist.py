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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 42
set_seed(42)

#Celltypist
model = Celltypist(majority_voting = False)
preprocessing_pipeline = model.preprocessing_pipeline()
dataset = ScDeepSortDataset(species="mouse", tissue="Brain",
                            train_dataset=["753", "3285"], test_dataset=["2695"])
data = dataset.load_data()
preprocessing_pipeline(data)
x_train, y_train = data.get_train_data(return_type="torch")
x_train = x_train.numpy()
y_train = torch.argmax(y_train, 1)
x_test, y_test = data.get_test_data(return_type="torch")
x_test = x_test.numpy()
y_test = torch.argmax(y_test, 1)
model.fit(indata = x_train, labels=y_train)
result = model.predict(x_test)
result = torch.tensor(result)
predicted = result#torch.argmax(result, 1)
print(predicted)

correct = (predicted == y_test).sum().item()
total = y_test.numel()
accuracy = correct / total
print('accuracy: ', accuracy)

