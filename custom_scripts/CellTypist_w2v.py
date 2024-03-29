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
from torcheval.metrics import MulticlassAUROC

from sklearn.model_selection import train_test_split
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

#ACTINN
from dance.modules.single_modality.cell_type_annotation.actinn import ACTINN

#Celltypist
from dance.modules.single_modality.cell_type_annotation.celltypist import Celltypist, Classifier


from WordSageimport import WordSAGE

#Celltypist

device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
seed = 42
set_seed(42)
in_channels = 100
hidden_channels = 100
out_channels = 100
num_classes = 21
model = WordSAGE(in_channels, hidden_channels, out_channels, num_classes).to(device)
x_train, x_test, y_train, y_test = WordSAGE.read_data(self=model, seed=seed)

y_train = torch.tensor(y_train[0].values, dtype=torch.long).to(device)
y_test = torch.tensor(y_test[0].values, dtype=torch.long).to(device)

x_train = torch.tensor(x_train.to_numpy(), dtype=torch.float32).to(device)
x_test = torch.tensor(x_test.to_numpy(), dtype=torch.float32).to(device)

model = Celltypist(majority_voting = False)
set_seed(42)

x_train.to(device)
y_train.to(device)
x_test.to(device)
y_test.to(device)
print(y_test.shape)

model.fit(indata = x_train, labels=y_train)

pred = model.predict(x_test.numpy()) 
probs = model.classifier.predict_proba(x_test.numpy())

acc = accuracy_score(y_test.cpu(), pred)

auc = MulticlassAUROC(num_classes=num_classes)
auc.update(torch.as_tensor(probs).cpu(), y_test.cpu())
f1 = f1_score(y_test.cpu(), pred, average='macro')
precision = precision_score(y_test.cpu(), pred, average='macro')
recall = recall_score(y_test.cpu(), pred, average='macro')

# For specificity, calculate the confusion matrix and derive specificity
cm = confusion_matrix(y_test.cpu(), pred)
specificity = np.sum(np.diag(cm)) / np.sum(cm)

print(f"ACC: {acc}")
print(f"Macro AUC: {auc.compute()}")
print(f"F1: {f1}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Specificity: {specificity}")

