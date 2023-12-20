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
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
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

class WordSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes):
        super(WordSAGE, self).__init__()
        self.seed = 42
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.classifier = torch.nn.Linear(out_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.classifier(x)
        return x#F.log_softmax(x, dim=1)

def read_data(seed):
    data = data_pre()
    inputs, targets, genes, normalized_raw_data, test, y_test = data.read_w2v()
    label_encoder = LabelEncoder().fit(targets)
    targets_encoded = label_encoder.transform(targets)
    test_encoded = label_encoder.transform(y_test)
    #train_inputs, train_targets, test_inputs, test_targets = mix_data(seed, inputs, targets_encoded)
    train_inputs, train_targets = mix_data(seed, inputs, targets_encoded)
    test_inputs, test_targets = mix_data(seed, test, test_encoded)
    train_graph, train_nodes = basic_dgl_graph(train_inputs, genes, normalized_raw_data)
    gc.collect()
    test_graph, test_nodes = basic_dgl_graph(test_inputs, genes, normalized_raw_data)
    gc.collect()
    return train_graph, train_targets, test_graph, test_targets, train_nodes, test_nodes

def basic_graph(train_inputs, genes, normalized):
    G = nx.Graph()
    for i in range(len(train_inputs)):
            G.add_node(i, features=train_inputs[i])
    nodes = int(G.number_of_nodes())
    for i in range(len(genes)):
            G.add_node(nodes+i, features=genes[i, 1:])
    for cell_name in normalized:
        if cell_name == '':
            pass
        vector = normalized[cell_name]
        nonzero = vector[vector!=0]
        for i, expression in enumerate(nonzero):
            G.add_edge(int(cell_name), nodes+i, weight=expression)
    del normalized
    gc.collect()

    adj = nx.adjacency_matrix(G)   
    adj_tensor = torch.sparse_coo_tensor(
        torch.LongTensor(np.vstack(adj.nonzero())),  
        torch.FloatTensor(adj.data),                 
        torch.Size(adj.shape)                        
    )
    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
    x_numpy = np.array([G.nodes[i]['features'] for i in G.nodes])
    x = torch.tensor(x_numpy, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, adj_matrix=adj_tensor), nodes

def basic_dgl_graph(train_inputs, genes, normalized):
    cell_name_to_index = {name: i for i, name in enumerate(normalized.keys())}
    
    num_train_nodes = len(train_inputs)
    num_gene_nodes = len(genes)
    G = dgl.DGLGraph()
    
    G.add_nodes(num_train_nodes + num_gene_nodes)

    train_feature_dim = train_inputs.shape[1]
    gene_feature_dim = genes.shape[1] - 1 
    max_feature_dim = max(train_feature_dim, gene_feature_dim)
    G.ndata['features'] = torch.zeros((num_train_nodes + num_gene_nodes, max_feature_dim), dtype=torch.float)

    train_features = torch.tensor(train_inputs, dtype=torch.float)
    G.ndata['features'][:num_train_nodes] = train_features
    
    gene_features = torch.tensor(genes[:, 1:], dtype=torch.float)
    G.ndata['features'][num_train_nodes:] = gene_features
    
    G.ndata['cell_id'] = torch.tensor(([-1] * num_train_nodes) + list(range(len(G.nodes())-num_train_nodes)))

    edge_src = []
    edge_dst = []
    edge_weights = []
    
    for cell_name in normalized:
        if cell_name == '':
            continue
        vector = normalized[cell_name]
        nonzero = vector[vector != 0]
        cell_index = cell_name_to_index[cell_name]
        
        for i, expression in enumerate(nonzero):
            edge_src.append(cell_index)
            edge_dst.append(num_train_nodes + i)
            edge_weights.append(expression)
            
    G.add_edges(edge_src, edge_dst)
    G.edata['weight'] = torch.tensor(edge_weights, dtype=torch.float).view(-1, 1)

    del normalized
    gc.collect()

    return G, num_train_nodes

def mix_data(seed, inputs, targets):
    np.random.seed(seed)
    
    combined = pd.concat([inputs, targets], axis=1)
    
    combined_shuffled = combined.sample(frac=1).reset_index(drop=True)

    num_input_columns = inputs.shape[1]
    inputs_shuffled = combined_shuffled.iloc[:, :num_input_columns]
    targets_shuffled = combined_shuffled.iloc[:, num_input_columns:]

    return inputs_shuffled, targets_shuffled


#ACTINN
print(torch.cuda.is_available())
device=torch.device('cuda')
model = ACTINN(hidden_dims=[256, 256], lambd=0.01, device='cuda')

preprocessing_pipeline = model.preprocessing_pipeline(normalize=True, filter_genes=True)
dataset = ScDeepSortDataset(species="mouse", tissue="Brain",
                            train_dataset=["753", "3285"], test_dataset=["2695"], data_dir = data_dir_)
data = dataset.load_data()
preprocessing_pipeline(data)
x_train, y_train = data.get_train_data(return_type="torch")
x_test, y_test = data.get_test_data(return_type="torch")
seed = 42
y_train = torch.argmax(y_train, dim=1)
y_test = torch.argmax(y_test, dim=1)

combined = torch.cat((x_train, x_test), dim=0)
y_comb = torch.cat((y_train, y_test), dim=0)
import pandas as pd
combined = pd.DataFrame(combined)
y_comb = pd.DataFrame(y_comb)
inputs, targets = mix_data(seed, combined, y_comb)
x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.2, random_state=seed, stratify=targets)
y_train = torch.tensor(y_train[0].values, dtype=torch.long).to(device)
y_test = torch.tensor(y_test[0].values, dtype=torch.long).to(device)

x_train = torch.tensor(x_train.to_numpy(), dtype=torch.float32).to(device)
x_test = torch.tensor(x_test.to_numpy(), dtype=torch.float32).to(device)

set_seed(42)

x_train.to(device)
y_train.to(device)
x_test.to(device)
y_test.to(device)
print(y_test.shape)

model.fit(x_train, y_train, lr=0.001, num_epochs=300,
          batch_size=1000, print_cost=True)
pred = model.predict(x_test)
acc = accuracy_score(y_test.cpu(), pred.cpu())

macro_auc = roc_auc_score(F.one_hot(y_test, num_classes=16).cpu(), F.softmax(model.model(x_test.to(device)).cpu()).detach(), multi_class='ovo', average='macro')
f1 = f1_score(y_test.cpu(), pred.cpu(), average='macro')
precision = precision_score(y_test.cpu(), pred.cpu(), average='macro')
recall = recall_score(y_test.cpu(), pred.cpu(), average='macro')

# For specificity, calculate the confusion matrix and derive specificity
cm = confusion_matrix(y_test.cpu(), pred.cpu())
specificity = np.sum(np.diag(cm)) / np.sum(cm)

print(f"ACC: {acc}")
print(f"Macro AUC: {macro_auc}")
print(f"F1: {f1}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Specificity: {specificity}")

