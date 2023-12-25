import os
import sys
current_script_path = __file__
current_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import torch
import torch.nn.functional as F
import numpy as np
from dgl.nn import SAGEConv
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
import pandas as pd
from torch.utils.data import DataLoader

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
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggregator_type='mean')
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggregator_type='mean')
        self.classifier = torch.nn.Linear(out_channels, num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.conv2(x, h)
        h = F.relu(h)
        h = self.classifier(h)
        return h

    def read_data(self, seed):
        data = data_pre()
        tissue_train, tissue_test, genes, y_values_train, y_values_test, normalized_train, normalized_test = data.read_w2v()
        normalized_train = normalized_train.T
        normalized_test = normalized_test.T
        print(len(np.unique(y_values_train)))
        tissue_train = tissue_train.reset_index(drop=True)
        tissue_test = tissue_test.reset_index(drop=True)
        normalized_train = normalized_train.reset_index(drop=True)
        normalized_test = normalized_test.reset_index(drop=True)
        label_encoder = LabelEncoder().fit(y_values_train)
        targets_encoded_train = pd.Series(label_encoder.transform(y_values_train))
        targets_encoded_test = pd.Series(label_encoder.transform(y_values_test))
        print(set(targets_encoded_train))
        inputs_train, targets_train = self.mix_data(seed, tissue_train, targets_encoded_train)
        inputs_test, targets_test = self.mix_data(seed, tissue_test, targets_encoded_test)
        #train_graph, train_nodes = self.basic_dgl_graph(X_train, genes, normalized)
        #test_graph, test_nodes = self.basic_dgl_graph(X_test, genes, normalized)
        return inputs_train, inputs_test, targets_train, targets_test
        #return train_graph, y_train, test_graph, y_test, train_nodes, test_nodes
    
    def basic_graph(self, train_inputs, genes, normalized):
        G = nx.Graph()
        for i in range(len(train_inputs)):
                G.add_node(i, features=train_inputs.iloc[i, :])
        nodes = int(G.number_of_nodes())
        for i in range(len(genes)):
                G.add_node(nodes+i, features=genes.iloc[i, 1:])
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
        x_list = [G.nodes[i]['features'] for i in G.nodes]
        x = torch.tensor(x_list, dtype=torch.float)

        return Data(x=x, edge_index=edge_index, adj_matrix=adj_tensor), nodes

    def basic_dgl_graph(self, train_inputs, genes, normalized):
        cell_name_to_index = {name: i for i, name in enumerate(normalized.keys())}
        
        num_train_nodes = len(train_inputs)
        num_gene_nodes = len(genes)
        G = dgl.DGLGraph()
        
        G.add_nodes(num_train_nodes + num_gene_nodes)

        train_feature_dim = train_inputs.shape[1]
        gene_feature_dim = genes.shape[1] - 1 
        max_feature_dim = max(train_feature_dim, gene_feature_dim)
        G.ndata['features'] = torch.zeros((num_train_nodes + num_gene_nodes, max_feature_dim), dtype=torch.float)

        train_features = torch.tensor(train_inputs.to_numpy(), dtype=torch.float32)
        G.ndata['features'][:num_train_nodes] = train_features
        
        gene_features = torch.tensor(genes.iloc[:, 1:].to_numpy(), dtype=torch.float32)
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

    def mix_data(self, seed, inputs, targets):
        np.random.seed(seed)

        combined = inputs
        combined['targets'] = targets.values
        combined_shuffled = combined.sample(frac=1).reset_index(drop=True)

        num_input_columns = inputs.shape[1] - 1
        inputs_shuffled = combined_shuffled.iloc[:, :num_input_columns]
        targets_shuffled = combined_shuffled.iloc[:, num_input_columns:]
        targets_shuffled.columns = [0]
        return inputs_shuffled, targets_shuffled

'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 42
set_seed(42)
in_channels = 100
hidden_channels = 100
out_channels = 100
num_classes = 16
model = WordSAGE(in_channels, hidden_channels, out_channels, num_classes).to(device)
train_graph, train_targets, test_graph, test_targets, train_nodes, test_nodes = WordSAGE.read_data(self=model, seed=seed)

train_targets = torch.tensor(train_targets[0].values, dtype=torch.long).to(device)
test_targets = torch.tensor(test_targets[0].values, dtype=torch.long).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_input_nodes = [i for i in range(train_graph.number_of_nodes()) if i < train_nodes]
test_input_nodes = [i for i in range(test_graph.number_of_nodes()) if i < test_nodes]

train_graph = train_graph.to(device)
test_graph = test_graph.to(device)
train_input_nodes = torch.as_tensor(train_input_nodes, dtype=torch.long).to(device)
test_input_nodes = torch.as_tensor(test_input_nodes, dtype=torch.long).to(device)

for epoch in range(300):
    model.train()
    optimizer.zero_grad()
    out = model(train_graph, train_graph.ndata['features'])
    loss = criterion(out[(range(train_nodes))], train_targets)
    loss.backward()
    optimizer.step()


model.eval()
with torch.no_grad():
    prob = model(test_graph, test_graph.ndata['features'])
    test_loss = criterion(prob[(range(test_nodes))], test_targets)

    test_out = F.softmax(prob[(range(test_nodes))])
    pred = torch.argmax(test_out, 1)

    acc = accuracy_score(test_targets.cpu(), pred.cpu())

    macro_auc = roc_auc_score(F.one_hot(test_targets, num_classes=num_classes).cpu(), prob[(range(test_nodes))].cpu(), multi_class='ovo', average='macro')
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
'''

