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
        #x = self.conv2(x, edge_index)
        #x = F.relu(x)
        x = self.classifier(x)
        return x#F.log_softmax(x, dim=1)

    def read_data(self, seed):
        data = data_pre()
        inputs, targets, genes, normalized_raw_data, test, y_test = data.read_w2v()
        encoding = np.hstack([targets, y_test])
        print(len(np.unique(encoding)))
        label_encoder = LabelEncoder().fit(encoding)
        targets_encoded = label_encoder.transform(targets)
        test_encoded = label_encoder.transform(y_test)
        #train_inputs, train_targets, test_inputs, test_targets = mix_data(seed, inputs, targets_encoded)
        train_inputs, train_targets = self.mix_data(seed, inputs, targets_encoded)
        test_inputs, test_targets = self.mix_data(seed, test, test_encoded)
        train_graph, train_nodes = self.basic_graph(train_inputs, genes, normalized_raw_data)
        gc.collect()
        test_graph, test_nodes = self.basic_graph(test_inputs, genes, normalized_raw_data)
        gc.collect()
        return train_graph, train_targets, test_graph, test_targets, train_nodes, test_nodes

    def basic_graph(self, train_inputs, genes, normalized):
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

    def mix_data(self, seed, inputs, targets):
        np.random.seed(seed)
        np.random.shuffle(inputs)
        np.random.seed(seed)
        np.random.shuffle(targets)
        #train_x, test_x = train_test_split(inputs, test_size=0.3, random_state=seed)
        #train_y, test_y = train_test_split(targets, test_size=0.3, random_state=seed)
        #return train_x, train_y, test_x, test_y

        return inputs, targets



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 42
set_seed(42)
in_channels = 100
hidden_channels = 100
out_channels = 100
num_classes = 16
model = WordSAGE(in_channels, hidden_channels, out_channels, num_classes).to(device)
train_graph, train_targets, test_graph, test_targets, train_nodes, test_nodes = WordSAGE.read_data(self=model, seed=seed)
train_targets = torch.tensor(train_targets, dtype=torch.long)
test_targets = torch.tensor(test_targets, dtype=torch.long)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_input_nodes = [i for i in range(train_graph.num_nodes) if i < train_nodes]
test_input_nodes = [i for i in range(test_graph.num_nodes) if i < test_nodes]

train_graph = train_graph.to(device)
test_graph = test_graph.to(device)
train_input_nodes = torch.tensor(train_input_nodes, dtype=torch.long).to(device)
test_input_nodes = torch.tensor(test_input_nodes, dtype=torch.long).to(device)
train_targets = torch.tensor(train_targets, dtype=torch.long).to(device)
test_targets = torch.tensor(test_targets, dtype=torch.long).to(device)

for epoch in range(300):
    model.train()
    optimizer.zero_grad()
    out = model(train_graph.x, train_graph.edge_index)
    loss = criterion(out[train_input_nodes], train_targets)
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    test_out = model(test_graph.x, test_graph.edge_index)
    test_loss = criterion(test_out[test_input_nodes], test_targets)
    test_out = F.softmax(test_out)
    predicted = torch.argmax(test_out[test_input_nodes], 1)
    print(predicted)
    print(test_targets)
    acc = accuracy_score(test_targets.cpu(), predicted.cpu())
    
    print(f"Epoch {epoch+1}, Train Loss: {loss.item()}, Test Loss: {test_loss.item()}, Test ACC: {acc}")


