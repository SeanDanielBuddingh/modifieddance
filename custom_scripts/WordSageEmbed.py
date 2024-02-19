import os
import sys
current_script_path = __file__
current_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
parent_parent = os.path.dirname(parent_dir)
parent_parent = parent_parent.replace("\\", "/")
data_dir_ = parent_parent+'/dance_data'
import torch
import torch.nn.functional as F
import numpy as np
from dgl.nn import SAGEConv
import networkx as nx
from data_pre  import data_pre
from differential import GeneMarkers
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
        self.self_attention = torch.nn.MultiheadAttention(hidden_channels, num_heads=1)
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggregator_type='mean')
        self.bn1 = torch.nn.BatchNorm2d(hidden_channels)
        self.ln1 = torch.nn.LayerNorm(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggregator_type='mean')
        self.bn2 = torch.nn.BatchNorm2d(out_channels)

        self.linear = torch.nn.Linear(out_channels, out_channels)
        self.bce = torch.nn.Linear(out_channels, 1022)
        self.ce = torch.nn.Linear(1022, num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.leaky_relu(h)
        h, weights = self.self_attention(h,h,h)
        h = self.ln1(h)
        h = F.leaky_relu(h)
        h = F.dropout(h, p=0.2, training=self.training)
        h = self.conv2(x, h)
        h = self.bn2(h)
        x = F.leaky_relu(h)
        #decoder
        h = self.linear(h)
        h = F.relu(h)
        h = self.bce(h)
        y = F.relu(h)
        y = self.ce(y)
        return x, h, y

    def read_data(self, seed):
        data = data_pre()
        tissue_train, tissue_test, genes, y_values_train, y_values_test, normalized_train, normalized_test = data.read_w2v()

        normalized_train = normalized_train.T.reset_index(drop=True)
        normalized_test = normalized_test.T.reset_index(drop=True)
        tissue_train = tissue_train.reset_index(drop=True)
        tissue_test = tissue_test.reset_index(drop=True)

        label_encoder = LabelEncoder().fit(y_values_train)
        targets_encoded_train = pd.Series(label_encoder.transform(y_values_train))
        targets_encoded_test = pd.Series(label_encoder.transform(y_values_test))

        genemarkers = GeneMarkers()
        full_list_train, full_list_test = genemarkers.ConstructTargets()

        inputs_train, bce_targets_train, targets_train = self.mix_data(seed, tissue_train, full_list_train, targets_encoded_train)
        inputs_test, bce_targets_test, targets_test = self.mix_data(seed, tissue_test, full_list_test, targets_encoded_test)
        train_graph, train_nodes = self.basic_dgl_graph(inputs_train, genes, normalized_train)
        test_graph, test_nodes = self.basic_dgl_graph(inputs_test, genes, normalized_test)

        return train_graph, bce_targets_train, targets_train, test_graph, bce_targets_test, targets_test, train_nodes, test_nodes
    
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

    def mix_data(self, seed, inputs, bce_targets, ce_targets):
        np.random.seed(seed)
        print('Mixing Data\n')
        # Combine inputs and targets
        combined = pd.concat([inputs, bce_targets], axis=1)
        print(combined.shape)
        combined = pd.concat([combined, ce_targets], axis=1)
        print(combined.shape)
        # Shuffle the combined DataFrame
        combined_shuffled = combined.sample(frac=1).reset_index(drop=True)

        # Convert each row of targets to a single list
        bce_targets_shuffled = combined_shuffled.iloc[:, 2500:-1]
        ce_targets_shuffled = combined_shuffled.iloc[:, -1]

        ls = []
        for row in bce_targets_shuffled.iloc:
            new = [row]
            ls.append(new)
        bce_targets_shuffled = ls

        ls = []
        for row in ce_targets_shuffled.iloc:
            new = [row]
            ls.append(new)
        ce_targets_shuffled = ls

        # Separate inputs and targets
        inputs_shuffled = combined_shuffled.iloc[:, :2500]
        return inputs_shuffled, bce_targets_shuffled, ce_targets_shuffled


device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
seed = 42
set_seed(42)
in_channels = 2500
hidden_channels = 2500
out_channels = 2500
num_classes = 16
model = WordSAGE(in_channels, hidden_channels, out_channels, num_classes).to(device)
train_graph, bce_train_targets, train_targets, test_graph, bce_test_targets, test_targets, train_nodes, test_nodes = WordSAGE.read_data(self=model, seed=seed)
#test_graph, test_targets, train_graph, train_targets, test_nodes, train_nodes = WordSAGE.read_data(self=model, seed=seed)
train_targets = torch.tensor(train_targets, dtype=torch.long).to(device)
test_targets = torch.tensor(test_targets, dtype=torch.long).to(device)
bce_train_targets = torch.tensor(bce_train_targets).to(device)
bce_test_targets = torch.tensor(bce_test_targets).to(device)

#print(train_targets.shape)

bce_loss = torch.nn.BCEWithLogitsLoss()
ce_loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_input_nodes = [i for i in range(train_graph.number_of_nodes()) if i < train_nodes]
test_input_nodes = [i for i in range(test_graph.number_of_nodes()) if i < test_nodes]

train_graph = train_graph.to(device)
test_graph = test_graph.to(device)
train_input_nodes = torch.as_tensor(train_input_nodes, dtype=torch.long).to(device)
test_input_nodes = torch.as_tensor(test_input_nodes, dtype=torch.long).to(device)

for epoch in range(25):
    optimizer.zero_grad()
    feature, bce, out = model(train_graph, train_graph.ndata['features'])
    bce_cells = bce[(range(train_nodes))]
    out_cells = out[(range(train_nodes))]
    bce_loss_value = bce_loss(bce_cells, bce_train_targets.squeeze(1))
    ce_loss_value = ce_loss(out_cells, train_targets.squeeze(1))
    loss = ce_loss_value + bce_loss_value
    count=0
    for feat_out, target_out in zip(bce_cells, bce_train_targets.squeeze(1)):
        for logit, correct in zip(feat_out, target_out):
            logit = torch.sigmoid(logit)
            if logit >= 0.5:
                logit = 1
            else:
                logit = 0
            if logit == correct:
                count+=1
    bce_acc= count/(len(target_out)*len(out_cells))
    correct_predictions = 0
    total_predictions = 0
    for feat_out, target_out in zip(out_cells, train_targets.squeeze(1)):
        feat_out = F.softmax(feat_out)
        feat_out = torch.argmax(feat_out)
        correct_predictions += (target_out.detach().numpy() == feat_out.detach().numpy())
        total_predictions += 1
    ce_acc = correct_predictions / total_predictions
    loss.backward()
    optimizer.step()
    if (epoch+1) % 1 == 0:
        print(f"[UPDATE] [EPOCH {epoch + 1}] BCE Loss: {bce_loss_value:.4f} | CE Loss: {ce_loss_value:.4f} | Total Loss: {loss:.4f} | BCE_Acc: {bce_acc:.4f} | CE_Acc: {ce_acc:.4f}")

saved_features = pd.DataFrame(feature.cpu().detach()[(range(train_nodes))])
saved_features.to_csv(data_dir_+"/tuned_brain_train.csv",index=False, header=False)

del model, bce_loss, ce_loss, optimizer
torch.cuda.empty_cache() 
gc.collect()
model = WordSAGE(in_channels, hidden_channels, out_channels, num_classes).to(device)
bce_loss = torch.nn.BCEWithLogitsLoss()
ce_loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(25):
    optimizer.zero_grad()
    feature, bce, out = model(test_graph, test_graph.ndata['features'])
    bce_cells = bce[(range(test_nodes))]
    out_cells = out[(range(test_nodes))]
    bce_loss_value = bce_loss(bce_cells, bce_test_targets.squeeze(1))
    ce_loss_value = ce_loss(out_cells, test_targets.squeeze(1))
    loss = ce_loss_value + bce_loss_value
    count=0
    for feat_out, target_out in zip(bce_cells, bce_test_targets.squeeze(1)):
        for logit, correct in zip(feat_out, target_out):
            logit = torch.sigmoid(logit)
            if logit >= 0.5:
                logit = 1
            else:
                logit = 0
            if logit == correct:
                count+=1
    bce_acc= count/(len(target_out)*len(out_cells))
    correct_predictions = 0
    total_predictions = 0
    for feat_out, target_out in zip(out_cells, test_targets.squeeze(1)):
        feat_out = F.softmax(feat_out)
        feat_out = torch.argmax(feat_out)
        correct_predictions += (target_out.detach().numpy() == feat_out.detach().numpy())
        total_predictions += 1
    ce_acc = correct_predictions / total_predictions
    loss.backward()
    optimizer.step()
    if (epoch+1) % 1 == 0:
        print(f"[UPDATE] [EPOCH {epoch + 1}] BCE Loss: {bce_loss_value:.4f} | CE Loss: {ce_loss_value:.4f} | Total Loss: {loss:.4f} | BCE_Acc: {bce_acc:.4f} | CE_Acc: {ce_acc:.4f}")

saved_features = pd.DataFrame(feature.cpu().detach()[(range(test_nodes))])
saved_features.to_csv(data_dir_+"/tuned_brain_test.csv",index=False, header=False)


