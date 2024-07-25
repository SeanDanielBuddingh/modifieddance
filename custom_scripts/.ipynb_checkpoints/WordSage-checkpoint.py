import os
import sys

os.environ["DGLBACKEND"] = "pytorch"

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
import gc
import dgl
import pandas as pd

from data_pre  import data_pre
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torcheval.metrics import MulticlassAUROC
from dance.utils import set_seed

class WordSAGE(torch.nn.Module):
    def __init__(self, dim_tuple, hidden_channels, out_channels, num_classes):
        super(WordSAGE, self).__init__()
        self.seed = 42
        src_dim, dst_dim = dim_tuple
        self.conv1 = dgl.nn.HeteroGraphConv({
            'connects': dgl.nn.SAGEConv((src_dim, dst_dim), hidden_channels, 'mean'),
        }, aggregate='sum')
        self.conv2 = dgl.nn.HeteroGraphConv({
            'connects': dgl.nn.SAGEConv(hidden_channels, out_channels, 'mean'),
        }, aggregate='sum')

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(out_channels, out_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(out_channels, num_classes))

    def forward(self, x, features):
        h = self.conv1(x, features)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(x, {'gene_node': features['gene_node'], 'train_node': h['train_node']})
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.classifier(h['train_node'])
        return h

    def read_data(self, seed, dataset):
        data = data_pre()
        tissue_train, tissue_test, genes, y_values_train, y_values_test, normalized_train, normalized_test, _, _ = data.read_w2v(dataset)

        # concatenating either y_hat_markers or y_hard_markers to the input data
        additional_vars = pd.read_csv(data_dir_+"/ft_y_train.csv", header=None)
        tissue_train = pd.concat([tissue_train, additional_vars], axis=1)
        additional_vars = pd.read_csv(data_dir_+"/ft_y_test.csv", header=None)
        tissue_test = pd.concat([tissue_test, additional_vars], axis=1)

        normalized_train = normalized_train.T.reset_index(drop=True)
        normalized_test = normalized_test.T.reset_index(drop=True)
        tissue_train = tissue_train.reset_index(drop=True)
        tissue_test = tissue_test.reset_index(drop=True)

        genes = genes.set_index(genes.iloc[:,0], drop=True)
        genes = genes.drop(0, axis=1)
        genes.columns = range(2500)

        genes = genes.loc[normalized_train.columns]

        if not genes.index.equals(normalized_train.columns):
            print('mismatch')
            print(genes.index, normalized_train.columns)

        label_encoder = LabelEncoder().fit(y_values_train[0])
        targets_encoded_train = pd.Series(label_encoder.transform(y_values_train[0]))
        targets_encoded_test = pd.Series(label_encoder.transform(y_values_test[0]))


        inputs_train, targets_train = self.mix_data(seed, tissue_train, targets_encoded_train)
        inputs_test, targets_test = self.mix_data(seed, tissue_test, targets_encoded_test)

        train_graph, train_nodes = self.basic_dgl_graph(inputs_train, genes, normalized_train)
        test_graph, test_nodes = self.basic_dgl_graph(inputs_test, genes, normalized_test)

        return train_graph, targets_train, test_graph, targets_test, train_nodes, test_nodes

    def basic_dgl_graph(self, train_inputs, genes, normalized):
        num_train_nodes = len(train_inputs)
        num_gene_nodes = len(genes)

        edge_src = []
        edge_dst = []
        edge_weights = []
        for i, cell_name in enumerate(train_inputs.index):
            vector = normalized.iloc[cell_name,:]

            for j, expression in enumerate(vector):
                if expression == 0:
                    continue
                else:
                    edge_dst.append(i)
                    edge_src.append(j)
                    edge_weights.append(expression)
 
        graph_data = {
            ('gene_node', 'connects', 'train_node'): (torch.tensor(edge_src), torch.tensor(edge_dst))
        }
        G = dgl.heterograph(graph_data)

        train_features = torch.tensor(train_inputs.to_numpy(), dtype=torch.float32)
        G.nodes['train_node'].data['features'] = train_features
        gene_features = torch.tensor(genes.to_numpy(), dtype=torch.float32)
        G.nodes['gene_node'].data['features'] = gene_features
        
        G.nodes['train_node'].data['cell_id'] = torch.tensor([-1] * num_train_nodes)
        G.nodes['gene_node'].data['cell_id'] = torch.tensor(list(range(num_gene_nodes)))

        G.edges['connects'].data['weight'] = torch.tensor(edge_weights, dtype=torch.float).view(-1, 1)

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
