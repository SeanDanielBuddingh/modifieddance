import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

import os
os.environ["DGLBACKEND"] = "pytorch"

import sys
sys.path.append("..")

from dgl.nn import SAGEConv

from data_pre  import data_pre

from sklearn.preprocessing import LabelEncoder


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
        tissue_train, tissue_test, genes, y_values_train, y_values_test, normalized_train, normalized_test, _, _ = data.read_w2v()
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

        return inputs_train, inputs_test, targets_train, targets_test

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


