import os
import sys
import gc

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
import pandas as pd

from sklearn.preprocessing import LabelEncoder

import dgl
from dgl.nn import SAGEConv

os.environ["DGLBACKEND"] = "pytorch"

from data_pre  import data_pre
from differential import GeneMarkers

from dance.utils import set_seed


class WordSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes):
        super(WordSAGE, self).__init__()
        self.seed = 42
        self.self_attention = torch.nn.MultiheadAttention(hidden_channels, num_heads=1)
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggregator_type='mean')
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.ln1 = torch.nn.LayerNorm(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggregator_type='mean')
        self.bn2 = torch.nn.BatchNorm1d(out_channels)

        self.linear = torch.nn.Linear(out_channels, out_channels)
        self.bce = torch.nn.Linear(out_channels, num_classes*10)
        self.ce = torch.nn.Linear(out_channels, num_classes)

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

        b = self.bce(h)

        y = self.ce(h)

        return x, b, y

    def read_data(self, seed):
        data = data_pre()
        tissue_train, tissue_test, genes, y_values_train, y_values_test, normalized_train, normalized_test = data.read_w2v()

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

        label_encoder = LabelEncoder().fit(y_values_train)
        targets_encoded_train = pd.Series(label_encoder.transform(y_values_train))
        targets_encoded_test = pd.Series(label_encoder.transform(y_values_test))

        genemarkers = GeneMarkers()
        full_list_train, full_list_test, _ = genemarkers.ConstructTargets(y_values_train[0], y_values_test[0], normalized_train, normalized_test)

        inputs_train, bce_targets_train_list, targets_train_list = self.mix_data(seed, tissue_train, full_list_train, targets_encoded_train)
        inputs_test, bce_targets_test_list, targets_test_list = self.mix_data(seed, tissue_test, full_list_test, targets_encoded_test)
        
        train_graphs, train_nodes_list = [], []
        for batch in inputs_train:
            train_graph, train_nodes = self.basic_dgl_graph(batch, genes, normalized_train)
            train_graphs.append(train_graph)
            train_nodes_list.append(train_nodes)

        test_graphs, test_nodes_list = [], []
        for batch in inputs_test:
            test_graph, test_nodes = self.basic_dgl_graph(batch, genes, normalized_test)
            test_graphs.append(test_graph)
            test_nodes_list.append(test_nodes)

        return train_graphs, bce_targets_train_list, targets_train_list, test_graphs, bce_targets_test_list, targets_test_list, train_nodes_list, test_nodes_list

    def basic_dgl_graph(self, train_inputs, genes, normalized):
        num_train_nodes = len(train_inputs)
        num_gene_nodes = len(genes)
        G = dgl.DGLGraph()
        
        G.add_nodes(num_train_nodes + num_gene_nodes)

        train_feature_dim = train_inputs.shape[1]
        gene_feature_dim = genes.shape[1] 
        max_feature_dim = max(train_feature_dim, gene_feature_dim)
        G.ndata['features'] = torch.zeros((num_train_nodes + num_gene_nodes, max_feature_dim), dtype=torch.float)

        train_features = torch.tensor(train_inputs.to_numpy(), dtype=torch.float32)
        G.ndata['features'][:num_train_nodes] = train_features
        
        gene_features = torch.tensor(genes.to_numpy(), dtype=torch.float32)
        G.ndata['features'][num_train_nodes:] = gene_features
        
        G.ndata['cell_id'] = torch.tensor(([-1] * num_train_nodes) + list(range(len(G.nodes())-num_train_nodes)))

        edge_src = []
        edge_dst = []
        edge_weights = []
        
        for i, cell_name in enumerate(train_inputs.index):
            vector = normalized.iloc[cell_name,:]
            cell_index = i

            for j, expression in enumerate(vector):
                if expression == 0:
                    continue
                else:
                    edge_src.append(cell_index)
                    edge_dst.append(num_train_nodes + j)
                    edge_weights.append(expression)
        
        G.add_edges(edge_src, edge_dst)
        G.edata['weight'] = torch.tensor(edge_weights, dtype=torch.float).view(-1, 1)

        del normalized
        gc.collect()

        return G, num_train_nodes

    def mix_data(self, seed, inputs, bce_targets, ce_targets):
        batch_size = 32
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

        inputs_shuffled = self.batch_data(inputs_shuffled, batch_size)
        bce_targets_shuffled = self.batch_data(bce_targets_shuffled, batch_size)
        ce_targets_shuffled = self.batch_data(ce_targets_shuffled, batch_size)

        return inputs_shuffled, bce_targets_shuffled, ce_targets_shuffled
    
    def batch_data(self, data, batch_size):
        data = pd.DataFrame(data)
        num_batches = len(data) // batch_size
        if len(data) % batch_size != 0:
            num_batches += 1
        print('num batches: ', num_batches)
        batches = []
        for i in range(num_batches):
            if i == num_batches-1:
                batch = data.iloc[ i * batch_size : ]
            else:
                batch = data.iloc[ i * batch_size : ( i + 1 ) * batch_size ]

            #batch = batch.values.tolist()
            batches.append(batch)
        
        return batches

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
seed = 42
in_channels = 2500
hidden_channels = 2500
out_channels = 2500
num_classes = 16
lr = 1e-2
momentum = 0.9

set_seed(seed)

model = WordSAGE(in_channels, hidden_channels, out_channels, num_classes).to(device)

train_graphs, bce_targets_train_list, targets_train_list, test_graphs, bce_targets_test_list, targets_test_list, train_nodes_list, test_nodes_list = WordSAGE.read_data(self=model, seed=seed)

bce_loss = torch.nn.BCEWithLogitsLoss()
ce_loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

for epoch in range(50):
    features_list = []
    targets_list = []
    count=0
    correct_predictions = 0
    total_predictions = 0
    bcc_denom = 0
    for train_graph, bce_targets_train, train_targets, train_nodes in zip(train_graphs, bce_targets_train_list, targets_train_list, train_nodes_list):

        train_targets = torch.tensor(train_targets.values, dtype=torch.long).to(device)

        bce_train_targets = torch.tensor(bce_targets_train.values.tolist()).to(device)

        train_input_nodes = [i for i in range(train_graph.number_of_nodes()) if i < train_nodes]

        train_graph = train_graph.to(device)

        train_input_nodes = torch.as_tensor(train_input_nodes, dtype=torch.long).to(device)

        optimizer.zero_grad()
        feature, bce, out = model(train_graph, train_graph.ndata['features'])

        bce_cells = bce[(range(train_nodes))]
        out_cells = out[(range(train_nodes))]
        bce_loss_value = bce_loss(bce_cells, bce_train_targets.squeeze(1))
        ce_loss_value = ce_loss(out_cells, train_targets.squeeze(1))
        loss = ce_loss_value + ( 0.1 * bce_loss_value )
        for feat_out, target_out in zip(bce_cells, bce_train_targets.squeeze(1)):
            for logit, correct in zip(feat_out, target_out):
                logit = torch.sigmoid(logit)
                if logit >= 0.5:
                    logit = 1
                else:
                    logit = 0
                if logit == correct:
                    count+=1
        bcc_denom += (len(target_out)*len(out_cells))
        for feat_out, target_out in zip(out_cells, train_targets.squeeze(1)):
            feat_out = F.softmax(feat_out, dim=0)
            feat_out = torch.argmax(feat_out)
            correct_predictions += (target_out.detach().cpu().numpy() == feat_out.detach().cpu().numpy())
            total_predictions += 1
        ce_acc = correct_predictions / total_predictions
        loss.backward()
        optimizer.step()
        features_list.append(feature.cpu().detach()[(range(train_nodes))])
        targets_list.append(train_targets)

    bce_acc= count/bcc_denom
    if (epoch+1) % 1 == 0:
        print(f"[UPDATE TRAIN SET] [EPOCH {epoch + 1}] BCE Loss: {bce_loss_value:.4f} | CE Loss: {ce_loss_value:.4f} | Total Loss: {loss:.4f} | BCE_Acc: {bce_acc:.4f} | CE_Acc: {ce_acc:.4f}")

features_dfs = [pd.DataFrame(tensor.numpy()) for tensor in features_list]
targets_dfs = [pd.DataFrame(tensor.cpu().detach().numpy()) for tensor in targets_list]

features_df = pd.concat(features_dfs, ignore_index=True)
targets_df = pd.concat(targets_dfs, ignore_index=True)

features_df.to_csv(data_dir_+"/tuned_brain_train.csv",index=False, header=False)
targets_df.to_csv(data_dir_+"/tuned_brain_train_labels.csv", index=False, header='Cell_Type')

del bce_loss, ce_loss, optimizer
torch.cuda.empty_cache() 
gc.collect()

bce_loss = torch.nn.BCEWithLogitsLoss()
ce_loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

for epoch in range(50):
    features_list = []
    targets_list = []
    count=0
    correct_predictions = 0
    total_predictions = 0
    bcc_denom = 0
    for test_graph, bce_targets_test, test_targets, test_nodes in zip(test_graphs, bce_targets_test_list, targets_test_list, test_nodes_list):

        test_targets = torch.tensor(test_targets.values, dtype=torch.long).to(device)

        bce_test_targets = torch.tensor(bce_targets_test.values.tolist()).to(device)

        test_input_nodes = [i for i in range(test_graph.number_of_nodes()) if i < test_nodes]

        test_graph = test_graph.to(device)

        test_input_nodes = torch.as_tensor(test_input_nodes, dtype=torch.long).to(device)

        optimizer.zero_grad()
        feature, bce, out = model(test_graph, test_graph.ndata['features'])

        bce_cells = bce[(range(test_nodes))]
        out_cells = out[(range(test_nodes))]
        bce_loss_value = bce_loss(bce_cells, bce_test_targets.squeeze(1))
        ce_loss_value = ce_loss(out_cells, test_targets.squeeze(1))
        loss = ce_loss_value + ( 0.1 * bce_loss_value )
        for feat_out, target_out in zip(bce_cells, bce_test_targets.squeeze(1)):
            for logit, correct in zip(feat_out, target_out):
                logit = torch.sigmoid(logit)
                if logit >= 0.5:
                    logit = 1
                else:
                    logit = 0
                if logit == correct:
                    count+=1
        bcc_denom += (len(target_out)*len(out_cells))
        for feat_out, target_out in zip(out_cells, test_targets.squeeze(1)):
            feat_out = F.softmax(feat_out, dim=0)
            feat_out = torch.argmax(feat_out)
            correct_predictions += (target_out.detach().cpu().numpy() == feat_out.detach().cpu().numpy())
            total_predictions += 1
        ce_acc = correct_predictions / total_predictions
        loss.backward()
        optimizer.step()
        features_list.append(feature.cpu().detach()[(range(test_nodes))])
        targets_list.append(test_targets)
        
    bce_acc= count/bcc_denom
    if (epoch+1) % 1 == 0:
        print(f"[UPDATE TEST SET] [EPOCH {epoch + 1}] BCE Loss: {bce_loss_value:.4f} | CE Loss: {ce_loss_value:.4f} | Total Loss: {loss:.4f} | BCE_Acc: {bce_acc:.4f} | CE_Acc: {ce_acc:.4f}")

features_dfs = [pd.DataFrame(tensor.numpy()) for tensor in features_list]
targets_dfs = [pd.DataFrame(tensor.cpu().detach().numpy()) for tensor in targets_list]

features_df = pd.concat(features_dfs, ignore_index=True)
targets_df = pd.concat(targets_dfs, ignore_index=True)

features_df.to_csv(data_dir_+"/tuned_brain_test.csv",index=False, header=False)
targets_df.to_csv(data_dir_+"/tuned_brain_test_labels.csv", index=False, header='Cell_Type')


