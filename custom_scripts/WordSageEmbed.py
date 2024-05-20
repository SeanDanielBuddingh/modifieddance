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
    def __init__(self, dim_tuple, hidden_channels, out_channels, num_classes, num_binary_targets):
        super(WordSAGE, self).__init__()
        self.seed = 42
        src_dim, dst_dim = dim_tuple
        self.dst_dim = dst_dim

        # Block 1
        self.self_attention = torch.nn.MultiheadAttention(hidden_channels, num_heads=1)
        self.conv1 = dgl.nn.HeteroGraphConv({
            'connects': dgl.nn.SAGEConv((src_dim, dst_dim), hidden_channels, 'mean'),
        }, aggregate='sum')
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.ln1 = torch.nn.LayerNorm(hidden_channels)
        self.conv2 = dgl.nn.HeteroGraphConv({
            'connects': dgl.nn.SAGEConv(hidden_channels, out_channels, 'mean'),
        }, aggregate='sum')
        self.bn2 = torch.nn.BatchNorm1d(out_channels)

        self.linear = torch.nn.Linear(out_channels, out_channels)
        self.bce = torch.nn.Linear(out_channels, num_binary_targets)

    def forward(self, x, features):

        # Block 1
        h = self.conv1(x, features)
        h = torch.cat([v for v in h.values()], dim=0)
        h = self.bn1(h)
        h = F.leaky_relu(h)
        h, weights = self.self_attention(h,h,h)
        h = self.ln1(h)
        h = F.leaky_relu(h)
        h = F.dropout(h, p=0.2, training=self.training)
        h = self.conv2(x, {'gene_node': features['gene_node'], 'train_node': h})
        h = torch.cat([v for v in h.values()], dim=0)
        h = self.bn2(h)
        y_hat = F.leaky_relu(h)
        # Block 1 Decoder
        h = self.linear(y_hat)
        h = F.relu(h)
        b = self.bce(h)

        return y_hat, b
        
class WordSAGEBLOCK2(torch.nn.Module):
    def __init__(self, dim_tuple, hidden_channels, out_channels, num_classes, num_binary_targets):
        super(WordSAGEBLOCK2, self).__init__()
        self.seed = 42
        src_dim, dst_dim = dim_tuple
        self.dst_dim = 2675

        # Block 2
        self.self_attention2 = torch.nn.MultiheadAttention(hidden_channels, num_heads=1)
        self.conv3 = dgl.nn.HeteroGraphConv({
            'connects': dgl.nn.SAGEConv((src_dim, self.dst_dim), hidden_channels, 'mean'),
        }, aggregate='sum')
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)
        self.ln2 = torch.nn.LayerNorm(hidden_channels)
        self.conv4 = dgl.nn.HeteroGraphConv({
            'connects': dgl.nn.SAGEConv(hidden_channels, out_channels, 'mean'),
        }, aggregate='sum')
        self.bn4 = torch.nn.BatchNorm1d(out_channels)
        self.linear2 = torch.nn.Linear(out_channels, out_channels)
        self.ce = torch.nn.Linear(out_channels, num_classes)

    def forward(self, x, features, y_hat):

        # Concatenate y_hat to the features
        features['train_node'] = torch.cat([features['train_node'], y_hat], dim=1)
        self.dst_dim = len(features['train_node'][0])

        # Block 2
        h = self.conv3(x, features)
        h = torch.cat([v for v in h.values()], dim=0)
        h = self.bn3(h)
        h = F.leaky_relu(h)
        h, weights = self.self_attention2(h,h,h)
        h = self.ln2(h)
        h = F.leaky_relu(h)
        h = F.dropout(h, p=0.2, training=self.training)
        h = self.conv4(x, {'gene_node': features['gene_node'], 'train_node': h})
        h = torch.cat([v for v in h.values()], dim=0)
        h = self.bn4(h)
        x_hat = F.leaky_relu(h)
        # Block 2 Decoder
        h = self.linear2(x_hat)
        h = F.relu(h)
        y = self.ce(h)

        return x_hat, y


def read_data(batch_size, seed):
    data = data_pre()
    tissue_train, tissue_test, genes, y_values_train, y_values_test, normalized_train, normalized_test, train_raw, test_raw = data.read_w2v()

    # Raw Data Handling for comparison in the specific gene marker identification phase (in differential.py)
    train_genes = set(train_raw.index)
    test_genes = set(test_raw.index)
    common_genes = train_genes.intersection(test_genes)
    combined_brain = pd.concat([train_raw, test_raw], axis=1, ignore_index=False).fillna(0).loc[list(common_genes)]
    combined_brain = combined_brain.T.reset_index(drop=True)

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

    genemarkers = GeneMarkers()
    if not os.path.exists(data_dir_+"/ft_y_train.csv"):
        full_list_train, full_list_test, _ = genemarkers.ConstructTargets(y_values_train[0], y_values_test[0], normalized_train, normalized_test, combined_brain, sublist_length=10)
    else:
        full_list_train = pd.read_csv(data_dir_+"/ft_y_train.csv", header=None, index_col=None)
        full_list_test = pd.read_csv(data_dir_+"/ft_y_test.csv", header=None, index_col=None)

    inputs_train, bce_targets_train_list, targets_train_list = mix_data(seed, tissue_train, full_list_train, targets_encoded_train, batch_size)
    inputs_test, bce_targets_test_list, targets_test_list = mix_data(seed, tissue_test, full_list_test, targets_encoded_test, batch_size)

    train_graphs, train_nodes_list = [], []
    for batch in inputs_train:
        train_graph, train_nodes = basic_dgl_graph(batch, genes, normalized_train)
        train_graphs.append(train_graph)
        train_nodes_list.append(train_nodes)

    test_graphs, test_nodes_list = [], []
    for batch in inputs_test:
        test_graph, test_nodes = basic_dgl_graph(batch, genes, normalized_test)
        test_graphs.append(test_graph)
        test_nodes_list.append(test_nodes)

    return train_graphs, bce_targets_train_list, targets_train_list, test_graphs, bce_targets_test_list, targets_test_list, train_nodes_list, test_nodes_list

def basic_dgl_graph(train_inputs, genes, normalized):
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
    G = dgl.heterograph(graph_data, num_nodes_dict={'gene_node': num_gene_nodes, 'train_node': num_train_nodes})

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

def mix_data(seed, inputs, bce_targets, ce_targets, batch_size):

    np.random.seed(seed)
    print('\nMixing Data\n')
    # Combine inputs and targets
    combined = pd.concat([inputs, bce_targets], axis=1)
    print(combined.shape)
    combined = pd.concat([combined, ce_targets], axis=1)
    print(combined.shape,'\n')
    # Shuffle the combined DataFrame - temporarily disabled
    combined_shuffled = combined #combined.sample(frac=1).reset_index(drop=True)

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

    inputs_shuffled = batch_data(inputs_shuffled, batch_size)
    bce_targets_shuffled = batch_data(bce_targets_shuffled, batch_size)
    ce_targets_shuffled = batch_data(ce_targets_shuffled, batch_size)

    return inputs_shuffled, bce_targets_shuffled, ce_targets_shuffled

def batch_data(data, batch_size):
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
batch_size = 32
train_graphs, bce_targets_train_list, targets_train_list, test_graphs, bce_targets_test_list, targets_test_list, train_nodes_list, test_nodes_list = read_data(batch_size, seed=seed)
src_dim = train_graphs[0].nodes['gene_node'].data['features'].shape[1] 
dst_dim = 2500
hidden_channels = 2500
out_channels = 2500
num_classes = 16
num_binary_targets = len(bce_targets_train_list[0][0][0])
print('\nnum_binary_targets: ', num_binary_targets)
lr = 1e-2
momentum = 0.9

set_seed(seed)

block1 = WordSAGE((src_dim, dst_dim), hidden_channels, out_channels, num_classes, num_binary_targets).to(device)
block2 = WordSAGEBLOCK2((src_dim, dst_dim), hidden_channels, out_channels, num_classes, num_binary_targets).to(device)

bce_loss = torch.nn.BCEWithLogitsLoss()
ce_loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(block1.parameters(), lr=lr, momentum=momentum)

if not os.path.exists(data_dir_+"/y_hat_train.csv"):
    y_hat_list = []
    for epoch in range(10):
        features_list = []
        targets_list = []
        count=0
        bcc_denom = 0
        for i, (train_graph, bce_targets_train, train_targets, train_nodes) in enumerate(zip(train_graphs, bce_targets_train_list, targets_train_list, train_nodes_list)):
        
            train_graph = train_graph.to(device)
            train_features = train_graph.nodes['train_node'].data['features']
            gene_features = train_graph.nodes['gene_node'].data['features']
            train_feature_map = {'gene_node': gene_features, 'train_node': train_features}

            train_targets = torch.tensor(train_targets.values, dtype=torch.long).to(device)
            bce_train_targets = torch.tensor(bce_targets_train.values.tolist()).to(device)

            optimizer.zero_grad()
            bce_feature, bce_cells = block1(train_graph, train_feature_map)

            bce_loss_value = bce_loss(bce_cells, bce_train_targets.squeeze(1))
            for feat_out, target_out in zip(bce_cells, bce_train_targets.squeeze(1)):
                for logit, correct in zip(feat_out, target_out):
                    logit = torch.sigmoid(logit)
                    if logit >= 0.5:
                        logit = 1
                    else:
                        logit = 0
                    if logit == correct:
                        count+=1
            bcc_denom += (len(target_out)*len(bce_cells))
            bce_acc= count/bcc_denom
            bce_loss_value.backward()
            optimizer.step()
            if epoch == 9:
                y_hat_list.append(bce_cells)

        if (epoch+1) % 1 == 0:
            print(f"\n[UPDATE TRAIN SET BLOCK 1] [EPOCH {epoch + 1}] BCE Loss: {bce_loss_value:.4f} | BCE_Acc: {bce_acc:.4f}")

    y_hat_dfs = [pd.DataFrame(tensor.cpu().detach().squeeze(1).numpy()) for tensor in y_hat_list]
    y_hat_df = pd.concat(y_hat_dfs, ignore_index=True)
    y_hat_df.to_csv(data_dir_+"/y_hat_train.csv", index=False, header=False)

else:
    y_hat_df = pd.read_csv(data_dir_+"/y_hat_train.csv", header=None)
    y_hat_list = [torch.tensor(y_hat_df.iloc[i*batch_size:min((i+1)*batch_size, len(y_hat_df))].values, dtype=torch.float32).to(device) for i in range(len(y_hat_df)//batch_size + 1)]


# Reinitialize optimizer for block 2
y_hat_list = [tensor.detach() for tensor in y_hat_list]
del bce_loss, ce_loss, optimizer
torch.cuda.empty_cache() 
gc.collect()
bce_loss = torch.nn.BCEWithLogitsLoss()
ce_loss = torch.nn.CrossEntropyLoss() 
optimizer = torch.optim.SGD(block2.parameters(), lr=lr, momentum=momentum)

for epoch in range(50):
    features_list = []
    targets_list = []
    correct_predictions = 0
    total_predictions = 0
    for i, (train_graph, bce_targets_train, train_targets, train_nodes) in enumerate(zip(train_graphs, bce_targets_train_list, targets_train_list, train_nodes_list)):

        train_graph = train_graph.to(device)
        train_features = train_graph.nodes['train_node'].data['features']
        gene_features = train_graph.nodes['gene_node'].data['features']
        train_feature_map = {'gene_node': gene_features, 'train_node': train_features}

        train_targets = torch.tensor(train_targets.values, dtype=torch.long).to(device)
        bce_train_targets = torch.tensor(bce_targets_train.values.tolist()).to(device)

        optimizer.zero_grad()
        feature, out_cells = block2(train_graph, train_feature_map, y_hat_list[i])
        ce_loss_value = ce_loss(out_cells, train_targets.squeeze(1))
        for feat_out, target_out in zip(out_cells, train_targets.squeeze(1)):
            feat_out = F.softmax(feat_out, dim=0)
            feat_out = torch.argmax(feat_out)
            correct_predictions += (target_out.detach().cpu().numpy() == feat_out.detach().cpu().numpy())
            total_predictions += 1
        ce_acc = correct_predictions / total_predictions
        ce_loss_value.backward()
        optimizer.step()
        features_list.append(feature.cpu().detach()[(range(train_nodes))])
        targets_list.append(train_targets)


    if (epoch+1) % 1 == 0:
        print(f"\n[UPDATE TRAIN SET BLOCK 2] [EPOCH {epoch + 1}] CE Loss: {ce_loss_value:.4f} | CE_Acc: {ce_acc:.4f}")

features_dfs = [pd.DataFrame(tensor.numpy()) for tensor in features_list]
targets_dfs = [pd.DataFrame(tensor.cpu().detach().numpy()) for tensor in targets_list]

features_df = pd.concat(features_dfs, ignore_index=True)
targets_df = pd.concat(targets_dfs, ignore_index=True)

features_df.to_csv(data_dir_+"/tuned_brain_train.csv",index=False, header=False)
targets_df.to_csv(data_dir_+"/tuned_brain_train_labels.csv", index=False, header='Cell_Type')

# Reinitialize optimizer for block 1
del bce_loss, ce_loss, optimizer
torch.cuda.empty_cache() 
gc.collect()
bce_loss = torch.nn.BCEWithLogitsLoss()
ce_loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(block1.parameters(), lr=lr, momentum=momentum)

if not os.path.exists(data_dir_+"/y_hat_test.csv"):
    y_hat_list = []
    for epoch in range(10):
        features_list = []
        targets_list = []
        count=0
        bcc_denom = 0
        for i, (test_graph, bce_targets_test, test_targets, test_nodes) in enumerate(zip(test_graphs, bce_targets_test_list, targets_test_list, test_nodes_list)):
        
            test_graph = test_graph.to(device)
            test_features = test_graph.nodes['train_node'].data['features']
            gene_features = test_graph.nodes['gene_node'].data['features']
            test_feature_map = {'gene_node': gene_features, 'train_node': test_features}

            test_targets = torch.tensor(test_targets.values, dtype=torch.long).to(device)
            bce_test_targets = torch.tensor(bce_targets_test.values.tolist()).to(device)

            optimizer.zero_grad()
            bce_feature, bce_cells = block1(test_graph, test_feature_map)

            bce_loss_value = bce_loss(bce_cells, bce_test_targets.squeeze(1))
            for feat_out, target_out in zip(bce_cells, bce_test_targets.squeeze(1)):
                for logit, correct in zip(feat_out, target_out):
                    logit = torch.sigmoid(logit)
                    if logit >= 0.5:
                        logit = 1
                    else:
                        logit = 0
                    if logit == correct:
                        count+=1
            bcc_denom += (len(target_out)*len(bce_cells))
            bce_acc= count/bcc_denom
            bce_loss_value.backward()
            optimizer.step()
            if epoch == 9:
                y_hat_list.append(bce_cells)

        if (epoch+1) % 1 == 0:
            print(f"\n[UPDATE TEST SET BLOCK 1] [EPOCH {epoch + 1}] BCE Loss: {bce_loss_value:.4f} | BCE_Acc: {bce_acc:.4f}")

    y_hat_dfs = [pd.DataFrame(tensor.cpu().detach().squeeze(1).numpy()) for tensor in y_hat_list]
    y_hat_df = pd.concat(y_hat_dfs, ignore_index=True)
    y_hat_df.to_csv(data_dir_+"/y_hat_test.csv", index=False, header=False)

else:
    y_hat_df = pd.read_csv(data_dir_+"/y_hat_test.csv", header=None)
    y_hat_list = [torch.tensor(y_hat_df.iloc[i*batch_size:min((i+1)*batch_size, len(y_hat_df))].values, dtype=torch.float32).to(device) for i in range(len(y_hat_df)//batch_size + 1)]


# Reinitialize optimizer for block 2
y_hat_list = [tensor.detach() for tensor in y_hat_list]
del bce_loss, ce_loss, optimizer
torch.cuda.empty_cache() 
gc.collect()
bce_loss = torch.nn.BCEWithLogitsLoss()
ce_loss = torch.nn.CrossEntropyLoss()     
optimizer = torch.optim.SGD(block2.parameters(), lr=lr, momentum=momentum)

for epoch in range(50):
    features_list = []
    targets_list = []
    correct_predictions = 0
    total_predictions = 0
    for i, (test_graph, bce_targets_test, test_targets, test_nodes) in enumerate(zip(test_graphs, bce_targets_test_list, targets_test_list, test_nodes_list)):

        test_graph = test_graph.to(device)
        test_features = test_graph.nodes['train_node'].data['features']
        gene_features = test_graph.nodes['gene_node'].data['features']
        test_feature_map = {'gene_node': gene_features, 'train_node': test_features}

        test_targets = torch.tensor(test_targets.values, dtype=torch.long).to(device)
        bce_test_targets = torch.tensor(bce_targets_test.values.tolist()).to(device)

        optimizer.zero_grad()
        feature, out_cells = block2(test_graph, test_feature_map, y_hat_list[i])
        ce_loss_value = ce_loss(out_cells, test_targets.squeeze(1))
        for feat_out, target_out in zip(out_cells, test_targets.squeeze(1)):
            feat_out = F.softmax(feat_out, dim=0)
            feat_out = torch.argmax(feat_out)
            correct_predictions += (target_out.detach().cpu().numpy() == feat_out.detach().cpu().numpy())
            total_predictions += 1
        ce_acc = correct_predictions / total_predictions
        ce_loss_value.backward()
        optimizer.step()
        features_list.append(feature.cpu().detach()[(range(test_nodes))])
        targets_list.append(test_targets)


    if (epoch+1) % 1 == 0:
        print(f"\n[UPDATE TEST SET BLOCK 2] [EPOCH {epoch + 1}] CE Loss: {ce_loss_value:.4f} | CE_Acc: {ce_acc:.4f}")

features_dfs = [pd.DataFrame(tensor.numpy()) for tensor in features_list]
targets_dfs = [pd.DataFrame(tensor.cpu().detach().numpy()) for tensor in targets_list]

features_df = pd.concat(features_dfs, ignore_index=True)
targets_df = pd.concat(targets_dfs, ignore_index=True)

features_df.to_csv(data_dir_+"/tuned_brain_test.csv",index=False, header=False)
targets_df.to_csv(data_dir_+"/tuned_brain_test_labels.csv", index=False, header='Cell_Type')


