import os
os.environ["DGLBACKEND"] = "pytorch"

import sys
sys.path.append("..")

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

from dance.utils import set_seed

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from torcheval.metrics import MulticlassAUROC

from WordSage import WordSAGE

#ScDeepSort
from dance.modules.single_modality.cell_type_annotation.scdeepsort import ScDeepSort

def custom_print(message, file=None):
    if file:
        with open(file, 'a') as f:
            f.write(message + '\n')

datasets = ['mouse_Brain', 'mouse_Kidney', 'human_Pancreas', 'human_Spleen', 'human_Bonemarrow', 
            'brain_hard', 'brain_soft', 'tuned_brain_hard', 'tuned_brain_soft',
            'kidney_hard', 'kidney_soft', 'tuned_kidney_hard', 'tuned_kidney_soft',
            'pancreas_hard', 'pancreas_soft', 'tuned_pancreas_hard', 'tuned_pancreas_soft',
            'human_spleen_hard', 'human_spleen_soft', 'tuned_human_spleen_hard', 'tuned_human_spleen_soft',
            'bone_hard', 'bone_soft', 'tuned_bone_hard', 'tuned_bone_soft']
for dataset in datasets:

    in_channels = 2500
    hidden_channels = 2500
    out_channels = 100
    num_classes = 20
    dim_tuple = 0,0
    WordSage = WordSAGE(dim_tuple, hidden_channels, out_channels, num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 42
    set_seed(42)
    train_graph, train_targets, test_graph, test_targets, train_nodes, test_nodes = WordSage.read_data(seed, dataset)
    train_targets = torch.tensor(train_targets[0].values, dtype=torch.long).to(device)
    test_targets = torch.tensor(test_targets[0].values, dtype=torch.long).to(device)
    num_classes = len(torch.unique(torch.cat([train_targets, test_targets], dim=0)))

    #ScDeepSort
    in_channels = train_graph.nodes['gene_node'].data['features'].shape[1] 
    hidden_channels = in_channels
    model = ScDeepSort(dim_in=in_channels, dim_hid=hidden_channels, num_layers=1, species='mouse', tissue='Brain', device=device)

    model.fit(graph=train_graph, labels=train_targets)
    prob = model.predict_proba(graph=test_graph)
    result = torch.tensor(prob)
    pred = torch.argmax(result, 1)

    acc = accuracy_score(test_targets.cpu(), pred.cpu())

    auc = MulticlassAUROC(num_classes=num_classes)
    auc.update(torch.as_tensor(prob).cpu(), test_targets.cpu())
    f1 = f1_score(test_targets.cpu(), pred.cpu(), average='macro')
    precision = precision_score(test_targets.cpu(), pred.cpu(), average='macro')
    recall = recall_score(test_targets.cpu(), pred.cpu(), average='macro')

    # For specificity, calculate the confusion matrix and derive specificity
    cm = confusion_matrix(test_targets.cpu(), pred.cpu())
    specificity = np.sum(np.diag(cm)) / np.sum(cm)

    file = 'results.txt'
    custom_print('ScDeepSort W2V '+dataset, file)
    custom_print(f"ACC: {acc}", file)
    custom_print(f"Macro AUC: {auc.compute()}", file)
    custom_print(f"F1: {f1}", file)
    custom_print(f"Precision: {precision}", file)
    custom_print(f"Recall: {recall}", file)
    custom_print(f"Specificity: {specificity}", file)