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

from WordSageimport import WordSAGE

#Celltypist
from dance.modules.single_modality.cell_type_annotation.celltypist import Celltypist, Classifier

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

    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    seed = 42
    set_seed(42)
    in_channels = 100
    hidden_channels = 100
    out_channels = 100
    num_classes = 21
    model = WordSAGE(in_channels, hidden_channels, out_channels, num_classes).to(device)
    x_train, x_test, y_train, y_test = WordSAGE.read_data(self=model, seed=seed, dataset=dataset)

    y_train = torch.tensor(y_train[0].values, dtype=torch.long).to(device)
    y_test = torch.tensor(y_test[0].values, dtype=torch.long).to(device)
    num_classes = len(torch.unique(torch.cat([y_train, y_test], dim=0)))
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

    file = 'results.txt'
    custom_print('\nCellTypist W2V '+dataset, file)
    custom_print(f"ACC: {acc}", file)
    custom_print(f"Macro AUC: {auc.compute()}", file)
    custom_print(f"F1: {f1}", file)
    custom_print(f"Precision: {precision}", file)
    custom_print(f"Recall: {recall}", file)
    custom_print(f"Specificity: {specificity}", file)

