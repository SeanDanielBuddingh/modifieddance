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

from dance.modules.single_modality.cell_type_annotation.actinn import ACTINN

from dance.utils import set_seed

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from torcheval.metrics import MulticlassAUROC

from WordSageimport import WordSAGE

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 42
    set_seed(42)
    in_channels = 100
    hidden_channels = 100
    out_channels = 100
    num_classes = 16
    model = WordSAGE(in_channels, hidden_channels, out_channels, num_classes).to(device)
    train_inputs, test_inputs, train_targets, test_targets = WordSAGE.read_data(self=model, seed=seed, dataset=dataset)

    # concatenating either y_hat_markers or y_hard_markers to the input data
    #additional_vars = pd.read_csv(data_dir_+"/y_hat_train.csv", header=None)
    #train_inputs = pd.concat([train_inputs, additional_vars], axis=1)
    #additional_vars = pd.read_csv(data_dir_+"/y_hat_test.csv", header=None)
    #test_inputs = pd.concat([test_inputs, additional_vars], axis=1)

    train_targets = torch.tensor(train_targets[0].values, dtype=torch.long).to(device)
    test_targets = torch.tensor(test_targets[0].values, dtype=torch.long).to(device)
    #print(test_targets)
    num_classes = len(torch.unique(torch.cat([train_targets, test_targets], dim=0)))

    train_inputs = torch.tensor(train_inputs.to_numpy(), dtype=torch.float32).to(device)
    test_inputs = torch.tensor(test_inputs.to_numpy(), dtype=torch.float32).to(device)
    print(train_inputs.size())
    print(train_targets.size())

    #ACTINN
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))

    model = ACTINN(lambd=0.01, device='cuda')
    model.fit(train_inputs, train_targets, lr=0.001, num_epochs=300,
              batch_size=1000, print_cost=True)
    pred = model.predict(test_inputs)
    acc = accuracy_score(test_targets.cpu(), pred.cpu())
    print(set(test_targets.cpu().numpy()))
    print(F.softmax(model.model(test_inputs).cpu()).detach())

    auc = MulticlassAUROC(num_classes=num_classes)
    auc.update(F.softmax(model.model(test_inputs).cpu()).detach(), test_targets.cpu())
    f1 = f1_score(test_targets.cpu(), pred.cpu(), average='macro')
    precision = precision_score(test_targets.cpu(), pred.cpu(), average='macro')
    recall = recall_score(test_targets.cpu(), pred.cpu(), average='macro')

    # For specificity, calculate the confusion matrix and derive specificity
    cm = confusion_matrix(test_targets.cpu(), pred.cpu())
    specificity = np.sum(np.diag(cm)) / np.sum(cm)

    file = 'results.txt'
    custom_print('\nACTINN W2V '+dataset, file)
    custom_print(f"ACC: {acc}", file)
    custom_print(f"Macro AUC: {auc.compute()}", file)
    custom_print(f"F1: {f1}", file)
    custom_print(f"Precision: {precision}", file)
    custom_print(f"Recall: {recall}", file)
    custom_print(f"Specificity: {specificity}", file)

#print(model.model)