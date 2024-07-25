import sys
import os
os.environ["DGLBACKEND"] = "pytorch"

current_script_path = __file__
current_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
parent_parent = os.path.dirname(parent_dir)
data_dir_ = parent_parent+'/dance_data'

import torch
import torch.nn.functional as F
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, accuracy_score
from torchmetrics.classification import MulticlassAUROC

from dance.utils import set_seed
from pprint import pprint
from dance.datasets.singlemodality import ScDeepSortDataset

#ACTINN
from dance.modules.single_modality.cell_type_annotation.actinn import ACTINN

def custom_print(message, file=None):
    if file:
        with open(file, 'a') as f:
            f.write(message + '\n')
            
datasets = ['mouse_Brain', 'mouse_Kidney', 'human_Pancreas', 'human_Spleen', 'human_Bonemarrow']
for datasetname in datasets:
    
    if datasetname == 'mouse_Brain':
        dataset = ScDeepSortDataset(species="mouse", tissue="Brain",
                                train_dataset=["753", "3285"], test_dataset=["2695"], data_dir = data_dir_)
        
    elif datasetname == 'mouse_Kidney':
        dataset = ScDeepSortDataset(species="mouse", tissue="Kidney",
                                train_dataset=["4682"], test_dataset=["203"], data_dir = data_dir_)
        
    elif datasetname == 'human_Pancreas':
        dataset = ScDeepSortDataset(species="human", tissue="Pancreas",
                                train_dataset=["9727"], test_dataset=["2227", "1841"], data_dir = data_dir_)
        
    elif datasetname == 'human_Spleen':
        dataset = ScDeepSortDataset(species="human", tissue="Spleen",
                                train_dataset=["15806"], test_dataset=["9887"], data_dir = data_dir_)
        
    elif datasetname == 'human_Bonemarrow':
        dataset = ScDeepSortDataset(species="human", tissue="Bonemarrow",
                                train_dataset=["2261"], test_dataset=["6443"], data_dir = data_dir_)
    
    set_seed(42)
    print(torch.cuda.is_available())
    device=torch.device('cuda')
    model = ACTINN(lambd=0.01, device='cuda')

    preprocessing_pipeline = model.preprocessing_pipeline(normalize=True, filter_genes=True)

    data = dataset.load_data()
    preprocessing_pipeline(data)

    x_train, y_train = data.get_train_data(return_type="torch")
    x_test, y_test = data.get_test_data(return_type="torch")
    seed = 42
    y_train = torch.argmax(y_train, dim=1)
    y_test = torch.argmax(y_test, dim=1)

    print(torch.unique(y_train))
    print(torch.unique(y_test))

    y_train = y_train.to(device)
    y_test = y_test.to(device)
    num_classes = len(torch.unique(torch.cat([y_train, y_test], dim=0)))
    x_train = x_train.to(device)
    x_test = x_test.to(device)

    model.fit(x_train, y_train, lr=0.001, num_epochs=300,
              batch_size=1000, print_cost=True)
    pred = model.predict(x_test)
    acc = accuracy_score(y_test.cpu(), pred.cpu())

    auc = MulticlassAUROC(num_classes=num_classes)
    auc.update(F.softmax(model.model(x_test).cpu()).detach(), y_test.cpu())

    f1 = f1_score(y_test.cpu(), pred.cpu(), average='macro')
    precision = precision_score(y_test.cpu(), pred.cpu(), average='macro')
    recall = recall_score(y_test.cpu(), pred.cpu(), average='macro')

    cm = confusion_matrix(y_test.cpu(), pred.cpu())
    specificity = np.sum(np.diag(cm)) / np.sum(cm)

    file = 'results.txt'
    custom_print('\nACTINN '+datasetname, file)
    custom_print(f"ACC: {acc}", file)
    custom_print(f"Macro AUC: {auc.compute()}", file)
    custom_print(f"F1: {f1}", file)
    custom_print(f"Precision: {precision}", file)
    custom_print(f"Recall: {recall}", file)
    custom_print(f"Specificity: {specificity}", file)


