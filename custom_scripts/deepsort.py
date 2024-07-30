import sys
import os

current_script_path = __file__
current_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
parent_parent = os.path.dirname(parent_dir)
data_dir_ = parent_parent+'/dance_data'

import torch
import torch.nn.functional as F
import numpy as np

from torcheval.metrics import MulticlassAUROC
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from dance.modules.single_modality.cell_type_annotation.scdeepsort import ScDeepSort
from dance.utils import set_seed
from dance.datasets.singlemodality import ScDeepSortDataset

os.environ["DGLBACKEND"] = "pytorch"

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
        dataset = ScDeepSortDataset(species="human", tissue="Bone_marrow",
                                train_dataset=["2261"], test_dataset=["6443"], data_dir = data_dir_)
        
    train_losses = []
    train_accuracies = []

    in_channels = 400
    hidden_channels = 400
    out_channels = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(42)

    species, tissue = datasetname.split('_')
    #ScDeepSort
    model = ScDeepSort(dim_in=in_channels, dim_hid=hidden_channels, num_layers=1, species=species, tissue=tissue, device=torch.device('cuda'))
    data = dataset.load_data()

    pipeline = model.preprocessing_pipeline()
    pipeline(data)

    graph = data.data.uns['CellFeatureGraph']
    labels = torch.tensor(data.data.uns['TrainLabels'])
    test_labels = torch.tensor(data.data.uns['TestLabels'])
    print('\nHERE ',labels, labels.shape, test_labels, test_labels.shape)

    num_classes = len(torch.unique(torch.cat([labels, test_labels], dim=0).argmax(dim=1)))
    print(
        f"Number of classes: {num_classes}, Number of training samples: {labels.shape[0]}, Number of test samples: {test_labels.shape[0]}"
    )
    #break
    y_train = torch.argmax(labels, 1)
    y_test = torch.argmax(test_labels, 1)
    all_labels = torch.cat([y_train, y_test], dim=0)

    model.fit(graph=graph, labels=all_labels)
    train_losses = model.train_losses
    train_accuracies = model.train_accuracies

    with torch.no_grad():
        result = model.predict_proba(graph=data.data.uns["CellFeatureGraph"])

    result = result[-len(y_test):]
    result = torch.tensor(result)
    predicted = torch.argmax(result, 1)
    #torch.set_printoptions(profile="full")
    #print(predicted)
    correct = (predicted == y_test).sum().item()
    total = y_test.numel()
    accuracy = correct / total

    auc = MulticlassAUROC(num_classes=num_classes)
    auc.update(result.cpu(), y_test.cpu())
    f1 = f1_score(y_test.cpu(), predicted.cpu(), average='macro')
    precision = precision_score(y_test.cpu(), predicted.cpu(), average='macro')
    recall = recall_score(y_test.cpu(), predicted.cpu(), average='macro')

    # For specificity, calculate the confusion matrix and derive specificity
    cm = confusion_matrix(y_test.cpu(), predicted.cpu())
    specificity = np.sum(np.diag(cm)) / np.sum(cm)

    file = 'results.txt'
    custom_print('\nScDeepSort '+datasetname, file)
    custom_print(f"ACC: {acc}", file)
    custom_print(f"Macro AUC: {auc.compute()}", file)
    custom_print(f"F1: {f1}", file)
    custom_print(f"Precision: {precision}", file)
    custom_print(f"Recall: {recall}", file)
    custom_print(f"Specificity: {specificity}", file)


