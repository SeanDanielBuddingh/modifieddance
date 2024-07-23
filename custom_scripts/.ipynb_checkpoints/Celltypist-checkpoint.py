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

#Celltypist
from dance.modules.single_modality.cell_type_annotation.celltypist import Celltypist, Classifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 42
set_seed(42)

#Celltypist
model = Celltypist(majority_voting = False)
preprocessing_pipeline = model.preprocessing_pipeline()
dataset = ScDeepSortDataset(species="mouse", tissue="Brain",
                            train_dataset=["753", "3285"], test_dataset=["2695"], data_dir = data_dir_)
data = dataset.load_data()
preprocessing_pipeline(data)

x_train, y_train = data.get_train_data(return_type="torch")
x_test, y_test = data.get_test_data(return_type="torch")
seed = 42
y_train = torch.argmax(y_train, dim=1)
y_test = torch.argmax(y_test, dim=1)
num_classes = len(torch.unique(torch.cat([y_train, y_test], dim=0)))

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

print(f"ACC: {acc}")
print(f"Macro AUC: {auc.compute()}")
print(f"F1: {f1}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Specificity: {specificity}")

