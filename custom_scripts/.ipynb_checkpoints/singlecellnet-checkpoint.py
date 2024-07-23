import sys
import os

current_script_path = __file__
current_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
parent_parent = os.path.dirname(parent_dir)
data_dir_ = parent_parent+'/dance_data'

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import accuracy_score

from torcheval.metrics import MulticlassAUROC

from dance.datasets.singlemodality import ScDeepSortDataset
from dance.transforms import AnnDataTransform, SCNFeature
from dance.modules.single_modality.cell_type_annotation import SingleCellNet
from dance.utils import set_seed
import numpy as np
import torch
import torch.nn.functional as F


model = SingleCellNet(num_trees=100)
device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
seed = 42
set_seed(42)
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

set_seed(42)

x_train.to(device)
y_train.to(device)
x_test.to(device)
y_test.to(device)

model.fit(x_train.numpy(), y_train.numpy())

pred = model.predict(x_test.numpy()) 
probs = model.predict_proba(x_test.numpy())

acc = accuracy_score(y_test.numpy(), pred)

auc = MulticlassAUROC(num_classes=num_classes)
auc.update(torch.as_tensor(probs).cpu(), y_test.cpu())
f1 = f1_score(y_test, pred, average='macro')
precision = precision_score(y_test, pred, average='macro')
recall = recall_score(y_test, pred, average='macro')

# For specificity, calculate the confusion matrix and derive specificity
cm = confusion_matrix(y_test, pred)
specificity = np.sum(np.diag(cm)) / np.sum(cm)

print(f"ACC: {acc}")
print(f"Macro AUC: {auc.compute()}")
print(f"F1: {f1}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Specificity: {specificity}")
