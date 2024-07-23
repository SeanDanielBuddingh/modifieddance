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
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, accuracy_scores, MulticlassAUROC

from dance.utils import set_seed
from pprint import pprint
from dance.datasets.singlemodality import ScDeepSortDataset

#ACTINN
from dance.modules.single_modality.cell_type_annotation.actinn import ACTINN

set_seed(42)
print(torch.cuda.is_available())
device=torch.device('cpu')
model = ACTINN(lambd=0.01, device='cpu')

preprocessing_pipeline = model.preprocessing_pipeline(normalize=True, filter_genes=True)
dataset = ScDeepSortDataset(species="human", tissue="Pancreas",
                            train_dataset=["9727"], test_dataset=["2227", "1841"], data_dir = data_dir_)
data = dataset.load_data()
preprocessing_pipeline(data)

# x_train, y_train = data.get_train_data(return_type="torch")
# x_test, y_test = data.get_test_data(return_type="torch")
# seed = 42
# y_train = torch.argmax(y_train, dim=1)
# y_test = torch.argmax(y_test, dim=1)

# print(torch.unique(y_train))
# print(torch.unique(y_test))

# combined = torch.cat((x_train, x_test), dim=0)
# y_comb = torch.cat((y_train, y_test), dim=0)
# import pandas as pd
# combined = pd.DataFrame(combined)
# y_comb = pd.DataFrame(y_comb)
# inputs, targets = mix_data(seed, combined, y_comb)

# x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.2, random_state=seed, stratify=targets)

# y_train = torch.tensor(y_train[0].values, dtype=torch.long).to(device)
# y_test = torch.tensor(y_test[0].values, dtype=torch.long).to(device)
# num_classes = len(torch.unique(torch.cat([y_train, y_test], dim=0)))
# x_train = torch.tensor(x_train.to_numpy(), dtype=torch.float32).to(device)
# x_test = torch.tensor(x_test.to_numpy(), dtype=torch.float32).to(device)

model.fit(x_train, y_train, lr=0.001, num_epochs=300,
          batch_size=1000, print_cost=True)
pred = model.predict(x_test)
acc = accuracy_scores(y_test.cpu(), pred.cpu())

#torch.set_printoptions(profile="full")
#print(F.softmax(model.model(x_test).cpu()).detach())

auc = MulticlassAUROC(num_classes=num_classes)
auc.update(F.softmax(model.model(x_test).cpu()).detach(), y_test.cpu())

#macro_auc = roc_auc_score(F.one_hot(y_test, num_classes=num_classes).cpu().numpy(), F.softmax(model.model(x_test).cpu()).detach().numpy(), multi_class='ovo', average='macro')
f1 = f1_score(y_test.cpu(), pred.cpu(), average='macro')
precision = precision_score(y_test.cpu(), pred.cpu(), average='macro')
recall = recall_score(y_test.cpu(), pred.cpu(), average='macro')

# For specificity, calculate the confusion matrix and derive specificity
cm = confusion_matrix(y_test.cpu(), pred.cpu())
specificity = np.sum(np.diag(cm)) / np.sum(cm)

print(f"ACC: {acc}")
print(f"Macro AUC: {auc.compute()}")
print(f"F1: {f1}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Specificity: {specificity}")

