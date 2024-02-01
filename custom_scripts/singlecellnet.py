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

def mix_data(seed, inputs, targets):
    np.random.seed(seed)
    
    combined = pd.concat([inputs, targets], axis=1)
    
    combined_shuffled = combined.sample(frac=1).reset_index(drop=True)

    num_input_columns = inputs.shape[1]
    inputs_shuffled = combined_shuffled.iloc[:, :num_input_columns]
    targets_shuffled = combined_shuffled.iloc[:, num_input_columns:]

    return inputs_shuffled, targets_shuffled

model = SingleCellNet(num_trees=100)
device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
seed = 42
set_seed(42)
preprocessing_pipeline = model.preprocessing_pipeline()
dataset = ScDeepSortDataset(species="mouse", tissue="Kidney",
                            train_dataset=["4682"], test_dataset=["203"], data_dir = data_dir_)
data = dataset.load_data()
preprocessing_pipeline(data)

x_train, y_train = data.get_train_data(return_type="torch")
x_test, y_test = data.get_test_data(return_type="torch")
seed = 42
y_train = torch.argmax(y_train, dim=1)
y_test = torch.argmax(y_test, dim=1)

combined = torch.cat((x_train, x_test), dim=0)
y_comb = torch.cat((y_train, y_test), dim=0)
import pandas as pd
combined = pd.DataFrame(combined)
y_comb = pd.DataFrame(y_comb)

inputs, targets = mix_data(seed, combined, y_comb)
x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.2, random_state=seed, stratify=targets)
y_train = torch.tensor(y_train[0].values, dtype=torch.long).to(device)
y_test = torch.tensor(y_test[0].values, dtype=torch.long).to(device)

x_train = torch.tensor(x_train.to_numpy(), dtype=torch.float32).to(device)
x_test = torch.tensor(x_test.to_numpy(), dtype=torch.float32).to(device)

set_seed(42)

x_train.to(device)
y_train.to(device)
x_test.to(device)
y_test.to(device)
print(y_test.shape)

model.fit(x_train.numpy(), y_train.numpy())

pred = model.predict(x_test.numpy()) 
probs = model.predict_proba(x_test.numpy())

acc = accuracy_score(y_test.numpy(), pred)

auc = MulticlassAUROC(num_classes=20)
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
