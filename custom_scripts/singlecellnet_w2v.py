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

from WordSageimport import WordSAGE

def mix_data(seed, inputs, targets):
    np.random.seed(seed)
    
    combined = pd.concat([inputs, targets], axis=1)
    
    combined_shuffled = combined.sample(frac=1).reset_index(drop=True)

    num_input_columns = inputs.shape[1]
    inputs_shuffled = combined_shuffled.iloc[:, :num_input_columns]
    targets_shuffled = combined_shuffled.iloc[:, num_input_columns:]

    return inputs_shuffled, targets_shuffled

device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
seed = 42
set_seed(42)
in_channels = 100
hidden_channels = 100
out_channels = 100
num_classes = 21
model = WordSAGE(in_channels, hidden_channels, out_channels, num_classes).to(device)
x_train, x_test, y_train, y_test = WordSAGE.read_data(self=model, seed=seed)

y_train = torch.tensor(y_train[0].values, dtype=torch.long).to(device)
y_test = torch.tensor(y_test[0].values, dtype=torch.long).to(device)

x_train = torch.tensor(x_train.to_numpy(), dtype=torch.float32).to(device)
x_test = torch.tensor(x_test.to_numpy(), dtype=torch.float32).to(device)
model = SingleCellNet(num_trees=100)
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
print(torch.unique(y_test))
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
