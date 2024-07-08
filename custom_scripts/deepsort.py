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


train_losses = []
train_accuracies = []

in_channels = 400
hidden_channels = 400
out_channels = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(42)


#ScDeepSort
model = ScDeepSort(dim_in=in_channels, dim_hid=hidden_channels, num_layers=1, species='mouse', tissue='Brain', device=torch.device('cuda'))
dataset = ScDeepSortDataset(species="mouse", tissue="Brain",
                            train_dataset=["753", "3285"], test_dataset=["2695"], data_dir = data_dir_)
data = dataset.load_data()
pipeline = model.preprocessing_pipeline()
pipeline(data)

graph = data.get_train_graph()
print('\n',graph,'\n')
gene_mask = graph.ndata["cell_id"] != -1
cell_mask = graph.ndata["cell_id"] == -1
num_genes = gene_mask.sum()
num_cells = cell_mask.sum()

print(f"Number of genes: {num_genes}")
print(f"Number of cells: {num_cells}")

y_train = data.get_train_data(return_type="torch")[1]
y_test = data.get_test_data(return_type="torch")[1]
num_classes = len(torch.unique(torch.cat([y_train, y_test], dim=0)))
y_train = torch.argmax(y_train, 1)
y_test = torch.argmax(y_test, 1)

model.fit(graph=2, labels=y_train)
train_losses = model.train_losses
train_accuracies = model.train_accuracies

# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.plot(range(1, 300 + 1), train_losses, label='Train Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training Loss')
# plt.legend()

# # Plotting the training accuracy
# plt.subplot(1, 2, 2)
# plt.plot(range(1, 300 + 1), train_accuracies, label='Train Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title('Training Accuracy')
# plt.legend()

# plt.tight_layout()
# plt.show()

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

print(f"ACC: {accuracy}")
print(f"Macro AUC: {auc.compute()}")
print(f"F1: {f1}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Specificity: {specificity}")

