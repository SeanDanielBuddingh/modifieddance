from dance.datasets.singlemodality import ScDeepSortDataset
from dance.transforms import AnnDataTransform, SCNFeature
from dance.modules.single_modality.cell_type_annotation import SingleCellNet
from dance.utils import set_seed
import numpy as np
import torch

model = SingleCellNet(num_trees=100)
set_seed(42)
preprocessing_pipeline = model.preprocessing_pipeline()
dataset = ScDeepSortDataset(species="mouse", tissue="Brain",
                            train_dataset=["753", "3285"], test_dataset=["2695"])
data = dataset.load_data()
preprocessing_pipeline(data)
x_train, y_train = data.get_train_data(return_type="torch")
x_train = x_train.numpy()
y_train = torch.argmax(y_train, 1).numpy()
x_test, y_test = data.get_test_data(return_type="torch")
x_test = x_test.numpy()
y_test = torch.argmax(y_test, 1).numpy()

model.fit(x_train, y_train)

#val_score = model.score(x_test, y_test)
#print(f"Validation Score: {val_score}")

result = model.predict(x_test)
result = torch.tensor(result)
predicted = result#torch.argmax(result, 1)
print(predicted)
y_test = torch.tensor(y_test)
correct = (predicted == y_test).sum().item()
total = y_test.numel()
accuracy = correct / total
print('accuracy: ', accuracy)
