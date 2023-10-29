import os
os.environ["DGLBACKEND"] = "pytorch"
from pprint import pprint
from dance.datasets.singlemodality import ScDeepSortDataset

dataset = ScDeepSortDataset(full_download=True, species="mouse", tissue="Brain",
                            train_dataset=["3285", "753"], test_dataset=["2695"])
data = dataset.load_data()