import os
import sys

os.environ["DGLBACKEND"] = "pytorch"

current_script_path = __file__
current_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
parent_parent = os.path.dirname(parent_dir)
data_dir_ = parent_parent+'/dance_data'

from pprint import pprint
from dance.datasets.singlemodality import ScDeepSortDataset

dataset = ScDeepSortDataset(full_download=True, species="mouse", tissue="Brain",
                            train_dataset=["3285", "753"], test_dataset=["2695"])
data = dataset.load_data()