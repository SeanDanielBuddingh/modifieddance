import numpy as np
import pandas as pd
import scanpy as sc

import sys
import os

current_script_path = __file__
current_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
parent_parent = os.path.dirname(parent_dir)
data_dir_ = parent_parent+'/dance_data'


def read_w2v():
    y_values_train = pd.read_csv(data_dir_.path+'/brain_train_labels.csv', skiprows=1, header=None, dtype=str)
    y_values_test = pd.read_csv(data_dir_.path+'/brain_test_labels.csv', skiprows=1, header=None, dtype=str)
    normalized_train = pd.read_csv(data_dir_.path+'/normalized_brain_train.csv', header=0, index_col=0)
    normalized_test = pd.read_csv(data_dir_.path+'/normalized_brain_test.csv', header=0, index_col=0)

    return y_values_train, y_values_test, normalized_train, normalized_test

y_values_train, y_values_test, normalized_train, normalized_test = read_w2v()

adata1 = sc.AnnData(X=normalized_train, obs=dict(condition=y_values_train))
adata2 = sc.AnnData(X=normalized_test, obs=dict(condition=y_values_test))
adata = sc.concat([adata1, adata2], join="outer")

sc.tl.rank_genes_groups(adata, groupby='condition', test_stat='wilcoxon')

results = adata.uns['rank_genes_groups']['names']
pvals, log2fc = adata.uns['rank_genes_groups']['pvals'] < 0.05 & adata.uns['rank_genes_groups']['log2fc'] > 1