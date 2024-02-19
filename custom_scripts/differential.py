import torch

import numpy as np
import pandas as pd
import scanpy as sc

import random

import sys
import os

from data_pre import data_pre

class GeneMarkers():
    def __init__(self):
        super(GeneMarkers, self).__init__()

    def FindGeneMarkers(self):
        data = data_pre()
        _, _, genes, y_values_train, y_values_test, normalized_train, normalized_test = data.read_w2v()

        obs_train = pd.DataFrame({'condition': y_values_train[0]})
        obs_test = pd.DataFrame({'condition': y_values_test[0]})

        adata1 = sc.AnnData(X=normalized_train.T.reset_index(drop=True), obs=obs_train)
        adata2 = sc.AnnData(X=normalized_test.T.reset_index(drop=True), obs=obs_test)
        adata = sc.concat([adata1, adata2], join="outer")

        sc.tl.rank_genes_groups(adata, groupby='condition', method='t-test')

        results = adata.uns['rank_genes_groups']

        groups = results['names'].dtype.names

        dfs = {}
        for group in groups:
            df = pd.DataFrame({
                'names': results['names'][group],
                'pvals': results['pvals'][group],
                'logfoldchanges': results['logfoldchanges'][group]
            })
            df = df.sort_values('pvals', ascending=True)
            dfs[group] = df

        return dfs, genes, y_values_train, y_values_test

    def ConstructTargets(self):
        dfs, genes, y_values_train, y_values_test = self.FindGeneMarkers()
        gene_idx = genes.index

        all_names = []
        for group, df in dfs.items():
            all_names.extend(df['names'].iloc[:100].tolist())

        unique_names = pd.unique(all_names)
        unique_names_series = pd.Series(unique_names)
        print('\nnum_classes: ', len(unique_names_series))

        targets = {}
        for group, df in dfs.items():
            group_idx = unique_names_series[unique_names_series.isin(df['names'].iloc[:100].tolist())].index
            target = np.zeros(len(unique_names_series))
            target[group_idx] = 1
            targets[group] = target

        ft_y_train = []
        for target in y_values_train[0]:
            new_target = targets[target]
            ft_y_train.append(new_target)

        ft_y_train = pd.DataFrame(ft_y_train)

        ft_y_test = []
        for target in y_values_test[0]:
            new_target = targets[target]
            ft_y_test.append(new_target)

        ft_y_test= pd.DataFrame(ft_y_test)

        print('\nTargets Constructed.\n')
        return ft_y_train, ft_y_test
