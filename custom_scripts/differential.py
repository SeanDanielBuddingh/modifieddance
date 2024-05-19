import numpy as np
import pandas as pd
import scanpy as sc
import itertools
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN

import torch

import sys
import os

current_script_path = __file__
current_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
parent_parent = os.path.dirname(parent_dir)
parent_parent = parent_parent.replace("\\", "/")
data_dir_ = parent_parent+'/dance_data'

class GeneMarkers():
    def __init__(self):
        super(GeneMarkers, self).__init__()

        current_script_path = __file__
        current_dir = os.path.dirname(current_script_path)
        parent_dir = os.path.dirname(current_dir)
        sys.path.append(parent_dir)
        parent_parent = os.path.dirname(parent_dir)
        parent_parent = parent_parent.replace("\\", "/")
        data_dir_ = parent_parent+'/dance_data'
        self.path = data_dir_

    def FindGeneMarkers(self, y_values_train, y_values_test, normalized_train, normalized_test):

        obs_train = pd.DataFrame({'condition': y_values_train})
        obs_train.index = obs_train.index.astype(str)

        obs_test = pd.DataFrame({'condition': y_values_test})
        obs_test.index = obs_test.index.astype(str)

        adata1 = sc.AnnData(X=normalized_train, obs=obs_train)
        adata2 = sc.AnnData(X=normalized_test, obs=obs_test)
        adata = sc.concat([adata1, adata2], join="outer")

        groups = adata.obs['condition'].unique()
        num_conditions = len(groups)
        print(f'\nThere are {num_conditions} cell-types in this dataset.\n')
        print(groups,'\n')

        sc.tl.rank_genes_groups(adata, groupby='condition', method='t-test')

        results = adata.uns['rank_genes_groups']

        dfs = {}
        for group in groups:
            df = pd.DataFrame({
                'names': results['names'][group],
                'pvals': results['pvals'][group],
            })
            df = df.sort_values('pvals', ascending=True)
            dfs[group] = df

        tsne = TSNE(n_components=2, random_state=0)
        y = pd.Categorical(adata.obs['condition'])
        y = y.codes
        Z = tsne.fit_transform(X=adata.X, y=y)
        plt.scatter(Z[:, 0], Z[:, 1], c=y, cmap='tab20', s=10)
        plt.savefig('tsne.png')

        dbscan = DBSCAN(eps=0.5, min_samples=5)
        clusters = dbscan.fit_predict(Z)
        adata.obs['dbscan']  = clusters
        plt.scatter(Z[:, 0], Z[:, 1], c=clusters, cmap='tab20', s=10)
        plt.savefig('dbscan.png')

        cluster_conditions = {}
        total_cells = adata.shape[0]
        grouped = adata.obs.groupby('dbscan')
        for cluster_label, group in grouped:
            cluster_cells = len(group)
    
            if cluster_cells / total_cells < 0.1:
                continue
        
            conditions = group['condition']
            unique_conditions = conditions.unique()

            condition_names = []
            for condition in unique_conditions:
                condition_proportion = (conditions == condition).sum() / cluster_cells

                if condition_proportion >= 0.1:
                    condition_names.append(condition)

            if len(condition_names) > 1:
                cluster_conditions[cluster_label] = condition_names

        print(cluster_conditions)

        group_markers = {}
        for cluster, conditions in cluster_conditions.items():
            adata.obs['grouping'] = adata.obs['condition'].apply(lambda x: 'group' if x in conditions else 'rest')
            
            sc.tl.rank_genes_groups(adata, groupby='grouping', method='t-test')

            names = adata.uns['rank_genes_groups']['names']['group']
            pvals = adata.uns['rank_genes_groups']['pvals']['group']

            significant_genes = names[pvals < 0.05]

            group_markers[cluster] = significant_genes

        df_group = pd.DataFrame.from_dict(group_markers, orient='index').transpose()

        return dfs, df_group, cluster_conditions

    def ConstructTargets(self, y_values_train, y_values_test, normalized_train, normalized_test, combined_brain, sublist_length):
        dfs, df_group, cluster_conditions = self.FindGeneMarkers(y_values_train, y_values_test, normalized_train, normalized_test)
        #sublist_length = 10

        all_names = []
        for group, df in dfs.items():
            selected_names = df['names'].iloc[:sublist_length].tolist()
            selected_pvals = df['pvals'].iloc[:sublist_length].tolist()

            if any(pval > 0.05 for pval in selected_pvals):
                raise ValueError("One or more p-values are greater than 0.05.")

            all_names.extend(selected_names)

        # Checks for co-occuring genes -> replaces them with the next set of genes.
        # Goal: Select Genes which do not co-occur, and are statistically significant. (Specific Gene Markers)
        # Instead of checking the list of top 10 genes, we will check directly against the unnormalized data.
        y_all = pd.concat([y_values_train, y_values_test], axis=0)
        y_all = y_all.reset_index(drop=True)

        print('\nChecking for co-occurring genes.\n')
        replacement_counters = {group: 0 for group in dfs.keys()}
        Flag = False
        while not Flag:
            co_occurring_indices = {}
            for name in set(all_names):
                raw_data_indices = combined_brain.loc[combined_brain[name] > 0].index
                labels_with_gene = y_all[raw_data_indices]
                unique_labels = labels_with_gene.unique()
                
                if len(unique_labels) > 1:
                    group = unique_labels[0]

                indices = [i for i, x in enumerate(all_names) if x == name]
                if len(indices) > 1:
                    co_occurring_indices[name] = indices

                    replacement_name = dfs[group]['names'].iloc[sublist_length + replacement_counters[group]]

                    # Check the p-value of the replacement name
                    replacement_pval = dfs[group]['pvals'].iloc[sublist_length + replacement_counters[group]]
                    if replacement_pval > 0.05:
                        raise ValueError("The p-value for the replacement name is greater than 0.05.")

                    replacement_counters[group] += 1
                    all_names[co_occurring_indices[name][0]] = replacement_name
            
            if not co_occurring_indices:
                Flag = True
        print('\nCo-occurring genes have been replaced.\n')

        group_names = [all_names[i:i+sublist_length] for i in range(0, len(all_names), sublist_length)]
        group_names_df = pd.DataFrame(group_names).T

        # group_names_df has columns as groups and rows as gene names
        # columns are in the same order as dfs.keys()
        # here is where i will check the key, and add the supplementary genes
        temp_df = pd.DataFrame(columns=group_names_df.columns)
        for cluster, celltypes in cluster_conditions.items():
            for celltype in celltypes:
                series_df = df_group[cluster][:sublist_length].to_frame()
                temp_df[list(dfs.keys()).index(celltype)] = series_df

        group_names_df = pd.concat([group_names_df, temp_df], axis=0, ignore_index=True)

        group_names_df.reset_index(drop=True, inplace=True)
        unique_names = pd.unique(group_names_df.values[~pd.isna(group_names_df.values)])

        # by enumerating over dfs.items(), we can get the group name and the corresponding df
        targets = {}
        for i, (group, df) in enumerate(dfs.items()):
            group_idx = [unique_names.tolist().index(name) for name in group_names_df.iloc[:, i] if not pd.isna(name)]
            target = np.zeros(len(unique_names))
            target[group_idx] = 1
            targets[group] = target

        ft_y_train = []
        for target in y_values_train:
            new_target = targets[target]
            ft_y_train.append(new_target)

        ft_y_train = pd.DataFrame(ft_y_train)

        ft_y_test = []
        for target in y_values_test:
            new_target = targets[target]
            ft_y_test.append(new_target)

        ft_y_test= pd.DataFrame(ft_y_test)

        print('\nTargets Constructed.\n')

        ft_y_train.to_csv(data_dir_+'/ft_y_train.csv', header=None, index=None)
        ft_y_test.to_csv(data_dir_+'/ft_y_test.csv', header=None, index=None)

        return ft_y_train, ft_y_test, dfs
