import numpy as np
import pandas as pd
import scanpy as sc

import sys
import os

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
        obs_test = pd.DataFrame({'condition': y_values_test})

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

        return dfs

    def ConstructTargets(self, y_values_train, y_values_test, normalized_train, normalized_test):
        dfs = self.FindGeneMarkers(y_values_train, y_values_test, normalized_train, normalized_test)
        sublist_length = 10

        all_names = []
        for group, df in dfs.items():
            selected_names = df['names'].iloc[:sublist_length].tolist()
            selected_pvals = df['pvals'].iloc[:sublist_length].tolist()

            if any(pval > 0.05 for pval in selected_pvals):
                raise ValueError("One or more p-values are greater than 0.05.")

            all_names.extend(selected_names)

        #Checks for co-occuring genes -> replaces them with the next set of genes.
        #Goal: Select Genes which do not co-occur, and are statistically significant. (Specific Gene Markers)
        
        replacement_counters = {group: 0 for group in dfs.keys()}
        while True:
            co_occurring_indices = {}
            for name in set(all_names):
                indices = [i for i, x in enumerate(all_names) if x == name]
                if len(indices) > 1:
                    co_occurring_indices[name] = indices

            if not co_occurring_indices:
                break

            for name, indices in co_occurring_indices.items():
                group_idx = indices[0] // sublist_length
                group = list(dfs.keys())[group_idx]

                replacement_name = dfs[group]['names'].iloc[sublist_length + replacement_counters[group]]

                # Check the p-value of the replacement name
                replacement_pval = dfs[group]['pvals'].iloc[sublist_length + replacement_counters[group]]
                if replacement_pval > 0.05:
                    raise ValueError("The p-value for the replacement name is greater than 0.05.")

                replacement_counters[group] += 1

                all_names[indices[0]] = replacement_name
 
        group_names = [all_names[i:i+sublist_length] for i in range(0, len(all_names), sublist_length)]
        group_names_df = pd.DataFrame(group_names)

        unique_names = pd.unique(group_names_df.values.flatten())
        print('\nnum_classes: ', len(unique_names))

        targets = {}
        for i, (group, df) in enumerate(dfs.items()):
            group_idx = [unique_names.tolist().index(name) for name in group_names[i]]
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
        return ft_y_train, ft_y_test, dfs
