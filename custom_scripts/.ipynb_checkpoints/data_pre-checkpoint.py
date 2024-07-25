import os
import sys
current_script_path = __file__
current_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

os.environ['DGLBACKEND'] = 'pytorch'
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from gensim.models import Word2Vec
#from transformers import BertTokenizer, BertModel



class data_pre():
    def __init__(self):
        self.seed = 42
        self.dimensions = 2500

        current_script_path = __file__
        current_dir = os.path.dirname(current_script_path)
        parent_dir = os.path.dirname(current_dir)
        sys.path.append(parent_dir)
        parent_parent = os.path.dirname(parent_dir)
        parent_parent = parent_parent.replace("\\", "/")
        data_dir_ = parent_parent+'/dance_data'
        self.path = data_dir_
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#         brain_train, brain_test, brain_train_labels, brain_test_labels, corpus_train_brain, spleen_train, spleen_test, spleen_train_labels, spleen_test_labels, corpus_train_spleen, kidney_train, kidney_test, kidney_train_labels, kidney_test_labels, corpus_train_kidney = self.load_data()
#         pancreas_train, pancreas_test, pancreas_train_labels, pancreas_test_labels, corpus_train_pancreas, spleen_train, spleen_test, spleen_train_labels, spleen_test_labels, corpus_train_spleen, bonemarrow_train, bonemarrow_test, bonemarrow_train_labels, bonemarrow_test_labels, corpus_train_bonemarrow = self.load_human_data()

#         self.get_w2v(brain_train, brain_test, brain_train_labels, brain_test_labels, corpus_train_brain, spleen_train, spleen_test, spleen_train_labels, spleen_test_labels, corpus_train_spleen, kidney_train, kidney_test, kidney_train_labels, kidney_test_labels, corpus_train_kidney)
#         self.get_w2v_human(pancreas_train, pancreas_test, pancreas_train_labels, pancreas_test_labels, corpus_train_pancreas, spleen_train, spleen_test, spleen_train_labels, spleen_test_labels, corpus_train_spleen, bonemarrow_train, bonemarrow_test, bonemarrow_train_labels, bonemarrow_test_labels, corpus_train_bonemarrow)

        #self.bert_embed(brain_test, corpus_brain, brain_y)
        #self.bert_embed(spleen_x, corpus_spleen, spleen_y)
        #self.bert_embed(spleen_x, corpus_spleen, kidney_y)
        #self.bert_embed(brain_train, corpus_btrain, btrain_y)
        #self.bert_embed(s_train, corpus_strain, strain_y)
        #self.bert_embed(k_train, corpus_ktrain, ktrain_y)
    def load_human_data(self):
        '''
        Human Pancreas
        '''
        pancreas_train = pd.read_csv(self.path+'/train/human/human_cell_atlas/human_Pancreas9727_data.csv', header=0, index_col=0)
        pancreas_train_y = pd.read_csv(self.path+'/train/human/human_cell_atlas/human_Pancreas9727_celltype.csv')['Cell_type']
            
        pancreas_test = pd.concat([pd.read_csv(self.path+'/test/human/human_test_data/human_Pancreas2227_data.csv', header=0, index_col=0),
                             pd.read_csv(self.path+'/test/human/human_test_data/human_Pancreas1841_data.csv', header=0, index_col=0)],axis=1, ignore_index=False)
        pancreas_test_y = pd.concat([pd.read_csv(self.path+'/test/human/human_test_data/human_Pancreas2227_celltype.csv')['Cell_type'],
                              pd.read_csv(self.path+'/test/human/human_test_data/human_Pancreas1841_celltype.csv')['Cell_type']], axis=0, ignore_index=True).reset_index(drop=True)
        
        #removes labels with only one occurence 
        name_counts = pancreas_train_y.value_counts()

        unique_name = name_counts[name_counts == 1].index
        
        unique_index = pancreas_train_y[pancreas_train_y.isin(unique_name)].index

        print(f"Unique name train: {unique_name}, Index: {unique_index}")

        pancreas_train_y.drop(unique_index, inplace=True)
        pancreas_train = pancreas_train.T.reset_index(drop=True)
        pancreas_train.drop(unique_index, inplace=True)
        pancreas_train = pancreas_train.T

        name_counts = pancreas_test_y.value_counts()

        unique_name = name_counts[name_counts == 1].index
        
        unique_index = pancreas_test_y[pancreas_test_y.isin(unique_name)].index

        print(f"Unique name train: {unique_name}, Index: {unique_index}")

        pancreas_test_y.drop(unique_index, inplace=True)
        pancreas_test = pancreas_test.T.reset_index(drop=True)
        pancreas_test.drop(unique_index, inplace=True)
        pancreas_test = pancreas_test.T


        combined_pancreas_labels = pd.concat([pancreas_train_y, pancreas_test_y], axis=0, ignore_index=True).reset_index(drop=True)
        train_genes = set(pancreas_train.index)
        test_genes = set(pancreas_test.index)
        common_genes = train_genes.intersection(test_genes)

        combined_pancreas = pd.concat([pancreas_train, pancreas_test], axis=1, ignore_index=False).fillna(0).loc[list(common_genes)]
        combined_pancreas.columns = range(combined_pancreas.shape[1])

        combined_pancreas = combined_pancreas.apply(lambda x: x/x.sum(), axis=0)
        combined_pancreas = combined_pancreas.apply(lambda x: x*1e4, axis=0)
        combined_pancreas = combined_pancreas.apply(lambda x: np.log2(1+x), axis=0).fillna(0)

        row_sums = combined_pancreas.sum(axis=1)
        low_percentile = np.percentile(row_sums, 1)
        high_percentile = np.percentile(row_sums, 99)
        combined_pancreas = combined_pancreas[(row_sums > low_percentile) & (row_sums < high_percentile)]
        row_stds = combined_pancreas.std(axis=1)
        low_std_percentile = np.percentile(row_stds, 1)
        high_std_percentile = np.percentile(row_stds, 99)
        combined_pancreas_filtered = combined_pancreas[(row_stds > low_std_percentile) & (row_stds < high_std_percentile)]
     
        pancreas_train, pancreas_test, pancreas_train_labels, pancreas_test_labels = train_test_split(combined_pancreas_filtered.T, combined_pancreas_labels, test_size=0.2, random_state=self.seed, stratify=combined_pancreas_labels)
        print('it split')
        pancreas_train=pancreas_train.T
        pancreas_test = pancreas_test.T

        pancreas_train.to_csv(self.path+'/normalized_pancreas_train.csv', index=True, header=True)
        pancreas_test.to_csv(self.path+'/normalized_pancreas_test.csv', index=True, header=True)      

        corpus_train_pancreas = []
        for c_name in pancreas_train.columns:
            cell = pancreas_train[c_name]
            sorted = cell[cell!=0].sort_values(ascending=False)
            corpus_train_pancreas.append(sorted.index.tolist())
        '''
        Human Spleen
        '''
        spleen_train = pd.read_csv(self.path+'/train/human/human_cell_atlas/human_Spleen15806_data.csv', header=0, index_col=0)
        spleen_train_y = pd.read_csv(self.path+'/train/human/human_cell_atlas/human_Spleen15806_celltype.csv')['Cell_type']
            
        spleen_test = pd.read_csv(self.path+'/test/human/human_test_data/human_Spleen9887_data.csv', header=0, index_col=0)
        spleen_test_y = pd.read_csv(self.path+'/test/human/human_test_data/human_Spleen9887_celltype.csv')['Cell_type']

        #removes labels with only one occurence 
        name_counts = spleen_train_y.value_counts()

        unique_name = name_counts[name_counts == 1].index
        
        unique_index = spleen_train_y[spleen_train_y.isin(unique_name)].index

        print(f"Unique name train: {unique_name}, Index: {unique_index}")

        spleen_train_y.drop(unique_index, inplace=True)
        spleen_train = spleen_train.T.reset_index(drop=True)
        spleen_train.drop(unique_index, inplace=True)
        spleen_train = spleen_train.T

        name_counts = spleen_test_y.value_counts()

        unique_name = name_counts[name_counts == 1].index
        
        unique_index = spleen_test_y[spleen_test_y.isin(unique_name)].index

        print(f"Unique name train: {unique_name}, Index: {unique_index}")

        spleen_test_y.drop(unique_index, inplace=True)
        spleen_test = spleen_test.T.reset_index(drop=True)
        spleen_test.drop(unique_index, inplace=True)
        spleen_test = spleen_test.T

        combined_spleen_labels = pd.concat([spleen_train_y, spleen_test_y], axis=0, ignore_index=True).reset_index(drop=True)
        train_genes = set(spleen_train.index)
        test_genes = set(spleen_test.index)
        common_genes = train_genes.intersection(test_genes)

        combined_spleen = pd.concat([spleen_train, spleen_test], axis=1, ignore_index=False).fillna(0).loc[list(common_genes)]
        combined_spleen.columns = range(combined_spleen.shape[1])

        combined_spleen = combined_spleen.apply(lambda x: x/x.sum(), axis=0)
        combined_spleen = combined_spleen.apply(lambda x: x*1e4, axis=0)
        combined_spleen = combined_spleen.apply(lambda x: np.log2(1+x), axis=0).fillna(0)

        row_sums = combined_spleen.sum(axis=1)
        low_percentile = np.percentile(row_sums, 1)
        high_percentile = np.percentile(row_sums, 99)
        combined_spleen = combined_spleen[(row_sums > low_percentile) & (row_sums < high_percentile)]
        row_stds = combined_spleen.std(axis=1)
        low_std_percentile = np.percentile(row_stds, 1)
        high_std_percentile = np.percentile(row_stds, 99)
        combined_spleen_filtered = combined_spleen[(row_stds > low_std_percentile) & (row_stds < high_std_percentile)]
     
        spleen_train, spleen_test, spleen_train_labels, spleen_test_labels = train_test_split(combined_spleen_filtered.T, combined_spleen_labels, test_size=0.2, random_state=self.seed, stratify=combined_spleen_labels)

        spleen_train=spleen_train.T
        spleen_test = spleen_test.T

        spleen_train.to_csv(self.path+'/normalized_spleen_human_train.csv', index=True, header=True)
        spleen_test.to_csv(self.path+'/normalized_spleen_human_test.csv', index=True, header=True)      

        corpus_train_spleen = []
        for c_name in spleen_train.columns:
            cell = spleen_train[c_name]
            sorted = cell[cell!=0].sort_values(ascending=False)
            corpus_train_spleen.append(sorted.index.tolist())
        '''
        Human Bone Marrow
        '''
        bonemarrow_train = pd.read_csv(self.path+'/train/human/human_cell_atlas/human_Bone_marrow2261_data.csv', header=0, index_col=0)
        bonemarrow_train_y = pd.read_csv(self.path+'/train/human/human_cell_atlas/human_Bone_marrow2261_celltype.csv')['Cell_type']
            
        bonemarrow_test = pd.read_csv(self.path+'/train/human/human_cell_atlas/human_Bone_marrow6443_data.csv', header=0, index_col=0)
        bonemarrow_test_y = pd.read_csv(self.path+'/train/human/human_cell_atlas/human_Bone_marrow6443_celltype.csv')['Cell_type']

        #removes labels with only one occurence 
        name_counts = bonemarrow_train_y.value_counts()

        unique_name = name_counts[name_counts == 1].index
        
        unique_index = bonemarrow_train_y[bonemarrow_train_y.isin(unique_name)].index

        print(f"Unique name train: {unique_name}, Index: {unique_index}")

        bonemarrow_train_y.drop(unique_index, inplace=True)
        bonemarrow_train = bonemarrow_train.T.reset_index(drop=True)
        bonemarrow_train.drop(unique_index, inplace=True)
        bonemarrow_train = bonemarrow_train.T

        name_counts = bonemarrow_test_y.value_counts()

        unique_name = name_counts[name_counts == 1].index
        
        unique_index = bonemarrow_test_y[bonemarrow_test_y.isin(unique_name)].index

        print(f"Unique name train: {unique_name}, Index: {unique_index}")

        bonemarrow_test_y.drop(unique_index, inplace=True)
        bonemarrow_test = bonemarrow_test.T.reset_index(drop=True)
        bonemarrow_test.drop(unique_index, inplace=True)
        bonemarrow_test = bonemarrow_test.T

        combined_bonemarrow_labels = pd.concat([bonemarrow_train_y, bonemarrow_test_y], axis=0, ignore_index=True).reset_index(drop=True)
        train_genes = set(bonemarrow_train.index)
        test_genes = set(bonemarrow_test.index)
        common_genes = train_genes.intersection(test_genes)

        combined_bonemarrow = pd.concat([bonemarrow_train, bonemarrow_test], axis=1, ignore_index=False).fillna(0).loc[list(common_genes)]
        combined_bonemarrow.columns = range(combined_bonemarrow.shape[1])

        combined_bonemarrow = combined_bonemarrow.apply(lambda x: x/x.sum(), axis=0)
        combined_bonemarrow = combined_bonemarrow.apply(lambda x: x*1e4, axis=0)
        combined_bonemarrow = combined_bonemarrow.apply(lambda x: np.log2(1+x), axis=0).fillna(0)

        row_sums = combined_bonemarrow.sum(axis=1)
        low_percentile = np.percentile(row_sums, 1)
        high_percentile = np.percentile(row_sums, 99)
        combined_bonemarrow = combined_bonemarrow[(row_sums > low_percentile) & (row_sums < high_percentile)]
        row_stds = combined_bonemarrow.std(axis=1)
        low_std_percentile = np.percentile(row_stds, 1)
        high_std_percentile = np.percentile(row_stds, 99)
        combined_bonemarrow_filtered = combined_bonemarrow[(row_stds > low_std_percentile) & (row_stds < high_std_percentile)]
     
        bonemarrow_train, bonemarrow_test, bonemarrow_train_labels, bonemarrow_test_labels = train_test_split(combined_bonemarrow_filtered.T, combined_bonemarrow_labels, test_size=0.2, random_state=self.seed, stratify=combined_bonemarrow_labels)

        bonemarrow_train=bonemarrow_train.T
        bonemarrow_test = bonemarrow_test.T

        bonemarrow_train.to_csv(self.path+'/normalized_bonemarrow_train.csv', index=True, header=True)
        bonemarrow_test.to_csv(self.path+'/normalized_bonemarrow_test.csv', index=True, header=True)      

        corpus_train_bonemarrow = []
        for c_name in bonemarrow_train.columns:
            cell = bonemarrow_train[c_name]
            sorted = cell[cell!=0].sort_values(ascending=False)
            corpus_train_bonemarrow.append(sorted.index.tolist())

        print('loaded')
        
        return pancreas_train, pancreas_test, pancreas_train_labels, pancreas_test_labels, corpus_train_pancreas, spleen_train, spleen_test, spleen_train_labels, spleen_test_labels, corpus_train_spleen, bonemarrow_train, bonemarrow_test, bonemarrow_train_labels, bonemarrow_test_labels, corpus_train_bonemarrow

    def load_data(self):
        '''
        Mouse Pre-Processing
        '''
        brain_train = pd.concat([pd.read_csv(self.path+'/train/mouse/mouse_Brain753_data.csv', header=0, index_col=0),
                             pd.read_csv(self.path+'/train/mouse/mouse_Brain3285_data.csv', header=0, index_col=0)],axis=1, ignore_index=False)
        brain_train_y = pd.concat([pd.read_csv(self.path+'/train/mouse/mouse_Brain753_celltype.csv')['Cell_type'],
                              pd.read_csv(self.path+'/train/mouse/mouse_Brain3285_celltype.csv')['Cell_type']], axis=0, ignore_index=True).reset_index(drop=True)
            
        brain_test = pd.read_csv(self.path+'/test/mouse/mouse_Brain2695_data.csv', header=0, index_col=0)
        brain_test_y = pd.read_csv(self.path+'/test/mouse/mouse_Brain2695_celltype.csv')['Cell_type'].reset_index(drop=True)

        combined_brain_labels = pd.concat([brain_train_y, brain_test_y], axis=0, ignore_index=True).reset_index(drop=True)
        train_genes = set(brain_train.index)
        test_genes = set(brain_test.index)
        common_genes = train_genes.intersection(test_genes)

        combined_brain = pd.concat([brain_train, brain_test], axis=1, ignore_index=False).fillna(0).loc[list(common_genes)]
        combined_brain.columns = range(combined_brain.shape[1])

        combined_brain = combined_brain.apply(lambda x: x/x.sum(), axis=0)
        combined_brain = combined_brain.apply(lambda x: x*1e4, axis=0)
        combined_brain = combined_brain.apply(lambda x: np.log2(1+x), axis=0).fillna(0)

        row_sums = combined_brain.sum(axis=1)
        low_percentile = np.percentile(row_sums, 1)
        high_percentile = np.percentile(row_sums, 99)
        combined_brain = combined_brain[(row_sums > low_percentile) & (row_sums < high_percentile)]
        row_stds = combined_brain.std(axis=1)
        low_std_percentile = np.percentile(row_stds, 1)
        high_std_percentile = np.percentile(row_stds, 99)
        combined_brain_filtered = combined_brain[(row_stds > low_std_percentile) & (row_stds < high_std_percentile)]
     
        brain_train, brain_test, brain_train_labels, brain_test_labels = train_test_split(combined_brain_filtered.T, combined_brain_labels, test_size=0.2, random_state=self.seed, stratify=combined_brain_labels)

        brain_train=brain_train.T
        brain_test = brain_test.T

        brain_train.to_csv(self.path+'/normalized_brain_train.csv', index=True, header=True)
        brain_test.to_csv(self.path+'/normalized_brain_test.csv', index=True, header=True)      

        corpus_train_brain = []
        for c_name in brain_train.columns:
            cell = brain_train[c_name]
            sorted = cell[cell!=0].sort_values(ascending=False)
            corpus_train_brain.append(sorted.index.tolist())

        
        spleen_train = pd.read_csv(self.path+'/train/mouse/mouse_Spleen1970_data.csv', header=0, index_col=0)
        spleen_train_y = pd.read_csv(self.path+'/train/mouse/mouse_Spleen1970_celltype.csv')['Cell_type'].reset_index(drop=True)

        spleen_test = pd.read_csv(self.path+'/test/mouse/mouse_Spleen1759_data.csv', header=0, index_col=0)
        spleen_test_y = pd.read_csv(self.path+'/test/mouse/mouse_Spleen1759_celltype.csv')['Cell_type'].reset_index(drop=True)

        combined_spleen_labels = pd.concat([spleen_train_y, spleen_test_y], axis=0, ignore_index=True).reset_index(drop=True)

        train_genes = set(spleen_train.index)
        test_genes = set(spleen_test.index)
        common_genes = train_genes.intersection(test_genes)

        combined_spleen = pd.concat([spleen_train, spleen_test], axis=1, ignore_index=False).fillna(0).loc[list(common_genes)]
        combined_spleen.columns = range(combined_spleen.shape[1])

        combined_spleen = combined_spleen.apply(lambda x: x/x.sum(), axis=0)
        combined_spleen = combined_spleen.apply(lambda x: x*1e4, axis=0)
        combined_spleen = combined_spleen.apply(lambda x: np.log2(1+x), axis=0).fillna(0)

        row_sums = combined_spleen.sum(axis=1)
        low_percentile = np.percentile(row_sums, 1)
        high_percentile = np.percentile(row_sums, 99)
        combined_spleen = combined_spleen[(row_sums > low_percentile) & (row_sums < high_percentile)]
        row_stds = combined_spleen.std(axis=1)
        low_std_percentile = np.percentile(row_stds, 1)
        high_std_percentile = np.percentile(row_stds, 99)
        combined_spleen_filtered = combined_spleen[(row_stds > low_std_percentile) & (row_stds < high_std_percentile)]

        spleen_train, spleen_test, spleen_train_labels, spleen_test_labels = train_test_split(combined_spleen_filtered.T, combined_spleen_labels, test_size=0.2, random_state=self.seed, stratify=combined_spleen_labels)

        spleen_train = spleen_train.T
        spleen_test = spleen_test.T

        spleen_train.to_csv(self.path+'/normalized_spleen_train.csv', index=True, header=True)
        spleen_test.to_csv(self.path+'/normalized_spleen_test.csv', index=True, header=True)      

        corpus_train_spleen = []
        for c_name in spleen_train.columns:
            cell = spleen_train[c_name]
            sorted = cell[cell!=0].sort_values(ascending=False)
            corpus_train_spleen.append(sorted.index.tolist())

        kidney_train = pd.read_csv(self.path+'/train/mouse/mouse_Kidney4682_data.csv', header=0, index_col=0).fillna(0)
        kidney_train_y = pd.read_csv(self.path+'/train/mouse/mouse_Kidney4682_celltype.csv')['Cell_type'].reset_index(drop=True)
        kidney_test = pd.read_csv(self.path+'/test/mouse/mouse_Kidney203_data.csv', header=0, index_col=0).fillna(0)
        kidney_test_y = pd.read_csv(self.path+'/test/mouse/mouse_Kidney203_celltype.csv')['Cell_type'].reset_index(drop=True)

        combined_kidney_labels = pd.concat([kidney_train_y, kidney_test_y], axis=0, ignore_index=True).reset_index(drop=True)

        train_genes = set(kidney_train.index)
        test_genes = set(kidney_test.index)
        common_genes = train_genes.intersection(test_genes)

        combined_kidney = pd.concat([kidney_train, kidney_test], axis=1, ignore_index=False).fillna(0).loc[list(common_genes)]
        combined_kidney.columns = range(combined_kidney.shape[1])

        combined_kidney = combined_kidney.apply(lambda x: x/x.sum(), axis=0)
        combined_kidney = combined_kidney.apply(lambda x: x*1e4, axis=0)
        combined_kidney = combined_kidney.apply(lambda x: np.log2(1+x), axis=0).fillna(0)

        row_sums = combined_kidney.sum(axis=1)
        low_percentile = np.percentile(row_sums, 1)
        high_percentile = np.percentile(row_sums, 99)
        combined_kidney = combined_kidney[(row_sums > low_percentile) & (row_sums < high_percentile)]
        row_stds = combined_kidney.std(axis=1)
        low_std_percentile = np.percentile(row_stds, 1)
        high_std_percentile = np.percentile(row_stds, 99)
        combined_kidney_filtered = combined_kidney[(row_stds > low_std_percentile) & (row_stds < high_std_percentile)]
        
        kidney_train, kidney_test, kidney_train_labels, kidney_test_labels = train_test_split(combined_kidney_filtered.T, combined_kidney_labels, test_size=0.2, random_state=self.seed, stratify=combined_kidney_labels)

        kidney_train = kidney_train.T
        kidney_test = kidney_test.T

        kidney_train.to_csv(self.path+'/normalized_kidney_train.csv', index=True, header=True)
        kidney_test.to_csv(self.path+'/normalized_kidney_test.csv', index=True, header=True)      

        corpus_train_kidney = []
        for c_name in kidney_train.columns:
            cell = kidney_train[c_name]
            sorted = cell[cell!=0].sort_values(ascending=False)
            corpus_train_kidney.append(sorted.index.tolist())
        
        print('loaded')
        
        return brain_train, brain_test, brain_train_labels, brain_test_labels, corpus_train_brain, spleen_train, spleen_test, spleen_train_labels, spleen_test_labels, corpus_train_spleen, kidney_train, kidney_test, kidney_train_labels, kidney_test_labels, corpus_train_kidney

    def get_w2v(self,  brain_train, brain_test, brain_train_labels, brain_test_labels, corpus_train_brain, spleen_train, spleen_test, spleen_train_labels, spleen_test_labels, corpus_train_spleen, kidney_train, kidney_test, kidney_train_labels, kidney_test_labels, corpus_train_kidney):
        
        b_w2v = self.w2v_embed(corpus_train_brain).wv
        b_matrix = np.zeros((len(brain_train), b_w2v.vector_size))

        for i, gene in enumerate(brain_train.index):
            try:
                b_matrix[i] = b_w2v[gene]
            except KeyError:
                b_matrix[i] = np.zeros(b_w2v.vector_size)
        #print(b_matrix.shape)
        b_cells = brain_train.T.values @ b_matrix
        b_cells = pd.DataFrame(b_cells)
        b_cells.to_csv(self.path+'/brain_train.csv', index=False, header=False)

        b_te_cells = brain_test.T.values @ b_matrix
        b_te_cells = pd.DataFrame(b_te_cells)
        b_te_cells.to_csv(self.path+'/brain_test.csv', index=False, header=False)

        brain_genes = pd.DataFrame(b_matrix)
        brain_genes.index = brain_train.index 
        brain_genes.to_csv(self.path+'/brain_w2v_genes.csv', index=True, header=False)
        
        s_w2v = self.w2v_embed(corpus_train_spleen).wv
        s_matrix = np.zeros((len(spleen_train), s_w2v.vector_size))

        for i, gene in enumerate(spleen_train.index):
            try:
                s_matrix[i] = s_w2v[gene]
            except KeyError:
                s_matrix[i] = np.zeros(s_w2v.vector_size)

        s_cells = spleen_train.T.values @ s_matrix
        s_cells = pd.DataFrame(s_cells)
        s_cells.to_csv(self.path+'/spleen_train.csv', index=False, header=False)

        s_te_cells = spleen_test.T.values @ s_matrix
        s_te_cells = pd.DataFrame(s_te_cells)
        s_te_cells.to_csv(self.path+'/spleen_test.csv', index=False, header=False)

        spleen_genes = pd.DataFrame(s_matrix)
        spleen_genes.index = spleen_train.index 
        spleen_genes.to_csv(self.path+'/spleen_w2v_genes.csv', index=True, header=False)

        k_w2v = self.w2v_embed(corpus_train_kidney).wv
        k_matrix = np.zeros((len(kidney_train), k_w2v.vector_size))

        for i, gene in enumerate(kidney_train.index):
            try:
                k_matrix[i] = k_w2v[gene]
            except KeyError:
                k_matrix[i] = np.zeros(k_w2v.vector_size)

        k_cells = kidney_train.T.values @ k_matrix
        k_cells = pd.DataFrame(k_cells)
        k_cells.to_csv(self.path+'/kidney_train.csv', index=False, header=False)

        k_te_cells = kidney_test.T.values @ k_matrix
        k_te_cells = pd.DataFrame(k_te_cells)
        k_te_cells.to_csv(self.path+'/kidney_test.csv', index=False, header=False)

        kidney_genes = pd.DataFrame(k_matrix)
        kidney_genes.index = kidney_train.index 
        kidney_genes.to_csv(self.path+'/kidney_w2v_genes.csv', index=True, header=False)
        
        brain_train_labels.to_csv(self.path+'/brain_train_labels.csv', index=False)
        brain_test_labels.to_csv(self.path+'/brain_test_labels.csv', index=False)
        spleen_train_labels.to_csv(self.path+'/spleen_train_labels.csv', index=False)
        spleen_test_labels.to_csv(self.path+'/spleen_test_labels.csv', index=False)
        kidney_train_labels.to_csv(self.path+'/kidney_train_labels.csv', index=False)
        kidney_test_labels.to_csv(self.path+'/kidney_test_labels.csv', index=False)
        
        print('printed')

    def get_w2v_human(self,  brain_train, brain_test, brain_train_labels, brain_test_labels, corpus_train_brain, spleen_train, spleen_test, spleen_train_labels, spleen_test_labels, corpus_train_spleen, kidney_train, kidney_test, kidney_train_labels, kidney_test_labels, corpus_train_kidney):
        
        b_w2v = self.w2v_embed(corpus_train_brain).wv
        b_matrix = np.zeros((len(brain_train), b_w2v.vector_size))

        for i, gene in enumerate(brain_train.index):
            try:
                b_matrix[i] = b_w2v[gene]
            except KeyError:
                b_matrix[i] = np.zeros(b_w2v.vector_size)
        #print(b_matrix.shape)
        b_cells = brain_train.T.values @ b_matrix
        b_cells = pd.DataFrame(b_cells)
        b_cells.to_csv(self.path+'/pancreas_train.csv', index=False, header=False)

        b_te_cells = brain_test.T.values @ b_matrix
        b_te_cells = pd.DataFrame(b_te_cells)
        b_te_cells.to_csv(self.path+'/pancreas_test.csv', index=False, header=False)

        brain_genes = pd.DataFrame(b_matrix)
        brain_genes.index = brain_train.index 
        brain_genes.to_csv(self.path+'/pancreas_w2v_genes.csv', index=True, header=False)
        
        s_w2v = self.w2v_embed(corpus_train_spleen).wv
        s_matrix = np.zeros((len(spleen_train), s_w2v.vector_size))

        for i, gene in enumerate(spleen_train.index):
            try:
                s_matrix[i] = s_w2v[gene]
            except KeyError:
                s_matrix[i] = np.zeros(s_w2v.vector_size)

        s_cells = spleen_train.T.values @ s_matrix
        s_cells = pd.DataFrame(s_cells)
        s_cells.to_csv(self.path+'/spleen_human_train.csv', index=False, header=False)

        s_te_cells = spleen_test.T.values @ s_matrix
        s_te_cells = pd.DataFrame(s_te_cells)
        s_te_cells.to_csv(self.path+'/spleen_human_test.csv', index=False, header=False)

        spleen_genes = pd.DataFrame(s_matrix)
        spleen_genes.index = spleen_train.index 
        spleen_genes.to_csv(self.path+'/spleen_human_w2v_genes.csv', index=True, header=False)

        k_w2v = self.w2v_embed(corpus_train_kidney).wv
        k_matrix = np.zeros((len(kidney_train), k_w2v.vector_size))

        for i, gene in enumerate(kidney_train.index):
            try:
                k_matrix[i] = k_w2v[gene]
            except KeyError:
                k_matrix[i] = np.zeros(k_w2v.vector_size)

        k_cells = kidney_train.T.values @ k_matrix
        k_cells = pd.DataFrame(k_cells)
        k_cells.to_csv(self.path+'/bone_train.csv', index=False, header=False)

        k_te_cells = kidney_test.T.values @ k_matrix
        k_te_cells = pd.DataFrame(k_te_cells)
        k_te_cells.to_csv(self.path+'/bone_test.csv', index=False, header=False)

        kidney_genes = pd.DataFrame(k_matrix)
        kidney_genes.index = kidney_train.index 
        kidney_genes.to_csv(self.path+'/bone_w2v_genes.csv', index=True, header=False)
        
        brain_train_labels.to_csv(self.path+'/pancreas_train_labels.csv', index=False)
        brain_test_labels.to_csv(self.path+'/pancreas_test_labels.csv', index=False)
        spleen_train_labels.to_csv(self.path+'/spleen_human_train_labels.csv', index=False)
        spleen_test_labels.to_csv(self.path+'/spleen_human_test_labels.csv', index=False)
        kidney_train_labels.to_csv(self.path+'/bone_train_labels.csv', index=False)
        kidney_test_labels.to_csv(self.path+'/bone_test_labels.csv', index=False)
        
        print('printed')

    def read_w2v(self, dataset):
        
        if dataset == 'mouse_Brain':
            tissue_train = pd.read_csv(self.path+'/brain_train.csv', header=None)
            tissue_test = pd.read_csv(self.path+'/brain_test.csv', header=None)
            genes = pd.read_csv(self.path+'/brain_w2v_genes.csv', header=None)
            y_values_train = pd.read_csv(self.path+'/brain_train_labels.csv', skiprows=1, header=None, dtype=str)
            y_values_test = pd.read_csv(self.path+'/brain_test_labels.csv', skiprows=1, header=None, dtype=str)
            normalized_train = pd.read_csv(self.path+'/normalized_brain_train.csv', header=0, index_col=0)
            normalized_test = pd.read_csv(self.path+'/normalized_brain_test.csv', header=0, index_col=0)

            # raw inputs

            brain_train = pd.concat([pd.read_csv(self.path+'/train/mouse/mouse_Brain753_data.csv', header=0, index_col=0),
                                 pd.read_csv(self.path+'/train/mouse/mouse_Brain3285_data.csv', header=0, index_col=0)],axis=1, ignore_index=False)
            brain_test = pd.read_csv(self.path+'/test/mouse/mouse_Brain2695_data.csv', header=0, index_col=0)

            return tissue_train, tissue_test, genes, y_values_train, y_values_test, normalized_train, normalized_test, brain_train, brain_test
        
        elif dataset == 'mouse_Kidney':
            tissue_train = pd.read_csv(self.path+'/kidney_train.csv', header=None)
            tissue_test = pd.read_csv(self.path+'/kidney_test.csv', header=None)
            genes = pd.read_csv(self.path+'/kidney_w2v_genes.csv', header=None)
            y_values_train = pd.read_csv(self.path+'/kidney_train_labels.csv', skiprows=1, header=None, dtype=str)
            y_values_test = pd.read_csv(self.path+'/kidney_test_labels.csv', skiprows=1, header=None, dtype=str)
            normalized_train = pd.read_csv(self.path+'/normalized_kidney_train.csv', header=0, index_col=0)
            normalized_test = pd.read_csv(self.path+'/normalized_kidney_test.csv', header=0, index_col=0)

            # raw inputs

            kidney_train = pd.read_csv(self.path+'/train/mouse/mouse_Kidney4682_data.csv', header=0, index_col=0).fillna(0)
            kidney_test = pd.read_csv(self.path+'/test/mouse/mouse_Kidney203_data.csv', header=0, index_col=0).fillna(0)

            return tissue_train, tissue_test, genes, y_values_train, y_values_test, normalized_train, normalized_test, kidney_train, kidney_test
        
        elif dataset == 'human_Pancreas':
            tissue_train = pd.read_csv(self.path+'/pancreas_train.csv', header=None)
            tissue_test = pd.read_csv(self.path+'/pancreas_test.csv', header=None)
            genes = pd.read_csv(self.path+'/pancreas_w2v_genes.csv', header=None)
            y_values_train = pd.read_csv(self.path+'/pancreas_train_labels.csv', skiprows=1, header=None, dtype=str)
            y_values_test = pd.read_csv(self.path+'/pancreas_test_labels.csv', skiprows=1, header=None, dtype=str)
            normalized_train = pd.read_csv(self.path+'/normalized_pancreas_train.csv', header=0, index_col=0)
            normalized_test = pd.read_csv(self.path+'/normalized_pancreas_test.csv', header=0, index_col=0)

            # raw inputs

            pancreas_train = pd.read_csv(self.path+'/train/human/human_cell_atlas/human_Pancreas9727_data.csv', header=0, index_col=0)
            pancreas_test = pd.concat([pd.read_csv(self.path+'/test/human/human_test_data/human_Pancreas2227_data.csv', header=0, index_col=0),
                                 pd.read_csv(self.path+'/test/human/human_test_data/human_Pancreas1841_data.csv', header=0, index_col=0)],axis=1, ignore_index=False)

            return tissue_train, tissue_test, genes, y_values_train, y_values_test, normalized_train, normalized_test, pancreas_train, pancreas_test
        
        elif dataset == 'human_Spleen':
            tissue_train = pd.read_csv(self.path+'/human_spleen_train.csv', header=None)
            tissue_test = pd.read_csv(self.path+'/human_spleen_test.csv', header=None)
            genes = pd.read_csv(self.path+'/human_spleen_w2v_genes.csv', header=None)
            y_values_train = pd.read_csv(self.path+'/human_spleen_train_labels.csv', skiprows=1, header=None, dtype=str)
            y_values_test = pd.read_csv(self.path+'/human_spleen_test_labels.csv', skiprows=1, header=None, dtype=str)
            normalized_train = pd.read_csv(self.path+'/normalized_human_spleen_train.csv', header=0, index_col=0)
            normalized_test = pd.read_csv(self.path+'/normalized_human_spleen_test.csv', header=0, index_col=0)

            # raw inputs

            spleen_train = pd.read_csv(self.path+'/train/human/human_cell_atlas/human_Spleen15806_data.csv', header=0, index_col=0)
            spleen_test = pd.read_csv(self.path+'/test/human/human_test_data/human_Spleen9887_data.csv', header=0, index_col=0)

            return tissue_train, tissue_test, genes, y_values_train, y_values_test, normalized_train, normalized_test, spleen_train, spleen_test
        
        elif dataset == 'human_Bonemarrow':
            tissue_train = pd.read_csv(self.path+'/bone_train.csv', header=None)
            tissue_test = pd.read_csv(self.path+'/bone_test.csv', header=None)
            genes = pd.read_csv(self.path+'/bone_w2v_genes.csv', header=None)
            y_values_train = pd.read_csv(self.path+'/bone_train_labels.csv', skiprows=1, header=None, dtype=str)
            y_values_test = pd.read_csv(self.path+'/bone_test_labels.csv', skiprows=1, header=None, dtype=str)
            normalized_train = pd.read_csv(self.path+'/normalized_bone_train.csv', header=0, index_col=0)
            normalized_test = pd.read_csv(self.path+'/normalized_bone_test.csv', header=0, index_col=0)

            # raw inputs

            bonemarrow_train = pd.read_csv(self.path+'/train/human/human_cell_atlas/human_Bone_marrow2261_data.csv', header=0, index_col=0)
            bonemarrow_test = pd.read_csv(self.path+'/train/human/human_cell_atlas/human_Bone_marrow6443_data.csv', header=0, index_col=0)

            return tissue_train, tissue_test, genes, y_values_train, y_values_test, normalized_train, normalized_test, bonemarrow_train, bonemarrow_test
    
        elif dataset == 'brain_hard':
            underlying = 'mouse_Brain'
            tissue_train = pd.read_csv(self.path+'/brain_train.csv', header=None)
            tissue_test = pd.read_csv(self.path+'/brain_test.csv', header=None)
            genes = pd.read_csv(self.path+'/brain_w2v_genes.csv', header=None)
            y_values_train = pd.read_csv(self.path+'/brain_train_labels.csv', skiprows=1, header=None, dtype=str)
            y_values_test = pd.read_csv(self.path+'/brain_test_labels.csv', skiprows=1, header=None, dtype=str)
            normalized_train = pd.read_csv(self.path+'/normalized_brain_train.csv', header=0, index_col=0)
            normalized_test = pd.read_csv(self.path+'/normalized_brain_test.csv', header=0, index_col=0)

            # raw inputs
            brain_train = pd.concat([pd.read_csv(self.path+'/train/mouse/mouse_Brain753_data.csv', header=0, index_col=0),
                                 pd.read_csv(self.path+'/train/mouse/mouse_Brain3285_data.csv', header=0, index_col=0)],axis=1, ignore_index=False)
            brain_test = pd.read_csv(self.path+'/test/mouse/mouse_Brain2695_data.csv', header=0, index_col=0)
            
            brain_hard_bce_train = pd.read_csv(self.path+"/ft_y_"+underlying+"_train.csv", header=None)
            brain_hard_bce_test = pd.read_csv(self.path+"/ft_y_"+underlying+"_test.csv", header=None)
            
            tissue_train = pd.concat([tissue_train, brain_hard_bce_train], axis=1)
            tissue_test = pd.concat([tissue_test, brain_hard_bce_test], axis=1)
            

            return tissue_train, tissue_test, genes, y_values_train, y_values_test, normalized_train, normalized_test, brain_train, brain_test

        elif dataset == 'brain_soft':
            underlying = 'mouse_Brain'
            tissue_train = pd.read_csv(self.path+'/brain_train.csv', header=None)
            tissue_test = pd.read_csv(self.path+'/brain_test.csv', header=None)
            genes = pd.read_csv(self.path+'/brain_w2v_genes.csv', header=None)
            y_values_train = pd.read_csv(self.path+'/brain_train_labels.csv', skiprows=1, header=None, dtype=str)
            y_values_test = pd.read_csv(self.path+'/brain_test_labels.csv', skiprows=1, header=None, dtype=str)
            normalized_train = pd.read_csv(self.path+'/normalized_brain_train.csv', header=0, index_col=0)
            normalized_test = pd.read_csv(self.path+'/normalized_brain_test.csv', header=0, index_col=0)

            # raw inputs
            brain_train = pd.concat([pd.read_csv(self.path+'/train/mouse/mouse_Brain753_data.csv', header=0, index_col=0),
                                 pd.read_csv(self.path+'/train/mouse/mouse_Brain3285_data.csv', header=0, index_col=0)],axis=1, ignore_index=False)
            brain_test = pd.read_csv(self.path+'/test/mouse/mouse_Brain2695_data.csv', header=0, index_col=0)
            
            brain_soft_bce_train = pd.read_csv(self.path+"/y_hat_"+underlying+"_train.csv", header=None)
            brain_soft_bce_test = pd.read_csv(self.path+"/y_hat_"+underlying+"_test.csv", header=None)
            
            train_targets_df.read_csv(data_dir_+"/tuned_"+underlying+"_train_labels.csv", skiprows=1, header=None, dtype=str)
            test_targets_df.read_csv(data_dir_+"/tuned_"+underlying+"_test_labels.csv", skiprows=1, header=None, dtype=str)
            
            y_values_train_list = y_values_train[0].tolist()
            y_values_test_list = y_values_test[0].tolist()
            
            train_targets_df.reset_index(drop=True, inplace=True)
            test_targets_df.reset_index(drop=True, inplace=True)
            train_targets_df = train_targets_df.iloc[y_values_train_list]
            test_targets_df = test_targets_df.iloc[y_values_test_list]
            
            brain_soft_bce_train = brain_soft_bce_train.loc[train_targets_df.index].reset_index(drop=True)
            brain_soft_bce_test = brain_soft_bce_test.loc[test_targets_df.index].reset_index(drop=True)
            
            tissue_train = pd.concat([tissue_train, brain_soft_bce_train], axis=1)
            tissue_test = pd.concat([tissue_test, brain_soft_bce_test], axis=1)

            return tissue_train, tissue_test, genes, y_values_train, y_values_test, normalized_train, normalized_test, brain_train, brain_test
        
        elif dataset == 'tuned_brain_hard':
            underlying = 'mouse_Brain'
            tissue_train = pd.read_csv(self.path+"/tuned_"+underlying+"_hard_train.csv", header=None)
            tissue_test = pd.read_csv(self.path+"/tuned_"+underlying+"_hard_test.csv", header=None)
            genes = pd.read_csv(self.path+'/brain_w2v_genes.csv', header=None)
            y_values_train = pd.read_csv(self.path+'/brain_train_labels.csv', skiprows=1, header=None, dtype=str)
            y_values_test = pd.read_csv(self.path+'/brain_test_labels.csv', skiprows=1, header=None, dtype=str)
            normalized_train = pd.read_csv(self.path+'/normalized_brain_train.csv', header=0, index_col=0)
            normalized_test = pd.read_csv(self.path+'/normalized_brain_test.csv', header=0, index_col=0)
            
            train_y = pd.read_csv(self.path+"/tuned_"+underlying+"_hard_train_labels.csv", skiprows=1, header=None, dtype=str)
            test_y = pd.read_csv(self.path+"/tuned_"+underlying+"_hard_test_labels.csv", skiprows=1, header=None, dtype=str)
            targets_train_df = train_y
            targets_test_df = test_y
            
            y_values_train_list = y_values_train[0].tolist()
            y_values_test_list = y_values_test[0].tolist()
            
            train_targets_df.reset_index(drop=True, inplace=True)
            test_targets_df.reset_index(drop=True, inplace=True)
            train_targets_df = train_targets_df.iloc[y_values_train_list]
            test_targets_df = test_targets_df.iloc[y_values_test_list]
            
            normalized_train = normalized_train.loc[train_targets_df.index].reset_index(drop=True)
            normalized_test = normalized_test.loc[test_targets_df.index].reset_index(drop=True)
            
            # raw inputs

            brain_train = pd.concat([pd.read_csv(self.path+'/train/mouse/mouse_Brain753_data.csv', header=0, index_col=0),
                                 pd.read_csv(self.path+'/train/mouse/mouse_Brain3285_data.csv', header=0, index_col=0)],axis=1, ignore_index=False)
            brain_test = pd.read_csv(self.path+'/test/mouse/mouse_Brain2695_data.csv', header=0, index_col=0)

            return tissue_train, tissue_test, genes, train_y, test_y, normalized_train, normalized_test, brain_train, brain_test
        
        elif dataset == 'tuned_brain_soft':
            underlying = 'mouse_Brain'
            tissue_train = pd.read_csv(self.path+"/tuned_"+underlying+"_train.csv", header=None)
            tissue_test = pd.read_csv(self.path+"/tuned_"+underlying+"_test.csv", header=None)
            genes = pd.read_csv(self.path+'/brain_w2v_genes.csv', header=None)
            y_values_train = pd.read_csv(self.path+'/brain_train_labels.csv', skiprows=1, header=None, dtype=str)
            y_values_test = pd.read_csv(self.path+'/brain_test_labels.csv', skiprows=1, header=None, dtype=str)
            normalized_train = pd.read_csv(self.path+'/normalized_brain_train.csv', header=0, index_col=0)
            normalized_test = pd.read_csv(self.path+'/normalized_brain_test.csv', header=0, index_col=0)
            
            train_y = pd.read_csv(self.path+"/tuned_"+underlying+"_train_labels.csv", skiprows=1, header=None, dtype=str)
            test_y = pd.read_csv(self.path+"/tuned_"+underlying+"_test_labels.csv", skiprows=1, header=None, dtype=str)
            targets_train_df = train_y
            targets_test_df = test_y
            
            y_values_train_list = y_values_train[0].tolist()
            y_values_test_list = y_values_test[0].tolist()
            
            train_targets_df.reset_index(drop=True, inplace=True)
            test_targets_df.reset_index(drop=True, inplace=True)
            train_targets_df = train_targets_df.iloc[y_values_train_list]
            test_targets_df = test_targets_df.iloc[y_values_test_list]
            
            normalized_train = normalized_train.loc[train_targets_df.index].reset_index(drop=True)
            normalized_test = normalized_test.loc[test_targets_df.index].reset_index(drop=True)
            
            # raw inputs

            brain_train = pd.concat([pd.read_csv(self.path+'/train/mouse/mouse_Brain753_data.csv', header=0, index_col=0),
                                 pd.read_csv(self.path+'/train/mouse/mouse_Brain3285_data.csv', header=0, index_col=0)],axis=1, ignore_index=False)
            brain_test = pd.read_csv(self.path+'/test/mouse/mouse_Brain2695_data.csv', header=0, index_col=0)

            return tissue_train, tissue_test, genes, train_y, test_y, normalized_train, normalized_test, brain_train, brain_test
        
        elif dataset == 'kidney_hard':
            underlying = 'mouse_Kidney'
            tissue_train = pd.read_csv(self.path+"/kidney_train.csv", header=None)
            tissue_test = pd.read_csv(self.path+"/kidney_test.csv", header=None)
            genes = pd.read_csv(self.path+'/kidney_w2v_genes.csv', header=None)
            y_values_train = pd.read_csv(self.path+'/kidney_train_labels.csv', skiprows=1, header=None, dtype=str)
            y_values_test = pd.read_csv(self.path+'/kidney_test_labels.csv', skiprows=1, header=None, dtype=str)
            normalized_train = pd.read_csv(self.path+'/normalized_kidney_train.csv', header=0, index_col=0)
            normalized_test = pd.read_csv(self.path+'/normalized_kidney_test.csv', header=0, index_col=0)

            kidney_hard_bce_train = pd.read_csv(self.path+"/ft_y_"+underlying+"_train.csv", header=None)
            kidney_hard_bce_test = pd.read_csv(self.path+"/ft_y_"+underlying+"_test.csv", header=None)

            tissue_train = pd.concat([tissue_train, kidney_hard_bce_train], axis=1)
            tissue_test = pd.concat([tissue_test, kidney_hard_bce_test], axis=1)

            kidney_train = pd.read_csv(self.path+'/train/mouse/mouse_Kidney4682_data.csv', header=0, index_col=0).fillna(0)
            kidney_test = pd.read_csv(self.path+'/test/mouse/mouse_Kidney203_data.csv', header=0, index_col=0).fillna(0)

            return tissue_train, tissue_test, genes, y_values_train, y_values_test, normalized_train, normalized_test, kidney_train, kidney_test
        
        elif dataset == 'kidney_soft':
            underlying = 'mouse_Kidney'
            tissue_train = pd.read_csv(self.path+"/kidney_train.csv", header=None)
            tissue_test = pd.read_csv(self.path+"/kidney_test.csv", header=None)
            genes = pd.read_csv(self.path+'/kidney_w2v_genes.csv', header=None)
            y_values_train = pd.read_csv(self.path+'/kidney_train_labels.csv', skiprows=1, header=None, dtype=str)
            y_values_test = pd.read_csv(self.path+'/kidney_test_labels.csv', skiprows=1, header=None, dtype=str)
            normalized_train = pd.read_csv(self.path+'/normalized_kidney_train.csv', header=0, index_col=0)
            normalized_test = pd.read_csv(self.path+'/normalized_kidney_test.csv', header=0, index_col=0)

            kidney_soft_bce_train = pd.read_csv(self.path+"/y_hat_"+underlying+"_train.csv", header=None)
            kidney_soft_bce_test = pd.read_csv(self.path+"/y_hat_"+underlying+"_test.csv", header=None)
            
            train_targets_df = pd.read_csv(self.path+"/tuned_"+underlying+"_train_labels.csv", skiprows=1, header=None, dtype=str)
            test_targets_df = pd.read_csv(self.path+"/tuned_"+underlying+"_test_labels.csv", skiprows=1, header=None, dtype=str)
            
            y_values_train_list = y_values_train[0].tolist()
            y_values_test_list = y_values_test[0].tolist()
            
            train_targets_df.reset_index(drop=True, inplace=True)
            test_targets_df.reset_index(drop=True, inplace=True)
            train_targets_df = train_targets_df.iloc[y_values_train_list]
            test_targets_df = test_targets_df.iloc[y_values_test_list]
            
            kidney_soft_bce_train = kidney_soft_bce_train.loc[train_targets_df.index].reset_index(drop=True)
            kidney_soft_bce_test = kidney_soft_bce_test.loc[test_targets_df.index].reset_index(drop=True)
            
            tissue_train = pd.concat([tissue_train, kidney_soft_bce_train], axis=1)
            tissue_test = pd.concat([tissue_test, kidney_soft_bce_test], axis=1)

            kidney_train = pd.read_csv(self.path+'/train/mouse/mouse_Kidney4682_data.csv', header=0, index_col=0).fillna(0)
            kidney_test = pd.read_csv(self.path+'/test/mouse/mouse_Kidney203_data.csv', header=0, index_col=0).fillna(0)

            return tissue_train, tissue_test, genes, y_values_train, y_values_test, normalized_train, normalized_test, kidney_train, kidney_test

        elif dataset == 'tuned_kidney_hard':
            underlying = 'mouse_Kidney'
            tissue_train = pd.read_csv(self.path+"/tuned_"+underlying+"_hard_train.csv", header=None)
            tissue_test = pd.read_csv(self.path+"/tuned_"+underlying+"_hard_test.csv", header=None)
            genes = pd.read_csv(self.path+'/kidney_w2v_genes.csv', header=None)
            y_values_train = pd.read_csv(self.path+'/kidney_train_labels.csv', skiprows=1, header=None, dtype=str)
            y_values_test = pd.read_csv(self.path+'/kidney_test_labels.csv', skiprows=1, header=None, dtype=str)
            normalized_train = pd.read_csv(self.path+'/normalized_kidney_train.csv', header=0, index_col=0)
            normalized_test = pd.read_csv(self.path+'/normalized_kidney_test.csv', header=0, index_col=0)

            train_y = pd.read_csv(self.path+"/tuned_"+underlying+"_hard_train_labels.csv", skiprows=1, header=None, dtype=str)
            test_y = pd.read_csv(self.path+"/tuned_"+underlying+"_hard_test_labels.csv", skiprows=1, header=None, dtype=str)
            
            y_values_train_list = y_values_train[0].tolist()
            y_values_test_list = y_values_test[0].tolist()
            
            train_targets_df = train_y
            test_targets_df = test_y
            
            train_targets_df.reset_index(drop=True, inplace=True)
            test_targets_df.reset_index(drop=True, inplace=True)
            train_targets_df = train_targets_df.iloc[y_values_train_list]
            test_targets_df = test_targets_df.iloc[y_values_test_list]
            
            normalized_train = normalized_train.loc[train_targets_df.index].reset_index(drop=True)
            normalized_test = normalized_test.loc[test_targets_df.index].reset_index(drop=True)
            
            kidney_train = pd.read_csv(self.path+'/train/mouse/mouse_Kidney4682_data.csv', header=0, index_col=0).fillna(0)
            kidney_test = pd.read_csv(self.path+'/test/mouse/mouse_Kidney203_data.csv', header=0, index_col=0).fillna(0)

            return tissue_train, tissue_test, genes, train_y, test_y, normalized_train, normalized_test, kidney_train, kidney_test
        
        elif dataset == 'tuned_kidney_soft':
            underlying = 'mouse_Kidney'
            tissue_train = pd.read_csv(self.path+"/tuned_"+underlying+"_train.csv", header=None)
            tissue_test = pd.read_csv(self.path+"/tuned_"+underlying+"_test.csv", header=None)
            genes = pd.read_csv(self.path+'/kidney_w2v_genes.csv', header=None)
            y_values_train = pd.read_csv(self.path+'/kidney_train_labels.csv', skiprows=1, header=None, dtype=str)
            y_values_test = pd.read_csv(self.path+'/kidney_test_labels.csv', skiprows=1, header=None, dtype=str)
            normalized_train = pd.read_csv(self.path+'/normalized_kidney_train.csv', header=0, index_col=0)
            normalized_test = pd.read_csv(self.path+'/normalized_kidney_test.csv', header=0, index_col=0)

            train_y = pd.read_csv(self.path+"/tuned_"+underlying+"_train_labels.csv", skiprows=1, header=None, dtype=str)
            test_y = pd.read_csv(self.path+"/tuned_"+underlying+"_test_labels.csv", skiprows=1, header=None, dtype=str)
            
            y_values_train_list = y_values_train[0].tolist()
            y_values_test_list = y_values_test[0].tolist()
            
            train_targets_df = train_y
            test_targets_df = test_y
            
            train_targets_df.reset_index(drop=True, inplace=True)
            test_targets_df.reset_index(drop=True, inplace=True)
            train_targets_df = train_targets_df.iloc[y_values_train_list]
            test_targets_df = test_targets_df.iloc[y_values_test_list]
            
            normalized_train = normalized_train.loc[train_targets_df.index].reset_index(drop=True)
            normalized_test = normalized_test.loc[test_targets_df.index].reset_index(drop=True)
            
            kidney_train = pd.read_csv(self.path+'/train/mouse/mouse_Kidney4682_data.csv', header=0, index_col=0).fillna(0)
            kidney_test = pd.read_csv(self.path+'/test/mouse/mouse_Kidney203_data.csv', header=0, index_col=0).fillna(0)

            return tissue_train, tissue_test, genes, train_y, test_y, normalized_train, normalized_test, kidney_train, kidney_test
        
        elif dataset == 'pancreas_hard':
            underlying = 'human_Pancreas'
            tissue_train = pd.read_csv(self.path+"/pancreas_train.csv", header=None)
            tissue_test = pd.read_csv(self.path+"/pancreas_test.csv", header=None)
            genes = pd.read_csv(self.path+'/pancreas_w2v_genes.csv', header=None)
            y_values_train = pd.read_csv(self.path+'/pancreas_train_labels.csv', skiprows=1, header=None, dtype=str)
            y_values_test = pd.read_csv(self.path+'/pancreas_test_labels.csv', skiprows=1, header=None, dtype=str)
            normalized_train = pd.read_csv(self.path+'/normalized_pancreas_train.csv', header=0, index_col=0)
            normalized_test = pd.read_csv(self.path+'/normalized_pancreas_test.csv', header=0, index_col=0)

            pancreas_hard_bce_train = pd.read_csv(self.path+"/ft_y_"+underlying+"_train.csv", header=None)
            pancreas_hard_bce_test = pd.read_csv(self.path+"/ft_y_"+underlying+"_test.csv", header=None)

            tissue_train = pd.concat([tissue_train, pancreas_hard_bce_train], axis=1)
            tissue_test = pd.concat([tissue_test, pancreas_hard_bce_test], axis=1)

            pancreas_train = pd.read_csv(self.path+'/train/human/human_cell_atlas/human_Pancreas9727_data.csv', header=0, index_col=0)
            pancreas_test = pd.concat([pd.read_csv(self.path+'/test/human/human_test_data/human_Pancreas2227_data.csv', header=0, index_col=0),
                                 pd.read_csv(self.path+'/test/human/human_test_data/human_Pancreas1841_data.csv', header=0, index_col=0)],axis=1, ignore_index=False)

            return tissue_train, tissue_test, genes, y_values_train, y_values_test, normalized_train, normalized_test, pancreas_train, pancreas_test
        
        elif dataset == 'pancreas_soft':
            underlying = 'human_Pancreas'
            tissue_train = pd.read_csv(self.path+"/pancreas_train.csv", header=None)
            tissue_test = pd.read_csv(self.path+"/pancreas_test.csv", header=None)
            genes = pd.read_csv(self.path+'/pancreas_w2v_genes.csv', header=None)
            y_values_train = pd.read_csv(self.path+'/pancreas_train_labels.csv', skiprows=1, header=None, dtype=str)
            y_values_test = pd.read_csv(self.path+'/pancreas_test_labels.csv', skiprows=1, header=None, dtype=str)
            normalized_train = pd.read_csv(self.path+'/normalized_pancreas_train.csv', header=0, index_col=0)
            normalized_test = pd.read_csv(self.path+'/normalized_pancreas_test.csv', header=0, index_col=0)

            pancreas_soft_bce_train = pd.read_csv(self.path+"/y_hat_"+underlying+"_train.csv", header=None)
            pancreas_soft_bce_test = pd.read_csv(self.path+"/y_hat_"+underlying+"_test.csv", header=None)
            
            train_targets_df = pd.read_csv(self.path+"/tuned_"+underlying+"_train_labels.csv", skiprows=1, header=None, dtype=str)
            test_targets_df = pd.read_csv(self.path+"/tuned_"+underlying+"_test_labels.csv", skiprows=1, header=None, dtype=str)
            
            y_values_train_list = y_values_train[0].tolist()
            y_values_test_list = y_values_test[0].tolist()
            
            train_targets_df.reset_index(drop=True, inplace=True)
            test_targets_df.reset_index(drop=True, inplace=True)
            train_targets_df = train_targets_df.iloc[y_values_train_list]
            test_targets_df = test_targets_df.iloc[y_values_test_list]
            
            pancreas_soft_bce_train = pancreas_soft_bce_train.loc[train_targets_df.index].reset_index(drop=True)
            pancreas_soft_bce_test = pancreas_soft_bce_test.loc[test_targets_df.index].reset_index(drop=True)
            
            tissue_train = pd.concat([tissue_train, pancreas_soft_bce_train], axis=1)
            tissue_test = pd.concat([tissue_test, pancreas_soft_bce_test], axis=1)

            pancreas_train = pd.read_csv(self.path+'/train/human/human_cell_atlas/human_Pancreas9727_data.csv', header=0, index_col=0)
            pancreas_test = pd.concat([pd.read_csv(self.path+'/test/human/human_test_data/human_Pancreas2227_data.csv', header=0, index_col=0),
                                 pd.read_csv(self.path+'/test/human/human_test_data/human_Pancreas1841_data.csv', header=0, index_col=0)],axis=1, ignore_index=False)

            return tissue_train, tissue_test, genes, y_values_train, y_values_test, normalized_train, normalized_test, pancreas_train, pancreas_test
        
        elif dataset == 'tuned_pancreas_hard':
            underlying = 'human_Pancreas'
            tissue_train = pd.read_csv(self.path+"/tuned_"+underlying+"_hard_train.csv", header=None)
            tissue_test = pd.read_csv(self.path+"/tuned_"+underlying+"_hard_test.csv", header=None)
            genes = pd.read_csv(self.path+'/pancreas_w2v_genes.csv', header=None)
            y_values_train = pd.read_csv(self.path+'/pancreas_train_labels.csv', skiprows=1, header=None, dtype=str)
            y_values_test = pd.read_csv(self.path+'/pancreas_test_labels.csv', skiprows=1, header=None, dtype=str)
            normalized_train = pd.read_csv(self.path+'/normalized_pancreas_train.csv', header=0, index_col=0)
            normalized_test = pd.read_csv(self.path+'/normalized_pancreas_test.csv', header=0, index_col=0)
            
            train_y = pd.read_csv(self.path+"/tuned_"+underlying+"_hard_train_labels.csv", skiprows=1, header=None, dtype=str)
            test_y = pd.read_csv(self.path+"/tuned_"+underlying+"_hard_test_labels.csv", skiprows=1, header=None, dtype=str)
            
            y_values_train_list = y_values_train[0].tolist()
            y_values_test_list = y_values_test[0].tolist()
            
            train_targets_df = train_y
            test_targets_df = test_y
            
            train_targets_df.reset_index(drop=True, inplace=True)
            test_targets_df.reset_index(drop=True, inplace=True)
            train_targets_df = train_targets_df.iloc[y_values_train_list]
            test_targets_df = test_targets_df.iloc[y_values_test_list]
            
            normalized_train = normalized_train.loc[train_targets_df.index].reset_index(drop=True)
            normalized_test = normalized_test.loc[test_targets_df.index].reset_index(drop=True)
            
            pancreas_train = pd.read_csv(self.path+'/train/human/human_cell_atlas/human_Pancreas9727_data.csv', header=0, index_col=0)
            pancreas_test = pd.concat([pd.read_csv(self.path+'/test/human/human_test_data/human_Pancreas2227_data.csv', header=0, index_col=0),
                                 pd.read_csv(self.path+'/test/human/human_test_data/human_Pancreas1841_data.csv', header=0, index_col=0)],axis=1, ignore_index=False)

            return tissue_train, tissue_test, genes, train_y, test_y, normalized_train, normalized_test, pancreas_train, pancreas_test
        
        elif dataset == 'tuned_pancreas_soft':
            underlying = 'human_Pancreas'
            tissue_train = pd.read_csv(self.path+"/tuned_"+underlying+"_train.csv", header=None)
            tissue_test = pd.read_csv(self.path+"/tuned_"+underlying+"_test.csv", header=None)
            genes = pd.read_csv(self.path+'/pancreas_w2v_genes.csv', header=None)
            y_values_train = pd.read_csv(self.path+'/pancreas_train_labels.csv', skiprows=1, header=None, dtype=str)
            y_values_test = pd.read_csv(self.path+'/pancreas_test_labels.csv', skiprows=1, header=None, dtype=str)
            normalized_train = pd.read_csv(self.path+'/normalized_pancreas_train.csv', header=0, index_col=0)
            normalized_test = pd.read_csv(self.path+'/normalized_pancreas_test.csv', header=0, index_col=0)

            train_y = pd.read_csv(self.path+"/tuned_"+underlying+"_train_labels.csv", skiprows=1, header=None, dtype=str)
            test_y = pd.read_csv(self.path+"/tuned_"+underlying+"_test_labels.csv", skiprows=1, header=None, dtype=str)
            
            y_values_train_list = y_values_train[0].tolist()
            y_values_test_list = y_values_test[0].tolist()
            
            train_targets_df = train_y
            test_targets_df = test_y
            
            train_targets_df.reset_index(drop=True, inplace=True)
            test_targets_df.reset_index(drop=True, inplace=True)
            train_targets_df = train_targets_df.iloc[y_values_train_list]
            test_targets_df = test_targets_df.iloc[y_values_test_list]
            
            normalized_train = normalized_train.loc[train_targets_df.index].reset_index(drop=True)
            normalized_test = normalized_test.loc[test_targets_df.index].reset_index(drop=True)
            
            pancreas_train = pd.read_csv(self.path+'/train/human/human_cell_atlas/human_Pancreas9727_data.csv', header=0, index_col=0)
            pancreas_test = pd.concat([pd.read_csv(self.path+'/test/human/human_test_data/human_Pancreas2227_data.csv', header=0, index_col=0),
                                 pd.read_csv(self.path+'/test/human/human_test_data/human_Pancreas1841_data.csv', header=0, index_col=0)],axis=1, ignore_index=False)

            return tissue_train, tissue_test, genes, train_y, test_y, normalized_train, normalized_test, pancreas_train, pancreas_test

        elif dataset == 'human_spleen_hard':
            underlying = 'human_Spleen'
            tissue_train = pd.read_csv(self.path+'/human_spleen_train.csv', header=None)
            tissue_test = pd.read_csv(self.path+'/human_spleen_test.csv', header=None)
            genes = pd.read_csv(self.path+'/human_spleen_w2v_genes.csv', header=None)
            y_values_train = pd.read_csv(self.path+'/human_spleen_train_labels.csv', skiprows=1, header=None, dtype=str)
            y_values_test = pd.read_csv(self.path+'/human_spleen_test_labels.csv', skiprows=1, header=None, dtype=str)
            normalized_train = pd.read_csv(self.path+'/normalized_human_spleen_train.csv', header=0, index_col=0)
            normalized_test = pd.read_csv(self.path+'/normalized_human_spleen_test.csv', header=0, index_col=0)
            
            spleen_hard_bce_train = pd.read_csv(self.path+"/ft_y_"+underlying+"_train.csv", header=None)
            spleen_hard_bce_test = pd.read_csv(self.path+"/ft_y_"+underlying+"_test.csv", header=None)
            
            tissue_train = pd.concat([tissue_train, spleen_hard_bce_train], axis=1)
            tissue_test = pd.concat([tissue_test, spleen_hard_bce_test], axis=1)

            # raw inputs
            spleen_train = pd.read_csv(self.path+'/train/human/human_cell_atlas/human_Spleen15806_data.csv', header=0, index_col=0)
            spleen_test = pd.read_csv(self.path+'/test/human/human_test_data/human_Spleen9887_data.csv', header=0, index_col=0)

            return tissue_train, tissue_test, genes, y_values_train, y_values_test, normalized_train, normalized_test, spleen_train, spleen_test
        
        elif dataset == 'human_spleen_soft':
            underlying = 'human_Spleen'
            tissue_train = pd.read_csv(self.path+"/human_spleen_train.csv", header=None)
            tissue_test = pd.read_csv(self.path+"/human_spleen_test.csv", header=None)
            genes = pd.read_csv(self.path+'/pancreas_w2v_genes.csv', header=None)
            y_values_train = pd.read_csv(self.path+'/human_spleen_train_labels.csv', skiprows=1, header=None, dtype=str)
            y_values_test = pd.read_csv(self.path+'/human_spleen_test_labels.csv', skiprows=1, header=None, dtype=str)
            normalized_train = pd.read_csv(self.path+'/normalized_human_spleen_train.csv', header=0, index_col=0)
            normalized_test = pd.read_csv(self.path+'/normalized_human_spleen_test.csv', header=0, index_col=0)

            human_spleen_soft_bce_train = pd.read_csv(self.path+"/y_hat_"+underlying+"_train.csv", header=None)
            human_spleen_soft_bce_test = pd.read_csv(self.path+"/y_hat_"+underlying+"_test.csv", header=None)
            
            train_targets_df = pd.read_csv(self.path+"/tuned_"+underlying+"_train_labels.csv", skiprows=1, header=None, dtype=str)
            test_targets_df = pd.read_csv(self.path+"/tuned_"+underlying+"_test_labels.csv", skiprows=1, header=None, dtype=str)
            
            y_values_train_list = y_values_train[0].tolist()
            y_values_test_list = y_values_test[0].tolist()
            
            train_targets_df.reset_index(drop=True, inplace=True)
            test_targets_df.reset_index(drop=True, inplace=True)
            train_targets_df = train_targets_df.iloc[y_values_train_list]
            test_targets_df = test_targets_df.iloc[y_values_test_list]
            
            human_spleen_soft_bce_train = human_spleen_soft_bce_train.loc[train_targets_df.index].reset_index(drop=True)
            human_spleen_soft_bce_test = human_spleen_soft_bce_test.loc[test_targets_df.index].reset_index(drop=True)
            
            tissue_train = pd.concat([tissue_train, human_spleen_soft_bce_train], axis=1)
            tissue_test = pd.concat([tissue_test, human_spleen_soft_bce_test], axis=1)

            spleen_train = pd.read_csv(self.path+'/train/human/human_cell_atlas/human_Spleen15806_data.csv', header=0, index_col=0)
            spleen_test = pd.read_csv(self.path+'/test/human/human_test_data/human_Spleen9887_data.csv', header=0, index_col=0)

            return tissue_train, tissue_test, genes, y_values_train, y_values_test, normalized_train, normalized_test, spleen_train, spleen_test
        
        elif dataset == 'tuned_human_spleen_hard':
            underlying = 'human_Spleen'
            tissue_train = pd.read_csv(self.path+"/tuned_"+underlying+"_hard_train.csv", header=None)
            tissue_test = pd.read_csv(self.path+"/tuned_"+underlying+"_hard_test.csv", header=None)
            genes = pd.read_csv(self.path+'/human_spleen_w2v_genes.csv', header=None)
            y_values_train = pd.read_csv(self.path+'/human_spleen_train_labels.csv', skiprows=1, header=None, dtype=str)
            y_values_test = pd.read_csv(self.path+'/human_spleen_test_labels.csv', skiprows=1, header=None, dtype=str)
            normalized_train = pd.read_csv(self.path+'/normalized_human_spleen_train.csv', header=0, index_col=0)
            normalized_test = pd.read_csv(self.path+'/normalized_human_spleen_test.csv', header=0, index_col=0)
            
            train_y = pd.read_csv(self.path+"/tuned_"+underlying+"_hard_train_labels.csv", skiprows=1, header=None, dtype=str)
            test_y = pd.read_csv(self.path+"/tuned_"+underlying+"_hard_test_labels.csv", skiprows=1, header=None, dtype=str)
            
            y_values_train_list = y_values_train[0].tolist()
            y_values_test_list = y_values_test[0].tolist()
            
            train_targets_df = train_y
            test_targets_df = test_y
            
            train_targets_df.reset_index(drop=True, inplace=True)
            test_targets_df.reset_index(drop=True, inplace=True)
            train_targets_df = train_targets_df.iloc[y_values_train_list]
            test_targets_df = test_targets_df.iloc[y_values_test_list]
            
            normalized_train = normalized_train.loc[train_targets_df.index].reset_index(drop=True)
            normalized_test = normalized_test.loc[test_targets_df.index].reset_index(drop=True)

            # raw inputs
            spleen_train = pd.read_csv(self.path+'/train/human/human_cell_atlas/human_Spleen15806_data.csv', header=0, index_col=0)
            spleen_test = pd.read_csv(self.path+'/test/human/human_test_data/human_Spleen9887_data.csv', header=0, index_col=0)

            return tissue_train, tissue_test, genes, train_y, test_y, normalized_train, normalized_test, spleen_train, spleen_test

        elif dataset == 'tuned_human_spleen_soft':
            underlying = 'human_Spleen'
            tissue_train = pd.read_csv(self.path+"/tuned_"+underlying+"_train.csv", header=None)
            tissue_test = pd.read_csv(self.path+"/tuned_"+underlying+"_test.csv", header=None)
            genes = pd.read_csv(self.path+'/human_spleen_w2v_genes.csv', header=None)
            y_values_train = pd.read_csv(self.path+'/human_spleen_train_labels.csv', skiprows=1, header=None, dtype=str)
            y_values_test = pd.read_csv(self.path+'/human_spleen_test_labels.csv', skiprows=1, header=None, dtype=str)
            normalized_train = pd.read_csv(self.path+'/normalized_human_spleen_train.csv', header=0, index_col=0)
            normalized_test = pd.read_csv(self.path+'/normalized_human_spleen_test.csv', header=0, index_col=0)
            
            train_y = pd.read_csv(self.path+"/tuned_"+underlying+"_train_labels.csv", skiprows=1, header=None, dtype=str)
            test_y = pd.read_csv(self.path+"/tuned_"+underlying+"_test_labels.csv", skiprows=1, header=None, dtype=str)
            
            y_values_train_list = y_values_train[0].tolist()
            y_values_test_list = y_values_test[0].tolist()
            
            train_targets_df = train_y
            test_targets_df = test_y
            
            train_targets_df.reset_index(drop=True, inplace=True)
            test_targets_df.reset_index(drop=True, inplace=True)
            train_targets_df = train_targets_df.iloc[y_values_train_list]
            test_targets_df = test_targets_df.iloc[y_values_test_list]
            
            normalized_train = normalized_train.loc[train_targets_df.index].reset_index(drop=True)
            normalized_test = normalized_test.loc[test_targets_df.index].reset_index(drop=True)

            # raw inputs
            spleen_train = pd.read_csv(self.path+'/train/human/human_cell_atlas/human_Spleen15806_data.csv', header=0, index_col=0)
            spleen_test = pd.read_csv(self.path+'/test/human/human_test_data/human_Spleen9887_data.csv', header=0, index_col=0)

            return tissue_train, tissue_test, genes, train_y, test_y, normalized_train, normalized_test, spleen_train, spleen_test

        elif dataset == 'bone_hard':
            underlying = 'human_Bonemarrow'
            tissue_train = pd.read_csv(self.path+'/bone_train.csv', header=None)
            tissue_test = pd.read_csv(self.path+'/bone_test.csv', header=None)
            genes = pd.read_csv(self.path+'/bone_w2v_genes.csv', header=None)
            y_values_train = pd.read_csv(self.path+'/bone_train_labels.csv', skiprows=1, header=None, dtype=str)
            y_values_test = pd.read_csv(self.path+'/bone_test_labels.csv', skiprows=1, header=None, dtype=str)
            normalized_train = pd.read_csv(self.path+'/normalized_bone_train.csv', header=0, index_col=0)
            normalized_test = pd.read_csv(self.path+'/normalized_bone_test.csv', header=0, index_col=0)

            bone_hard_bce_train = pd.read_csv(self.path+"/ft_y_"+underlying+"_train.csv", header=None)
            bone_hard_bce_test = pd.read_csv(self.path+"/ft_y_"+underlying+"_test.csv", header=None)
            
            tissue_train = pd.concat([tissue_train, bone_hard_bce_train], axis=1)
            tissue_test = pd.concat([tissue_test, bone_hard_bce_test], axis=1)

            # raw inputs
            bonemarrow_train = pd.read_csv(self.path+'/train/human/human_cell_atlas/human_Bone_marrow2261_data.csv', header=0, index_col=0)
            bonemarrow_test = pd.read_csv(self.path+'/train/human/human_cell_atlas/human_Bone_marrow6443_data.csv', header=0, index_col=0)

            return tissue_train, tissue_test, genes, y_values_train, y_values_test, normalized_train, normalized_test, bonemarrow_train, bonemarrow_test

        elif dataset == 'bone_soft':
            underlying = 'human_Bonemarrow'
            tissue_train = pd.read_csv(self.path+"/bone_train.csv", header=None)
            tissue_test = pd.read_csv(self.path+"/bone_test.csv", header=None)
            genes = pd.read_csv(self.path+'/bone_w2v_genes.csv', header=None)
            y_values_train = pd.read_csv(self.path+'/bone_train_labels.csv', skiprows=1, header=None, dtype=str)
            y_values_test = pd.read_csv(self.path+'/bone_test_labels.csv', skiprows=1, header=None, dtype=str)
            normalized_train = pd.read_csv(self.path+'/normalized_bone_train.csv', header=0, index_col=0)
            normalized_test = pd.read_csv(self.path+'/normalized_bone_test.csv', header=0, index_col=0)

            bone_soft_bce_train = pd.read_csv(self.path+"/y_hat_"+underlying+"_train.csv", header=None)
            bone_soft_bce_test = pd.read_csv(self.path+"/y_hat_"+underlying+"_test.csv", header=None)
            
            train_targets_df = pd.read_csv(self.path+"/tuned_"+underlying+"_train_labels.csv", skiprows=1, header=None, dtype=str)
            test_targets_df = pd.read_csv(self.path+"/tuned_"+underlying+"_test_labels.csv", skiprows=1, header=None, dtype=str)
            
            y_values_train_list = y_values_train[0].tolist()
            y_values_test_list = y_values_test[0].tolist()
            
            train_targets_df.reset_index(drop=True, inplace=True)
            test_targets_df.reset_index(drop=True, inplace=True)
            train_targets_df = train_targets_df.iloc[y_values_train_list]
            test_targets_df = test_targets_df.iloc[y_values_test_list]
            
            bone_soft_bce_train = bone_soft_bce_train.loc[train_targets_df.index].reset_index(drop=True)
            bone_soft_bce_test = bone_soft_bce_test.loc[test_targets_df.index].reset_index(drop=True)
            
            tissue_train = pd.concat([tissue_train, bone_soft_bce_train], axis=1)
            tissue_test = pd.concat([tissue_test, bone_soft_bce_test], axis=1)

            # raw inputs
            bonemarrow_train = pd.read_csv(self.path+'/train/human/human_cell_atlas/human_Bone_marrow2261_data.csv', header=0, index_col=0)
            bonemarrow_test = pd.read_csv(self.path+'/train/human/human_cell_atlas/human_Bone_marrow6443_data.csv', header=0, index_col=0)

            return tissue_train, tissue_test, genes, y_values_train, y_values_test, normalized_train, normalized_test, bonemarrow_train, bonemarrow_test

        elif dataset == 'tuned_bone_hard':
            underlying = 'human_Bonemarrow'
            tissue_train = pd.read_csv(self.path+"/tuned_"+underlying+"_hard_train.csv", header=None)
            tissue_test = pd.read_csv(self.path+"/tuned_"+underlying+"_hard_test.csv", header=None)
            genes = pd.read_csv(self.path+'/bone_w2v_genes.csv', header=None)
            y_values_train = pd.read_csv(self.path+'/bone_train_labels.csv', skiprows=1, header=None, dtype=str)
            y_values_test = pd.read_csv(self.path+'/bone_test_labels.csv', skiprows=1, header=None, dtype=str)
            normalized_train = pd.read_csv(self.path+'/normalized_bone_train.csv', header=0, index_col=0)
            normalized_test = pd.read_csv(self.path+'/normalized_bone_test.csv', header=0, index_col=0)
            
            train_y = pd.read_csv(self.path+"/tuned_"+underlying+"_hard_train_labels.csv", skiprows=1, header=None, dtype=str)
            test_y = pd.read_csv(self.path+"/tuned_"+underlying+"_hard_test_labels.csv", skiprows=1, header=None, dtype=str)
            
            y_values_train_list = y_values_train[0].tolist()
            y_values_test_list = y_values_test[0].tolist()
            
            train_targets_df = train_y
            test_targets_df = test_y
            
            train_targets_df.reset_index(drop=True, inplace=True)
            test_targets_df.reset_index(drop=True, inplace=True)
            train_targets_df = train_targets_df.iloc[y_values_train_list]
            test_targets_df = test_targets_df.iloc[y_values_test_list]
            
            normalized_train = normalized_train.loc[train_targets_df.index].reset_index(drop=True)
            normalized_test = normalized_test.loc[test_targets_df.index].reset_index(drop=True)


            # raw inputs
            bonemarrow_train = pd.read_csv(self.path+'/train/human/human_cell_atlas/human_Bone_marrow2261_data.csv', header=0, index_col=0)
            bonemarrow_test = pd.read_csv(self.path+'/train/human/human_cell_atlas/human_Bone_marrow6443_data.csv', header=0, index_col=0)

            return tissue_train, tissue_test, genes, train_y, test_y, normalized_train, normalized_test, bonemarrow_train, bonemarrow_test

        elif dataset == 'tuned_bone_soft':
            underlying = 'human_Bonemarrow'
            tissue_train = pd.read_csv(self.path+"/tuned_"+underlying+"_train.csv", header=None)
            tissue_test = pd.read_csv(self.path+"/tuned_"+underlying+"_test.csv", header=None)
            genes = pd.read_csv(self.path+'/bone_w2v_genes.csv', header=None)
            y_values_train = pd.read_csv(self.path+'/bone_train_labels.csv', skiprows=1, header=None, dtype=str)
            y_values_test = pd.read_csv(self.path+'/bone_test_labels.csv', skiprows=1, header=None, dtype=str)
            normalized_train = pd.read_csv(self.path+'/normalized_bone_train.csv', header=0, index_col=0)
            normalized_test = pd.read_csv(self.path+'/normalized_bone_test.csv', header=0, index_col=0)
            
            train_y = pd.read_csv(self.path+"/tuned_"+underlying+"_train_labels.csv", skiprows=1, header=None, dtype=str)
            test_y = pd.read_csv(self.path+"/tuned_"+underlying+"_test_labels.csv", skiprows=1, header=None, dtype=str)
            
            y_values_train_list = y_values_train[0].tolist()
            y_values_test_list = y_values_test[0].tolist()
            
            train_targets_df = train_y
            test_targets_df = test_y
            
            train_targets_df.reset_index(drop=True, inplace=True)
            test_targets_df.reset_index(drop=True, inplace=True)
            train_targets_df = train_targets_df.iloc[y_values_train_list]
            test_targets_df = test_targets_df.iloc[y_values_test_list]
            
            normalized_train = normalized_train.loc[train_targets_df.index].reset_index(drop=True)
            normalized_test = normalized_test.loc[test_targets_df.index].reset_index(drop=True)

            # raw inputs
            bonemarrow_train = pd.read_csv(self.path+'/train/human/human_cell_atlas/human_Bone_marrow2261_data.csv', header=0, index_col=0)
            bonemarrow_test = pd.read_csv(self.path+'/train/human/human_cell_atlas/human_Bone_marrow6443_data.csv', header=0, index_col=0)

            return tissue_train, tissue_test, genes, train_y, test_y, normalized_train, normalized_test, bonemarrow_train, bonemarrow_test

        else:
            raise ValueError(f"Dataset '{dataset}' is not recognized.")
            
    def w2v_embed(self, corpus):
        '''
        Word2Vec Embeddings
        '''
        w2v_embed = Word2Vec(corpus, min_count=1, vector_size=self.dimensions, window=500)
        return w2v_embed
    
#     @torch.no_grad()
#     def bert_embed(self, x, corpus, y):
#         chunk_size = 256
#         tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
#         model = BertModel.from_pretrained("bert-base-uncased")
#         model = torch.nn.DataParallel(model)
#         model.to(self.device)
#         outputs = []
#         for sentence in corpus:
#             tokenized_text = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
#             tokenized_text = {key: value.to(self.device) for key, value in tokenized_text.items()}
#             chunk_outputs = model(**tokenized_text)
#             outputs.append(chunk_outputs)

#         print(outputs)
#         return outputs
# data = data_pre()