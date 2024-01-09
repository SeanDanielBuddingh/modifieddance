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
from transformers import BertTokenizer, BertModel



class data_pre():
    def __init__(self):
        self.seed = 42
        self.dimensions = 500

        current_script_path = __file__
        current_dir = os.path.dirname(current_script_path)
        parent_dir = os.path.dirname(current_dir)
        sys.path.append(parent_dir)
        parent_parent = os.path.dirname(parent_dir)
        parent_parent = parent_parent.replace("\\", "/")
        data_dir_ = parent_parent+'/dance_data'
        self.path = data_dir_
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        ##brain_train, brain_test, brain_train_labels, brain_test_labels, corpus_train_brain, spleen_train, spleen_test, spleen_train_labels, spleen_test_labels, corpus_train_spleen, kidney_train, kidney_test, kidney_train_labels, kidney_test_labels, corpus_train_kidney = self.load_data()
        brain_train, brain_test, brain_train_labels, brain_test_labels, corpus_train_brain, corpus_test_brain = self.load_data()
        ##self.get_w2v(brain_train, brain_test, brain_train_labels, brain_test_labels, corpus_train_brain, spleen_train, spleen_test, spleen_train_labels, spleen_test_labels, corpus_train_spleen, kidney_train, kidney_test, kidney_train_labels, kidney_test_labels, corpus_train_kidney)
        self.get_w2v(brain_train, brain_test, brain_train_labels, brain_test_labels, corpus_train_brain, corpus_test_brain)

        #self.bert_embed(brain_test, corpus_brain, brain_y)
        #self.bert_embed(spleen_x, corpus_spleen, spleen_y)
        #self.bert_embed(spleen_x, corpus_spleen, kidney_y)
        #self.bert_embed(brain_train, corpus_btrain, btrain_y)
        #self.bert_embed(s_train, corpus_strain, strain_y)
        #self.bert_embed(k_train, corpus_ktrain, ktrain_y)
        
    def load_data(self):
        '''
        Data Pre-Processing
        '''

        pca = PCA(n_components=self.dimensions)

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
        corpus_test_brain = []

        for i, g_name in enumerate(brain_train.iloc):
            prelim_brain_train = []
            gene_exp = brain_train.iloc[i, :]
            sorted = pd.DataFrame(gene_exp[gene_exp!=0].sort_values(ascending=False))
            #sorted = sorted.T
            for x, exp in enumerate(sorted.iloc):
                c_name = sorted.index.tolist()[x]
                prelim_brain_train.append(c_name)
            corpus_train_brain.append(prelim_brain_train)

        for i, g_name in enumerate(brain_test.iloc):
            prelim_brain_test = []
            gene_exp = brain_test.iloc[i, :]
            sorted = pd.DataFrame(gene_exp[gene_exp!=0].sort_values(ascending=False))
            #sorted = sorted.T
            for x, exp in enumerate(sorted.iloc):
                c_name = sorted.index.tolist()[x]
                prelim_brain_test.append(c_name)
            corpus_test_brain.append(prelim_brain_test)
        
        '''
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
        '''
        print('loaded')
        return brain_train, brain_test, brain_train_labels, brain_test_labels, corpus_train_brain, corpus_test_brain#, spleen_train, spleen_test, spleen_train_labels, spleen_test_labels, corpus_train_spleen, kidney_train, kidney_test, kidney_train_labels, kidney_test_labels, corpus_train_kidney

    def get_w2v(self, brain_train, brain_test, brain_train_labels, brain_test_labels, corpus_train_brain, corpus_test_brain):

        b_w2v = self.w2v_embed(corpus_train_brain).wv
        brain_train_w2v = []

        for cell in brain_train.columns:
            brain_train_w2v.append([cell, *b_w2v[cell]])

        b_cells = pd.DataFrame(brain_train_w2v)
        b_cells.drop(columns=0, inplace=True)
        b_cells.to_csv(self.path+'/brain_train.csv', index=False, header=False)

        b_te_w2v = self.w2v_embed(corpus_test_brain).wv
        brain_test_w2v = []

        for cell in brain_test.columns:
            brain_test_w2v.append([cell, *b_te_w2v[cell]])

        b_te_cells = pd.DataFrame(brain_test_w2v)
        b_te_cells.drop(columns=0, inplace=True)
        b_te_cells.to_csv(self.path+'/brain_test.csv', index=False, header=False)

        #brain_genes = pd.DataFrame(b_matrix)
        #brain_genes.index = brain_train.index 
        #brain_genes.to_csv(self.path+'/brain_w2v_genes.csv', index=True, header=False)
        '''
        s_w2v = self.w2v_embed(corpus_train_spleen).wv
        s_matrix = np.zeros((len(spleen_train), s_w2v.vector_size))

        for i, gene in enumerate(spleen_train.index):
            try:
                s_matrix[i] = s_w2v[gene]
            except KeyError:
                s_matrix[i] = np.zeros(s_w2v.vector_size)

        s_cells = spleen_train.T.values @ s_matrix
        s_cells = np.multiply(s_cells, spleen_train_pca)
        s_cells = pd.DataFrame(s_cells)
        s_cells.to_csv(self.path+'/spleen_train.csv', index=False, header=False)

        s_te_cells = spleen_test.T.values @ s_matrix
        s_te_cells = np.multiply(s_te_cells, spleen_test_pca)
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
        k_cells = np.multiply(k_cells, kidney_train_pca)
        k_cells = pd.DataFrame(k_cells)
        k_cells.to_csv(self.path+'/kidney_train.csv', index=False, header=False)

        k_te_cells = kidney_test.T.values @ k_matrix
        k_te_cells = np.multiply(k_te_cells, kidney_test_pca)
        k_te_cells = pd.DataFrame(k_te_cells)
        k_te_cells.to_csv(self.path+'/kidney_test.csv', index=False, header=False)

        kidney_genes = pd.DataFrame(k_matrix)
        kidney_genes.index = kidney_train.index 
        kidney_genes.to_csv(self.path+'/kidney_w2v_genes.csv', index=True, header=False)
        '''

        brain_train_labels.to_csv(self.path+'/brain_train_labels.csv', index=False)
        brain_test_labels.to_csv(self.path+'/brain_test_labels.csv', index=False)
        #spleen_train_labels.to_csv(self.path+'/spleen_train_labels.csv', index=False)
        #spleen_test_labels.to_csv(self.path+'/spleen_test_labels.csv', index=False)
        #kidney_train_labels.to_csv(self.path+'/kidney_train_labels.csv', index=False)
        #kidney_test_labels.to_csv(self.path+'/kidney_test_labels.csv', index=False)

        print('printed')

    def read_w2v(self):
        tissue_train = pd.read_csv(self.path+'/brain_train.csv', header=None)
        tissue_test = pd.read_csv(self.path+'/brain_test.csv', header=None)
        genes = pd.read_csv(self.path+'/brain_w2v_genes.csv', header=None)
        y_values_train = pd.read_csv(self.path+'/brain_train_labels.csv', skiprows=1, header=None, dtype=str)
        y_values_test = pd.read_csv(self.path+'/brain_test_labels.csv', skiprows=1, header=None, dtype=str)
        normalized_train = pd.read_csv(self.path+'/normalized_brain_train.csv', header=0, index_col=0)
        normalized_test = pd.read_csv(self.path+'/normalized_brain_test.csv', header=0, index_col=0)

        return tissue_train, tissue_test, genes, y_values_train, y_values_test, normalized_train, normalized_test
    
    def w2v_embed(self, corpus):
        '''
        Word2Vec Embeddings
        '''
        w2v_embed = Word2Vec(corpus, min_count=1, vector_size=self.dimensions)
        return w2v_embed
    
    @torch.no_grad()
    def bert_embed(self, x, corpus, y):
        chunk_size = 256
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased")
        model = nn.DataParallel(model)
        model.to(self.device)
        outputs = []
        for sentence in corpus:
            tokenized_text = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
            tokenized_text = {key: value.to(self.device) for key, value in tokenized_text.items()}
            chunk_outputs = model(**tokenized_text)
            outputs.append(chunk_outputs)

        print(outputs)
        return outputs

data = data_pre()