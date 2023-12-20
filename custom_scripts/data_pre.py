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

from gensim.models import Word2Vec
from transformers import BertTokenizer, BertModel



class data_pre():
    def __init__(self):
        
        current_script_path = __file__
        current_dir = os.path.dirname(current_script_path)
        parent_dir = os.path.dirname(current_dir)
        sys.path.append(parent_dir)
        parent_parent = os.path.dirname(parent_dir)
        parent_parent = parent_parent.replace("\\", "/")
        data_dir_ = parent_parent+'/dance_data'
        self.path = data_dir_
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #combined_brain, combined_brain_labels, corpus_combined_brain, combined_spleen, combined_spleen_labels, corpus_combined_spleen, combined_kidney, combined_kidney_labels, corpus_combined_kidney = self.load_data()
        
        #self.get_w2v(combined_brain, combined_brain_labels, corpus_combined_brain, combined_spleen, combined_spleen_labels, corpus_combined_spleen, combined_kidney, combined_kidney_labels, corpus_combined_kidney)

        #self.bert_embed(brain_test, corpus_brain, brain_y)
        #self.bert_embed(spleen_x, corpus_spleen, spleen_y)
        #self.bert_embed(spleen_x, corpus_spleen, kidney_y)
        #self.bert_embed(brain_train, corpus_btrain, btrain_y)
        #self.bert_embed(s_train, corpus_strain, strain_y)
        #self.bert_embed(k_train, corpus_ktrain, ktrain_y)
        
    def load_data(self):
        '''
        Data Pre-Processing

        Loads from CSV downloaded from original dance package, and applies LogNormalize 
        (ScDeepSort choice of normalization)

        '''

        '''
        This Block processes the Train set
        brain_train, corpus_btrain, s_train, corpus_strain, k_train, corpus_ktrain, btrain_y, strain_y, ktrain_y
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
        combined_brain = combined_brain.apply(lambda x: np.log1p(x), axis=0).fillna(0)

        row_sums = combined_brain.sum(axis=1)
        low_percentile = np.percentile(row_sums, 1)
        high_percentile = np.percentile(row_sums, 99)
        combined_brain = combined_brain[(row_sums > low_percentile) & (row_sums < high_percentile)]
        row_stds = combined_brain.std(axis=1)
        low_std_percentile = np.percentile(row_stds, 1)
        high_std_percentile = np.percentile(row_stds, 99)
        combined_brain_filtered = combined_brain[(row_stds > low_std_percentile) & (row_stds < high_std_percentile)]
     
        combined_brain_filtered.to_csv(self.path+'/normalized_brain.csv', index=True, header=True)

        corpus_combined_brain = []
        for c_name in combined_brain_filtered.columns:
            cell = combined_brain_filtered[c_name]
            sorted = cell[cell!=0].sort_values(ascending=True)
            corpus_combined_brain.append(sorted.index.tolist())

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
        combined_spleen = combined_spleen.apply(lambda x: np.log1p(x), axis=0).fillna(0)

        row_sums = combined_spleen.sum(axis=1)
        low_percentile = np.percentile(row_sums, 1)
        high_percentile = np.percentile(row_sums, 99)
        combined_spleen = combined_spleen[(row_sums > low_percentile) & (row_sums < high_percentile)]
        row_stds = combined_spleen.std(axis=1)
        low_std_percentile = np.percentile(row_stds, 1)
        high_std_percentile = np.percentile(row_stds, 99)
        combined_spleen_filtered = combined_spleen[(row_stds > low_std_percentile) & (row_stds < high_std_percentile)]

        combined_spleen_filtered.to_csv(self.path+'/normalized_spleen.csv', index=True, header=True)

        corpus_combined_spleen = []
        for c_name in combined_spleen_filtered:
            cell = combined_spleen_filtered[c_name]
            sorted = cell[cell!=0].sort_values(ascending=True)
            corpus_combined_spleen.append(sorted.index.tolist())

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
        combined_kidney = combined_kidney.apply(lambda x: np.log1p(x), axis=0).fillna(0)

        row_sums = combined_kidney.sum(axis=1)
        low_percentile = np.percentile(row_sums, 1)
        high_percentile = np.percentile(row_sums, 99)
        combined_kidney = combined_kidney[(row_sums > low_percentile) & (row_sums < high_percentile)]
        row_stds = combined_kidney.std(axis=1)
        low_std_percentile = np.percentile(row_stds, 1)
        high_std_percentile = np.percentile(row_stds, 99)
        combined_kidney_filtered = combined_kidney[(row_stds > low_std_percentile) & (row_stds < high_std_percentile)]
        
        combined_kidney_filtered.to_csv(self.path+'/normalized_kidney.csv', index=True, header=True)

        corpus_combined_kidney = []
        for c_name in combined_kidney_filtered.columns:
            cell = combined_kidney_filtered[c_name]
            sorted = cell[cell!=0].sort_values(ascending=True)
            corpus_combined_kidney.append(sorted.index.tolist())
        

        return combined_brain_filtered, combined_brain_labels, corpus_combined_brain, combined_spleen_filtered, combined_spleen_labels, corpus_combined_spleen, combined_kidney_filtered, combined_kidney_labels, corpus_combined_kidney

    def get_w2v(self, combined_brain, combined_brain_labels, corpus_combined_brain, combined_spleen, combined_spleen_labels, corpus_combined_spleen, combined_kidney, combined_kidney_labels, corpus_combined_kidney):

        b_w2v = self.w2v_embed(corpus_combined_brain).wv
        b_matrix = np.zeros((len(combined_brain), b_w2v.vector_size))

        for i, gene in enumerate(combined_brain.index):
            try:
                b_matrix[i] = b_w2v[gene]
            except KeyError:
                b_matrix[i] = np.zeros(b_w2v.vector_size)

        b_cells = combined_brain.T.values @ b_matrix
        b_cells = pd.DataFrame(b_cells)
        b_cells.to_csv(self.path+'/brain.csv', index=False, header=False)
        brain_genes = pd.DataFrame(b_matrix)
        brain_genes.index = combined_brain.index 
        brain_genes.to_csv(self.path+'/brain_w2v_genes.csv', index=True, header=False)

        s_w2v = self.w2v_embed(corpus_combined_spleen).wv
        s_matrix = np.zeros((len(combined_spleen), s_w2v.vector_size))
        
        for i, gene in enumerate(combined_spleen.index):
            try:
                s_matrix[i] = s_w2v[gene]
            except KeyError:
                s_matrix[i] = np.zeros(s_w2v.vector_size)

        s_cells = combined_spleen.T.values @ s_matrix
        s_cells = pd.DataFrame(s_cells)
        s_cells.to_csv(self.path+'/spleen.csv', index=False, header=False)
        spleen_genes = pd.DataFrame(s_matrix)
        spleen_genes.index = combined_spleen.index
        spleen_genes.to_csv(self.path+'/spleen_w2v_genes.csv', index=True, header=False)

        k_w2v = self.w2v_embed(corpus_combined_kidney).wv
        k_matrix = np.zeros((len(combined_kidney), k_w2v.vector_size))

        for i, gene in enumerate(combined_kidney.index):
            try:
                k_matrix[i] = k_w2v[gene]
            except KeyError:
                k_matrix[i] = np.zeros(k_w2v.vector_size)

        k_cells = combined_kidney.T.values @ k_matrix
        k_cells = pd.DataFrame(k_cells)
        k_cells.to_csv(self.path+'/kidney.csv', index=False, header=False)
        kidney_genes = pd.DataFrame(k_matrix)
        kidney_genes.index = combined_kidney.index
        kidney_genes.to_csv(self.path+'/kidney_w2v_genes.csv', index=True, header=False)

        y_all = pd.Series()
        embeddings = np.vstack([b_cells, s_cells, k_cells])
        embeddings = pd.DataFrame(embeddings)
        embeddings.to_csv(self.path+'/combined_embeddings.csv', index=False)
        y_all = pd.concat([combined_brain_labels, combined_spleen_labels, combined_kidney_labels])
        y_all.to_csv(self.path+'/y_train.csv', index=False)
        combined_brain_labels.to_csv(self.path+'/brain_y.csv', index=False)
        combined_spleen_labels.to_csv(self.path+'/spleen_y.csv', index=False)
        combined_kidney_labels.to_csv(self.path+'/kidney_y.csv', index=False)

        print('printed')

    def read_w2v(self):
        tissue = pd.read_csv(self.path+'/brain.csv', header=None)
        genes = pd.read_csv(self.path+'/brain_w2v_genes.csv', header=None)
        y_values = pd.read_csv(self.path+'/brain_y.csv', skiprows=1, header=None, dtype=str)
        normalized = pd.read_csv(self.path+'/normalized_brain.csv', header=0, index_col=0)

        return tissue, y_values, genes, normalized
    
    def w2v_embed(self, corpus):
        '''
        Word2Vec Embeddings
        '''
        w2v_embed = Word2Vec(corpus, min_count=1)
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

#data = data_pre()