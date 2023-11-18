import os
import sys
current_script_path = __file__
current_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import dance
from dance.datasets.singlemodality import ScDeepSortDataset
os.environ['DGLBACKEND'] = 'pytorch'
import torch
import numpy as np
import pandas as pd
from pprint import pprint
from gensim.models import Word2Vec
from transformers import LongformerTokenizer, LongformerModel
import gc


class data_pre():
    def __init__(self):
        
        current_script_path = __file__
        current_dir = os.path.dirname(current_script_path)
        parent_dir = os.path.dirname(current_dir)
        sys.path.append(parent_dir)
        parent_parent = os.path.dirname(parent_dir)
        data_dir_ = parent_parent+'\dance_data'
        self.path = data_dir_

        #brain_x, corpus_brain, spleen_x, corpus_spleen, kidney_x, corpus_kidney, brain_y, spleen_y, kidney_y, b_train, corpus_btrain, s_train, corpus_strain, k_train, corpus_ktrain, btrain_y, strain_y, ktrain_y = self.load_data()
        
        #self.get_w2v(brain_x, corpus_brain, spleen_x, corpus_spleen, kidney_x, corpus_kidney, brain_y, spleen_y, kidney_y,
        #             b_train, corpus_btrain, s_train, corpus_strain, k_train, corpus_ktrain, btrain_y, strain_y, ktrain_y)
        
    def load_data(self):
        '''
        Data Pre-Processing

        Loads from CSV downloaded from original dance package, and applies LogNormalize 
        (ScDeepSort choice of normalization)

        '''

        '''
        This Block processes the Train set
        b_train, corpus_btrain, s_train, corpus_strain, k_train, corpus_ktrain, btrain_y, strain_y, ktrain_y
        '''
        b_train = pd.concat([pd.read_csv('train/mouse/mouse_Brain753_data.csv', header=0, index_col=0),
                             pd.read_csv('train/mouse/mouse_Brain3285_data.csv', header=0, index_col=0)],axis=1, ignore_index=True).fillna(0)
                             #pd.read_csv('test/mouse/mouse_Brain2695_data.csv', header=0, index_col=0)], axis=1, ignore_index=True).fillna(0)
        b_train = b_train.apply(lambda x: x/x.sum(), axis=0)
        b_train = b_train.apply(lambda x: x*10000, axis=0)
        b_train = b_train.apply(lambda x: np.log1p(x), axis=0).fillna(0)
        btrain_y = pd.concat([pd.read_csv('train/mouse/mouse_Brain753_celltype.csv')['Cell_type'],
                              pd.read_csv('train/mouse/mouse_Brain3285_celltype.csv')['Cell_type']], axis=0, ignore_index=True)
        b_train.to_csv('normalized_brain.csv', index=True, header=True)
        #mask = btrain_y.str.lower().str.contains('oligodendrocyte', na=False)
        #btrain_y.loc[mask] = 'oligodendrocyte'
                              #pd.read_csv('test/mouse/mouse_Brain2695_celltype.csv')['Cell_type']], axis=0, ignore_index=True)

        corpus_btrain = []
        for c_name in b_train:
            cell = b_train[c_name]
            sorted = cell[cell!=0].sort_values(ascending=True)
            corpus_btrain.append(sorted.index.tolist())

        s_train = pd.read_csv('train/mouse/mouse_Spleen1970_data.csv', header=0, index_col=0).fillna(0)
        s_train = s_train.apply(lambda x: x/x.sum(), axis=0)
        s_train = s_train.apply(lambda x: x*10000, axis=0)
        s_train = s_train.apply(lambda x: np.log1p(x), axis=0).fillna(0)
        strain_y = pd.read_csv('train/mouse/mouse_Spleen1970_celltype.csv')['Cell_type']
        s_train.to_csv('normalized_spleen.csv', index=True, header=True)


        corpus_strain = []
        for c_name in s_train:
            cell = s_train[c_name]
            sorted = cell[cell!=0].sort_values(ascending=False)
            corpus_strain.append(sorted.index.tolist())

        k_train = pd.read_csv('train/mouse/mouse_Kidney4682_data.csv', header=0, index_col=0).fillna(0)
        k_train = k_train.apply(lambda x: x/x.sum(), axis=0)
        k_train = k_train.apply(lambda x: x*10000, axis=0)
        k_train = k_train.apply(lambda x: np.log1p(x), axis=0).fillna(0)
        ktrain_y = pd.read_csv('train/mouse/mouse_Kidney4682_celltype.csv')['Cell_type']
        k_train.to_csv('normalized_kidney.csv', index=True, header=True)


        corpus_ktrain = []
        for c_name in k_train:
            cell = k_train[c_name]
            sorted = cell[cell!=0].sort_values(ascending=False)
            corpus_ktrain.append(sorted.index.tolist())

 
        '''
        This Block processes the Test set
        brain_x, corpus_brain, spleen_x, corpus_spleen, kidney_x, corpus_kidney, brain_y, spleen_y, kidney_y
        '''
        brain_x = pd.read_csv('test/mouse/mouse_Brain2695_data.csv', header=0, index_col=0)
        brain_x.columns = range(len(brain_x.columns))
        brain_x = brain_x.apply(lambda x: x/x.sum(), axis=0)
        brain_x = brain_x.apply(lambda x: x*10000, axis=0)
        brain_x = brain_x.apply(lambda x: np.log1p(x), axis=0)
        brain_y = pd.read_csv('test/mouse/mouse_Brain2695_celltype.csv')['Cell_type'].reset_index(drop=True)

        corpus_brain = []
        for c_name in brain_x:
            cell = brain_x[c_name]
            sorted = cell[cell!=0].sort_values(ascending=True)
            corpus_brain.append(sorted.index.tolist())
   
        spleen_x = pd.read_csv('test/mouse/mouse_Spleen1759_data.csv', header=0, index_col=0)
        spleen_x.columns = range(len(spleen_x.columns))
        spleen_x = spleen_x.apply(lambda x: x/x.sum(), axis=0)
        spleen_x = spleen_x.apply(lambda x: x*10000, axis=0)
        spleen_x = spleen_x.apply(lambda x: np.log1p(x), axis=0)
        spleen_y = pd.read_csv('test/mouse/mouse_Spleen1759_celltype.csv')['Cell_type'].reset_index(drop=True)

        corpus_spleen = []
        for c_name in spleen_x:
            cell = spleen_x[c_name]
            sorted = cell[cell!=0].sort_values(ascending=False)
            corpus_spleen.append(sorted.index.tolist())


        kidney_x = pd.read_csv('test/mouse/mouse_Kidney203_data.csv', header=0, index_col=0)
        kidney_x.columns = range(len(kidney_x.columns))
        kidney_x = kidney_x.apply(lambda x: x/x.sum(), axis=0)
        kidney_x = kidney_x.apply(lambda x: x*10000, axis=0)
        kidney_x = kidney_x.apply(lambda x: np.log1p(x), axis=0)
        kidney_y = pd.read_csv('test/mouse/mouse_Kidney203_celltype.csv')['Cell_type'].reset_index(drop=True)

        corpus_kidney = []
        for c_name in kidney_x:
            cell = kidney_x[c_name]
            sorted = cell[cell!=0].sort_values(ascending=False)
            corpus_kidney.append(sorted.index.tolist())
        

        return brain_x, corpus_brain, spleen_x, corpus_spleen, kidney_x, corpus_kidney, brain_y, spleen_y, kidney_y, b_train, corpus_btrain, s_train, corpus_strain, k_train, corpus_ktrain, btrain_y, strain_y, ktrain_y

    def get_w2v(self, brain_x, corpus_brain, spleen_x, corpus_spleen, kidney_x, corpus_kidney, brain_y, spleen_y, kidney_y,
                b_train, corpus_btrain, s_train, corpus_strain, k_train, corpus_ktrain, btrain_y, strain_y, ktrain_y):
        '''
        Train Block
        '''
        b_w2v = self.w2v_embed(corpus_btrain).wv
        b_matrix = np.zeros((len(b_train), b_w2v.vector_size))
        for i, gene in enumerate(b_train.index):
            try:
                b_matrix[i] = b_w2v[gene]
            except KeyError:
                b_matrix[i] = np.zeros(b_w2v.vector_size)
        b_cells = np.dot(b_train.T.values, b_matrix)
        np.savetxt('b_train.csv', b_cells, delimiter=',')
        brain_genes = pd.DataFrame(b_matrix)
        brain_genes.index = b_train.index 
        brain_genes.to_csv('brain_w2v_genes.csv', index=True, header=False)

        s_w2v = self.w2v_embed(corpus_strain).wv
        s_matrix = np.zeros((len(s_train), s_w2v.vector_size))
        for i, gene in enumerate(s_train.index):
            try:
                s_matrix[i] = s_w2v[gene]
            except KeyError:
                s_matrix[i] = np.zeros(s_w2v.vector_size)
        s_cells = np.dot(s_train.T.values, s_matrix)
        np.savetxt('s_train.csv', s_cells, delimiter=',')
        spleen_genes = pd.DataFrame(s_matrix)
        spleen_genes.index = s_train.index
        spleen_genes.to_csv('spleen_w2v_genes.csv', index=True, header=False)

        k_w2v = self.w2v_embed(corpus_ktrain).wv
        k_matrix = np.zeros((len(k_train), k_w2v.vector_size))
        for i, gene in enumerate(k_train.index):
            try:
                k_matrix[i] = k_w2v[gene]
            except KeyError:
                k_matrix[i] = np.zeros(k_w2v.vector_size)
        k_cells = np.dot(k_train.T.values, k_matrix)
        np.savetxt('k_train.csv', k_cells, delimiter=',')
        kidney_genes = pd.DataFrame(k_matrix)
        kidney_genes.index = k_train.index
        kidney_genes.to_csv('kidney_w2v_genes.csv', index=True, header=False)

        y_all = pd.Series()
        embeddings = np.vstack([b_cells, s_cells, k_cells])
        np.savetxt('train_embeddings.csv', embeddings, delimiter=',')
        y_all = pd.concat([btrain_y, strain_y, ktrain_y])
        y_all.to_csv('y_train.csv', index=False)
        btrain_y.to_csv('btrain_y.csv', index=False)
        strain_y.to_csv('strain_y.csv', index=False)
        ktrain_y.to_csv('ktrain_y.csv', index=False)
        '''
        Test Block
        '''
        brain_w2v = self.w2v_embed(corpus_brain).wv
        brain_matrix = np.zeros((len(brain_x), brain_w2v.vector_size))
        for i, gene in enumerate(brain_x.index):
            try:
                brain_matrix[i] = brain_w2v[gene]
            except KeyError:
                brain_matrix[i] = np.zeros(brain_w2v.vector_size)

        brain_cells = np.dot(brain_x.T.values, brain_matrix)
        np.savetxt('brain_cells.csv', brain_cells, delimiter=',')        

        spleen_w2v = self.w2v_embed(corpus_spleen).wv
        spleen_matrix = np.zeros((len(spleen_x), spleen_w2v.vector_size))
        for i, gene in enumerate(spleen_x.index):
            try:
                spleen_matrix[i] = spleen_w2v[gene]
            except KeyError:
                spleen_matrix[i] = np.zeros(spleen_w2v.vector_size)

        spleen_cells = np.dot(spleen_x.T.values, spleen_matrix)
        np.savetxt('spleen_cells.csv', spleen_cells, delimiter=',')

        kidney_w2v = self.w2v_embed(corpus_kidney).wv
        kidney_matrix = np.zeros((len(kidney_x), kidney_w2v.vector_size))
        for i, gene in enumerate(kidney_x.index):
            try:
                kidney_matrix[i] = kidney_w2v[gene]
            except KeyError:
                kidney_matrix[i] = np.zeros(kidney_w2v.vector_size)

        kidney_cells = np.dot(kidney_x.T.values, kidney_matrix)
        np.savetxt('kidney_cells.csv', kidney_cells, delimiter=',')

        y_all = pd.Series()
        embeddings = np.vstack([brain_cells, spleen_cells, kidney_cells])
        np.savetxt('w2v_embeddings.csv', embeddings, delimiter=',')
        y_all = pd.concat([brain_y, spleen_y, kidney_y])
        y_all.to_csv('y_values.csv', index=False)
        brain_y.to_csv('brain_y.csv', index=False)
        spleen_y.to_csv('spleen_y.csv', index=False)
        kidney_y.to_csv('kidney_y.csv', index=False)

    def read_w2v(self):
        train = np.genfromtxt(self.path+'\\b_train.csv', delimiter=',')
        genes = np.genfromtxt(self.path+'\\brain_w2v_genes.csv', delimiter=',')
        y_train = np.loadtxt(self.path+'\\train_y.csv', delimiter=',', unpack=True, skiprows=1, dtype=str)
        normalized = pd.read_csv(self.path+'\\normalized_brain.csv', header=0, index_col=0)
        test = np.genfromtxt(self.path+'\\brain_cells.csv', delimiter=',')
        y_test = np.loadtxt(self.path+'\\brain_y.csv', delimiter=',', unpack=True, skiprows=1, dtype=str)

        return train, y_train, genes, normalized, test, y_test
    def w2v_embed(self, corpus):
        '''
        Word2Vec Embeddings
        '''
        w2v_embed = Word2Vec(corpus, min_count=1)
        return w2v_embed

    def longformer_embed(self, corpus):
        '''
        LongFormer Embeddings
        '''
        tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        model = LongformerModel.from_pretrained('allenai/longformer-base-4096')
        model.eval()
        input = [' '.join(words) for words in corpus]
        encoded_input = tokenizer(input, padding=True, truncation=True, return_tensors='pt')

        for sentence in input:
            tokens = tokenizer.encode(sentence, truncation=False)
            if len(tokens) > tokenizer.model_max_length:
                print('get truncated noob.')
        return 0
        with torch.no_grad():
            outputs = model(**encoded_input)

        self.bert_embed = outputs.last_hidden_state
        print(self.bert_embed)

