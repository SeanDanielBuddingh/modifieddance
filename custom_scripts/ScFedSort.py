import torch
import numpy as np
import pandas as pd
import copy
from data_pre  import data_pre
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import networkx as nx

class ScFedSort(torch.nn.Module):

    def __init__(self):

        super(ScFedSort, self).__init__()
        self.seed = 42
        torch.manual_seed(self.seed)
        self.num_clients = 1

        input_dim = 100
        hidden_dim = input_dim
        output_dim = 21

        self.layer1 = torch.nn.Linear(input_dim, hidden_dim)
        #self.activation = torch.nn.Tanh()
        self.layer2 = torch.nn.Linear(hidden_dim, output_dim)

    def eval_run(self):
        train_inputs, train_targets, test_inputs, test_targets = self.read_data()
        acc, loss = self.run(train_inputs, train_targets, test_inputs, test_targets)
        print('Mean Accuracy: ', acc)
        print('Median Loss: ', loss)
   

    def read_data(self):
        self.data = data_pre()
        #inputs, targets, test, test_targets = self.data.read_w2v()
        inputs, targets, genes = self.data.read_w2v()

        #all_targets = np.concatenate((targets, test_targets))
        print(len(np.unique(targets)))
        
        #unique_train_targets = set(targets)
        #unique_test_targets = set(test_targets)
        
        # Find labels in the test set that are not in the training set
        #test_not_in_train = unique_test_targets - unique_train_targets

        #onehot_encoder = OneHotEncoder(sparse_output=False).fit(targets.reshape(-1, 1))
        #unique_labels = onehot_encoder.categories_[0]

        # Find the positions of test_not_in_train labels in the one-hot encoding
        #positions = [np.where(unique_labels == label)[0][0] for label in test_not_in_train]
        #print("Positions in one-hot encoding for labels in test but not in train:", positions)

        #targets_onehot = onehot_encoder.transform(targets.reshape(-1, 1))
        #test_targets_onehot = onehot_encoder.transform(test_targets.reshape(-1, 1))
        label_encoder = LabelEncoder().fit(targets)

        targets_encoded = label_encoder.transform(targets)

        train_inputs, train_targets, test_inputs, test_targets = self.mix_data(inputs, targets_encoded)
        #test_inputs, test_targets = self.mix_data(test, test_targets_onehot)

        return train_inputs, train_targets, test_inputs, test_targets

    def basic_graph(self,train_inputs, train_targets, genes):
        G = nx.Graph()

        # Adding input nodes
        for i in range(len(train_inputs)):
                G.add_node(f"input_{i}")

        # Adding gene nodes
        for i in range(len(genes)):
                G.add_node(genes[i, 0], features=genes[i, 1:])

        # Adding edges
        for i in range(len(train_inputs)):
            input_node = f"input_{i}"
                
            for j in range(len(genes)):
                gene_node = genes[j, 0]
                        
                G.add_edge(input_node, gene_node, weight=train_inputs[i, j])

    '''
    def read_data(self):
        self.data = data_pre()
        inputs, targets, test, test_targets = self.data.read_w2v()

        all_targets = np.concatenate((targets, test_targets))
        
        label_encoder = LabelEncoder().fit(all_targets)

        targets_encoded = label_encoder.transform(targets)
        test_targets_encoded = label_encoder.transform(test_targets)


        #label_encoder = LabelEncoder().fit(targets)
        #targets_encoded = self.encode_with_unknown(targets, label_encoder)
        #test_targets_encoded = self.encode_with_unknown(test_targets, label_encoder)

        train_inputs, train_targets = self.mix_data(inputs, targets_encoded)
        test_inputs, test_targets = self.mix_data(test, test_targets_encoded)
        return train_inputs, train_targets, test_inputs, test_targets
    '''

    def encode_with_unknown(self, labels, label_encoder):
        known_labels = set(label_encoder.classes_)
        return np.array([label_encoder.transform([label])[0] if label in known_labels else 1 for label in labels])
    
    def forward(self, z):
        z = z.to(torch.float32)
        z = self.layer1(z)
        z = torch.nn.functional.relu(z)
        z = self.layer2(z)
        #return torch.nn.functional.softmax(z, dim = 1)
        return z

    def mix_data(self, inputs, targets):
        np.random.seed(self.seed)
        np.random.shuffle(inputs)

        np.random.seed(self.seed)
        np.random.shuffle(targets)

        train_x, test_x = train_test_split(inputs, test_size=0.3, random_state=self.seed)
        train_y, test_y = train_test_split(targets, test_size=0.3, random_state=self.seed)
        #client_x = np.array_split(inputs, self.num_clients)
        #client_y = np.array_split(targets, self.num_clients)

        #return client_x, client_y
        return train_x, train_y, test_x, test_y

    def cross_entropy(self, outputs, targets, lambda_l2=0.01):
        # Standard cross-entropy loss
        ce_loss = -torch.sum(targets * torch.log(outputs), dim=1)
        ce_loss = torch.mean(ce_loss)

        # L2 regularization term
        l2_reg = torch.tensor(0., requires_grad=True)
        for param in self.parameters():
            l2_reg = l2_reg + torch.norm(param, 2)**2
        l2_reg *= lambda_l2

        # Combined loss
        total_loss = ce_loss + l2_reg

        return total_loss

    def run(self, train_inputs, train_targets, test_inputs, test_targets):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        
        train_inputs = torch.from_numpy(train_inputs).to(device)
        test_inputs = torch.from_numpy(test_inputs).to(device)
        train_targets = torch.from_numpy(train_targets).to(device)
        test_targets = torch.from_numpy(test_targets).to(device)

        self.train()

        num_rounds = 100
        num_epochs = 5
        loss = torch.nn.CrossEntropyLoss()

        inputs_train = np.empty(self.num_clients, dtype=object)
        inputs_test = np.empty(self.num_clients, dtype=object)
        targets_train = np.empty(self.num_clients, dtype=object)
        targets_test = np.empty(self.num_clients, dtype=object)
        
        epochs = 1000
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        for epoch in range(epochs):
            outputs = self(train_inputs)
            #print(outputs)
            loss_realized = loss(outputs, train_targets)#self.cross_entropy(outputs, train_targets, )
            optimizer.zero_grad()
            loss_realized.backward()
            optimizer.step()


        '''
        for round in range(num_rounds):

            global_model = self.state_dict()
            client_models = []

            for client in range(self.num_clients):
                client_model = type(self)()
                client_model.to(device)
                client_model.train()
                client_model.load_state_dict(global_model)
                optimizer = torch.optim.Adam(client_model.parameters(), lr=0.001)

                inputs_train[client] = torch.from_numpy(train_inputs[client]).to(device)
                inputs_test[client] = torch.from_numpy(test_inputs[client]).to(device)
                targets_train[client] = torch.from_numpy(train_targets[client]).to(device)
                targets_test[client] = torch.from_numpy(test_targets[client]).to(device)

                for epoch in range(num_epochs):
                    client_outputs = client_model(inputs_train[client])
                    loss_client = loss(client_outputs, targets_train[client])
                    optimizer.zero_grad()
                    loss_client.backward()
                    optimizer.step()


                client_models.append(copy.deepcopy(client_model.state_dict()))
                
            with torch.no_grad():

                weights = [len(train_inputs[client]) for client in range(self.num_clients)]
                total_samples = sum(weights)

                for key in global_model.keys():
                    global_model[key] = torch.stack([client_models[i][key].cpu() * weights[i] for i in range(len(client_models))], 0).sum(0) / total_samples
                    global_model[key] = global_model[key].to(device)

            self.load_state_dict(global_model)
        '''

        self.eval()
        acc = []
        loss_list = []

        for param in self.parameters():
            print(torch.isnan(param).any(), torch.isinf(param).any())
        '''
        for client in range(self.num_clients):
            with torch.no_grad():
                outputs = self(inputs_test[client])
        '''    
        with torch.no_grad(): 
            #loss_eval = loss(outputs, targets_test[client])
            #loss_list.append(loss_eval)
            outputs_eval = self(test_inputs)
            loss_eval = loss(outputs_eval, test_targets)#self.cross_entropy(outputs_eval, test_targets)
            predicted = torch.argmax(outputs_eval, 1)
            #test_targets_decoded = torch.argmax(test_targets, 1)
            correct = (predicted == test_targets).sum().item()
            total = test_targets.numel()
            accuracy = correct / total
            print('accuracy: ', accuracy)
            acc.append(accuracy)

        top1 = acc#np.mean(acc)
        topl = loss_eval#np.median(loss_list)
        return top1, topl


