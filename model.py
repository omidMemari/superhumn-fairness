import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression


torch.manual_seed(1)

class base_superhuman_model:
    def __init__(self):
        self.logi_params = {
        'C': 100,
        'penalty': 'l2',
        'solver': 'newton-cg',
        'max_iter': 1000
        }
        
    @abstractmethod
    def fit(self,X,Y):
        pass

    @abstractmethod
    def predict_proba(self,X):
        pass

    def predict(self,X):
        return np.round(self.predict_proba(X))

    def score(self,X,Y):
        return 1 - np.mean(abs(self.predict(X) - Y))

    def expected_error(self,X,Y):
        proba = self.predict_proba(X)
        return np.mean(np.where(Y == 1 , 1 - proba, proba))

class LogisticRegression_pytorch(nn.Module):
    # large = 512,
    # small = 256
    def __init__(self, n_inputs, n_outputs, n_nodes=512):
        super(LogisticRegression_pytorch, self).__init__()
        # 4 more inputs, posA negA posB negB
        self.linear = nn.Linear(n_inputs, n_outputs)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim = 1)
        self.type = 'nn_pytorch'
        self.n_nodes = n_nodes
        self.fc1 = nn.Linear(n_inputs, n_nodes)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        
        self.fc2 = nn.Linear(int(n_nodes), int(n_nodes/2))
        self.fc3 = nn.Linear(int(n_nodes/2), 2)
        self.out = nn.Softmax(dim=1)
        # self.dropout1 = nn.Dropout(0.1)
        # self.dropout2 = nn.Dropout(0.1)      
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.00001)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        # x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        # x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.out(x)
        return x
    
    # defaut max_iter = 1000
    def fit(self, X_train, Y_train, max_iter = 15000):
        X_train = torch.tensor(X_train.to_numpy(), dtype=torch.float32).cuda()
        Y_train = torch.tensor(Y_train.to_numpy(), dtype=torch.float32).cuda()
        Y_train = Y_train.view(-1, 1)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(max_iter):
            self.optimizer.zero_grad()
            outputs = self(X_train)
            loss = criterion(outputs, Y_train.squeeze().type(torch.LongTensor).cuda())
            loss.backward()
            self.optimizer.step()
        return self
    
    def predict_proba(self, X):
        # check if it is already in numpy format
        if type(X) is list:
            X = torch.tensor(X, dtype=torch.float32)
        else:
            X = torch.tensor(X.to_numpy(), dtype=torch.float32)
        # 
        out = torch.cat((1 - self(X.cuda()), self(X.cuda())), 1).detach().cpu().numpy()
        return out
    
    def get_model_theta(self):
        # original is self.linear.weight.data.cpu().numpy().ravel()
        return self.linear.weight.data.cpu().numpy().ravel()
    
    def update_model_theta(self, new_theta):
        self.linear.weight.data = torch.tensor([new_theta], dtype=torch.float32)
        return self
        
class LR_base_superhuman_model(base_superhuman_model):

    def __init__(self):
        super().__init__()
        self.model = LogisticRegression(**self.logi_params)
        self.type = "LR base"
    
    def fit(self, X_train, Y_train):
        return self.model.fit(X_train, Y_train)

    def predict_proba(self, X):
        out = self.model.predict_proba(X)
        return out
    
    def get_model_theta(self):
        return self.model.coef_[0]
  
    def update_model_theta(self, new_theta):
        self.model.coef_ = np.asarray([new_theta]) # update the coefficient of our logistic regression model with the new theta
