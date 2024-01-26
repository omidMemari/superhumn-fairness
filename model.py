import pandas as pd
import seaborn as sns
from tqdm import tqdm
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from abc import ABC,abstractmethod
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



class LR_base_superhuman_model(base_superhuman_model):

    def __init__(self):
        super().__init__()
        self.model = LogisticRegression(**self.logi_params)
        self.type = "LR"
    
    def fit(self, X_train, Y_train):
        return self.model.fit(X_train, Y_train)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    
    def get_model_theta(self):
        return self.model.coef_[0]
  
    def update_model_theta(self, new_theta):
        self.model.coef_ = np.asarray([new_theta]) # update the coefficient of our logistic regression model with the new theta




class Net(nn.Module):
    def __init__(self, n_features, n_hidden=32, p_dropout=0.2):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_features, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(int(256), 256)
        self.fc3 = nn.Linear(256, 2)
        self.out = nn.Softmax(dim=1)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)        
        torch.set_default_tensor_type(torch.FloatTensor)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.out(x)
        return x
        
    def subdom_loss(self, demo_list, num_of_demos, num_of_features, subdom_constant, alpha, sample_loss):
        subdom_tensor = torch.empty([num_of_demos, num_of_features])
        sample_loss_arr = torch.tensor(sample_loss, requires_grad=True)
        for j, x in enumerate(demo_list):
            for k in range(num_of_features):
                sample_loss = sample_loss_arr[j, k]
                demo_loss = torch.tensor(demo_list[j].metric[k], requires_grad=True)
                subdom_val = torch.clamp(alpha[k]*(sample_loss - demo_loss) + 1, 0)    
                subdom_tensor[j, k] =  subdom_val - subdom_constant       # subtract constant c to optimize for useful demonstation instead of avoiding from noisy ones
                # torch.clamp(self.alpha[k] * (sample_loss - demo_loss) + 1 - self.subdom_constant, 0)
                #grad_theta += self.subdom_tensor[j, k] * self.feature_matching(j)
            
        #print("--- %s end of compute_grad_theta ---" % (time.time() - start_time))
        subdom_tensor_sum = torch.sum(subdom_tensor)
        return subdom_tensor_sum

    def fit(self):
        self.model = Net(n_features=109)
        print(self.num_of_features)
        num_iter = 100
        optim1 = optim.Adam([self.alpha], lr=1e-05)
        optim2 = optim.Adam(self.model.parameters(), lr=1e-05)
        for i in range(num_iter):
            for demo in self.demo_list:
                subdom_tensor = torch.empty([self.num_of_demos, self.num_of_features])
                sample_loss_arr = torch.tensor(self.sample_loss, requires_grad=True)
                # print loss per 100 iterations
                if i % 100 == 0:
                    print(i, loss.item())
                optim1.zero_grad()
                optim2.zero_grad()
                (loss).backward()
                optim1.step()
                optim2.step()
        # save model to file
        # torch.save(self.model.state_dict(), 'model.pth')
        print(self.demo_list[0].train_x)
        test = self.model(torch.Tensor(self.demo_list[0].train_x.values), self.demo_list, self.num_of_demos, self.num_of_features, self.subdom_constant, self.alpha, self.sample_loss)
        print(test)
        print(self.demo_list[0].train_y)
        return self.model

    def predict_proba(self, X):
        if type(X) is list: # used when X is only an item
            X = np.asarray(X, dtype=np.float32)
            # data = torch.Tensor(X)
            data = torch.from_numpy(X)
        else: # when X is a dataframe of items
            if isinstance(X, np.ndarray):
                data = torch.from_numpy(X)
            elif isinstance(X, pd.DataFrame):
                data = X.to_numpy()
                data = torch.from_numpy(data)
            else:
                data = X
        
        with torch.no_grad():
            pred = self.Net(data, self.demo_list, self.num_of_demos, self.num_of_features, self.subdom_constant, self.alpha, self.sample_loss)
        return pred.cpu().numpy()#.ravel()