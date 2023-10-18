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




class NN_base_superhuman_model(base_superhuman_model):
    def __init__(self, demo_list, num_of_demos, num_of_features, subdom_constant, alpha, sample_loss):
        super().__init__()
        self.type = "NN"
        self.demo_list = demo_list
        self.num_of_demos = num_of_demos
        self.num_of_features = num_of_features
        self.subdom_constant = subdom_constant
        self.alpha = alpha
        self.sample_loss = sample_loss
        self.theta = 0


    def pretrain_classifier(self, data_loader, optimizer):
        running_loss = 0
        last_loss = 0
        for x, y in data_loader:
            self.model.zero_grad()
            #p_y = clf(x)
            # self.model.cuda()
            loss, pred = self.model(x, self.demo_list, self.num_of_demos, self.num_of_features, self.subdom_constant, self.alpha, self.sample_loss)
            #loss = torch.tensor(loss)
            loss.backward()
            optimizer.step()
        return self.model

    def fit(self, X_train, Y_train):
        
        train_data = X_train, Y_train#PandasDataSet(X_train, Y_train)
        n_features = X_train.shape[1]
        train_loader = DataLoader(train_data, batch_size=32, shuffle=False, drop_last=True)
        # train_data.train_data.to(torch.device('cuda:0'))
        print('# training samples:', len(train_data))
        print('# batches:', len(train_loader))
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("running here")
        self.model = Classifier(n_features=n_features)
        model_optimizer = optim.Adam(self.model.parameters())


        N_CLF_EPOCHS = 30
        for epoch in range(N_CLF_EPOCHS):
            print(epoch)
            self.model = self.pretrain_classifier(train_loader, model_optimizer)
        
        #Additional Info when using cuda
        
        # if device.type == 'cuda':
        #     print(torch.cuda.get_device_name(0))
        #     print('Memory Usage:')
        #     print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        #     print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
        print("it returned")
        return self.model

    def predict_proba(self, X):
        if type(X) is list: # used when X is only an item
            X = np.asarray(X, dtype=np.float32)
            # data = torch.Tensor(X)
            data = torch.from_numpy(X)
        else:               # when X is a dataframe of items
            if isinstance(X, np.ndarray):
                data = torch.from_numpy(X)
            elif isinstance(X, pd.DataFrame):
                data = X.to_numpy()
                data = torch.from_numpy(data)
            else:
                data = X
            #X = pd.DataFrame(X)
            # data = torch.tensor(test_data.values.astype(np.float32))
                
            #print("test_data: ", test_data)
            #print("test_data.tensors[0]: ", test_data.tensors[0])

        with torch.no_grad():
            loss, pred = self.model(data, self.demo_list, self.num_of_demos, self.num_of_features, self.subdom_constant, self.alpha, self.sample_loss) #clf(test_data.tensors[0])

        #print(pred.numpy().ravel())
        #acc = accuracy_score(Y, pred.numpy().ravel().round())
        #print(acc)
        #print("pred: ",  pred)
        #print("pred.numpy(): ", pred.numpy())
        #print("pred.numpy().ravel(): ", pred.numpy().ravel())
        return pred.cpu().numpy()#.ravel()

    def get_model_theta(self):
        return self.theta
  
    def update_model_theta(self, new_theta):
        self.theta = np.asarray([new_theta]) # update the coefficient of our logistic regression model with the new theta




class PandasDataSet(TensorDataset):

    def __init__(self, *dataframes):
        tensors = (self._df_to_tensor(df) for df in dataframes)
        super(PandasDataSet, self).__init__(*tensors)

    def _df_to_tensor(self, df):
        if isinstance(df, pd.Series):
            df = df.to_frame('dummy')
        return torch.from_numpy(df.values).float().to(torch.device('cuda:0'))


class Classifier(nn.Module):

    def __init__(self, n_features, n_hidden=32, p_dropout=0.2):
        super(Classifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(n_hidden, 2),
        )
        torch.set_default_tensor_type(torch.FloatTensor)
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # print('Using device:', self.device)

    def forward(self, x, demo_list, num_of_demos, num_of_features, subdom_constant, alpha, sample_loss):
        loss = self.subdom_loss(demo_list, num_of_demos, num_of_features, subdom_constant, alpha, sample_loss) 
        return loss, torch.sigmoid(self.network(x))
    
    def subdom_loss(self, demo_list, num_of_demos, num_of_features, subdom_constant, alpha, sample_loss):
        start_time = time.time()
        subdom_tensor = torch.empty([num_of_demos, num_of_features])
        sample_loss_arr = torch.tensor(sample_loss, requires_grad=True)
        for j, x in enumerate(demo_list):
            for k in range(num_of_features):
                sample_loss = sample_loss_arr[j, k]
                demo_loss = torch.tensor(demo_list[j].metric[k], requires_grad=True)
                subdom_val = max(torch.tensor(alpha[k], requires_grad=True)*(sample_loss - demo_loss) + 1, 0)    
                subdom_tensor[j, k] =  subdom_val - subdom_constant       # subtract constant c to optimize for useful demonstation instead of avoiding from noisy ones
                #grad_theta += self.subdom_tensor[j, k] * self.feature_matching(j)
            
        #print("--- %s end of compute_grad_theta ---" % (time.time() - start_time))
        subdom_tensor_sum = torch.sum(subdom_tensor)
        return subdom_tensor_sum
    




def pretrain_classifier(clf, data_loader, optimizer, sh_obj):
    print("this is running")
    for x, y, _ in data_loader:
        print("running")
        clf.zero_grad()
        #p_y = clf(x)
        loss, pred = clf(x, sh_obj.demo_list, sh_obj.num_of_demos, sh_obj.num_of_features, sh_obj.subdom_constant, sh_obj.alpha, sh_obj.sample_loss)
        #loss = torch.tensor(loss)
        loss.backward()
        optimizer.step()
    return clf



def predict_nn(X_train, y_train, Z_train, X_test, y_test, Z_test, sh_obj):
    
    train_data = X_train, y_train, Z_train #PandasDataSet(X_train, y_train, Z_train)
    test_data = X_test, y_test, Z_test #PandasDataSet(X_test, y_test, Z_test)
    n_features = X_train.shape[1]

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, drop_last=True)
    print("predict_nn")
    print('# training samples:', len(train_data))
    print('# batches:', len(train_loader))

    clf = Classifier(n_features=n_features, sh_obj = sh_obj)
    clf_optimizer = optim.Adam(clf.parameters())


    N_CLF_EPOCHS = 2
    

    for epoch in range(N_CLF_EPOCHS):
        clf = pretrain_classifier(clf, train_loader, clf_optimizer, clf_criterion, sh_obj)

    with torch.no_grad():
        loss, pre_clf_test = clf(test_data.tensors[0], sh_obj.demo_list, sh_obj.num_of_demos, sh_obj.num_of_features, sh_obj.subdom_constant, sh_obj.alpha, sh_obj.sample_loss) #clf(test_data.tensors[0])

    print(pre_clf_test.numpy().ravel())
    acc = accuracy_score(y_test, pre_clf_test.numpy().ravel().round())
    print(acc)
    return pre_clf_test.numpy().ravel().round()
