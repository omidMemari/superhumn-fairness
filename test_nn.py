import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import seaborn as sns
from tqdm import tqdm
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

from fairness.helpers import load_ICU_data
from fairness.helpers import plot_distributions


torch.manual_seed(1)

def subdom_loss(demo_list, num_of_demos, num_of_features, subdom_constant, alpha, sample_loss):
    start_time = time.time()
    subdom_tensor = torch.empty([num_of_demos, num_of_features])
    sample_loss_arr = torch.tensor(sample_loss, requires_grad=True)
    for j, x in enumerate(demo_list):
      #if j == 0: sh_obj.subdom_constant = torch.tensor(0., requires_grad=True)
      #else: sh_obj.subdom_constant = torch.tensor(sh_obj.get_subdom_constant(), requires_grad=True)
      for k in range(num_of_features):
        sample_loss = sample_loss_arr[j, k]
        demo_loss = torch.tensor(demo_list[j].metric[k], requires_grad=True)
        subdom_val = max(torch.tensor(alpha[k], requires_grad=True)*(sample_loss - demo_loss) + 1, 0)    
        subdom_tensor[j, k] =  subdom_val - subdom_constant       # subtract constant c to optimize for useful demonstation instead of avoiding from noisy ones
        #grad_theta += self.subdom_tensor[j, k] * self.feature_matching(j)
        
    #print("--- %s end of compute_grad_theta ---" % (time.time() - start_time))
    subdom_tensor_sum = torch.sum(subdom_tensor)
    return subdom_tensor_sum


class PandasDataSet(TensorDataset):

    def __init__(self, *dataframes):
        tensors = (self._df_to_tensor(df) for df in dataframes)
        super(PandasDataSet, self).__init__(*tensors)

    def _df_to_tensor(self, df):
        if isinstance(df, pd.Series):
            df = df.to_frame('dummy')
        return torch.from_numpy(df.values).float()


class Classifier(nn.Module):

    def __init__(self, n_features, n_hidden=32, p_dropout=0.2, sh_obj=None):
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
            nn.Linear(n_hidden, 1),
        )
        self.sh_obj = sh_obj

    def forward(self, x, demo_list, num_of_demos, num_of_features, subdom_constant, alpha, sample_loss):
        loss = subdom_loss(demo_list, num_of_demos, num_of_features, subdom_constant, alpha, sample_loss) 
        return loss, torch.sigmoid(self.network(x))



def pretrain_classifier(clf, data_loader, optimizer, criterion, sh_obj):
    for x, y, _ in data_loader:
        clf.zero_grad()

        #p_y = clf(x)
        loss, pred = clf(x, sh_obj.demo_list, sh_obj.num_of_demos, sh_obj.num_of_features, sh_obj.subdom_constant, sh_obj.alpha, sh_obj.sample_loss)
        #loss = torch.tensor(loss)
        #loss = criterion(p_y, y)
        loss.backward()
        optimizer.step()
    return clf



def predict_nn(X_train, y_train, Z_train, X_test, y_test, Z_test, sh_obj):
    train_data = PandasDataSet(X_train, y_train, Z_train)
    test_data = PandasDataSet(X_test, y_test, Z_test)
    n_features = X_train.shape[1]

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, drop_last=True)

    print('# training samples:', len(train_data))
    print('# batches:', len(train_loader))


    clf = Classifier(n_features=n_features, sh_obj = sh_obj)
    #clf_criterion = criterion()
    clf_criterion = nn.BCELoss()
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

