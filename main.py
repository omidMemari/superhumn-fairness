import os
import sys
import seaborn as sns
import numpy as np
import pandas as pd
from pandas.core.strings import base
import random
import pickle
import time
from sklearn.utils import shuffle
import warnings
import copy
import math
import argparse
from tqdm import tqdm
from test_nn import predict_nn
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
# Fairlearn algorithms and utils
from fairlearn.postprocessing import ThresholdOptimizer
from util import compute_error, get_metrics_df, create_features_dict, find_gamma_superhuman, find_gamma_superhuman_all, load_object, store_object, make_demo_list_filename, make_experiment_filename
from fair_logloss.fair_logloss import DP_fair_logloss_classifier, EOPP_fair_logloss_classifier, EODD_fair_logloss_classifier
from model import LR_base_superhuman_model, NN_base_superhuman_model
"""
task_name: acs_west_poverty
protected attribute: RAC1P
unique values: [0 1] // 0 us White, 1 is Black
label: POVPIP
unique values: [1 0]

task_name: acs_west_mobility
protected attribute: RAC1P
unique values: [0 1]
label: MIG
unique values: [1 0]

task_name: acs_west_income
protected attribute: RAC1P
unique values: [0 1]
label: PINCP
unique values: [0 1]

task_name: acs_west_insurance
protected attribute: RAC1P
unique values: [0 1]
label: HINS2
unique values: [0 1]

task_name: acs_west_public
protected attribute: RAC1P
unique values: [0 1]
label: PUBCOV
unique values: [0 1]

task_name: acs_west_travel
protected attribute: RAC1P
unique values: [0 1]
label: JWMNP
unique values: [0 1]

task_name: acs_west_poverty
protected attribute: RAC1P
unique values: [0 1]
label: POVPIP
unique values: [1 0]

task_name: acs_west_employment
protected attribute: RAC1P
unique values: [0 1]
label: ESR
unique values: [0 1]

"""
default_args = {'dataset': 'Adult', 'iters': 30, 'num_of_demos': 50, 'num_of_features': 4, 'lr_theta': 0.01, 'noise': 'False', 'noise_ratio': 0.2, 'demo_baseline': 'pp', 'features': ['inacc, dp, eqodds, prp'], 'base_model_type': 'LR', 'num_experiment': 10}
label_dict = {'Adult': 'label', 'COMPAS':'two_year_recid', 'Diabetes': 'label', 'acs_west_poverty': 'POVPIP', 'acs_west_mobility': 'MIG', 'acs_west_income': 'PINCP', 'acs_west_insurance': 'HINS2', 'acs_west_public': 'PUBCOV', 'acs_west_travel': 'JWMNP', 'acs_west_employment': 'ESR'}
protected_dict = {'Adult': 'gender', 'COMPAS':'race',  'Diabetes': 'gender', 'acs_west_poverty': 'RAC1P', 'acs_west_mobility': 'RAC1P', 'acs_west_income': 'RAC1P', 'acs_west_insurance': 'RAC1P', 'acs_west_public': 'RAC1P', 'acs_west_travel': 'RAC1P', 'acs_west_employment': 'RAC1P'}
protected_map = {'Adult': {2:"Female", 1:"Male"}, 'COMPAS': {1:'Caucasian', 0:'African-American'}, \
  'Diabetes': {2:"Female", 1:"Male"},\
  'acs_west_poverty': {0:'White', 1:'Black'}, 'acs_west_mobility': {0:'White', 1:'Black'},\
  'acs_west_income': {0:'White', 1:'Black'}, 'acs_west_insurance': {0:'White', 1:'Black'},\
  'acs_west_public': {0:'White', 1:'Black'}, 'acs_west_travel': {0:'White', 1:'Black'}, 'acs_west_employment': {0:'White', 1:'Black'}}

lr_theta = default_args['lr_theta']
num_of_demos = default_args['num_of_demos']
noise_ratio = default_args['noise_ratio']
iters = default_args['iters']
alpha = 0.5
beta = 0.5
lamda = 0.001
#demo_baseline = "pp" #"fair_logloss" 
model = "logistic_regression"
noise_list = [0.2]#0.03, 0.04]#[0.06, 0.07, 0.08, 0.09]##[0.16, 0.17, 0.18, 0.19, 0.20]#[0.11, 0.12, 0.13, 0.14, 0.15]


class Super_human:

  def __init__(self, dataset, num_of_demos, feature, num_of_features, lr_theta, noise, noise_ratio, demo_baseline, base_model_type):
    self.dataset = dataset
    self.num_of_demos = num_of_demos
    self.num_of_features = num_of_features
    self.feature = feature
    self.alpha = [1.0 for _ in range(self.num_of_features)]
    self.gamma_superhuman = [0.0 for _ in range(self.num_of_features)]
    self.label = label_dict[dataset] ##
    print("self.label: ", self.label)
    self.sensitive_feature = protected_dict[dataset] ##
    self.dict_map = protected_map[dataset] ##
    self.lamda = lamda
    self.c = None
    self.lr_theta = lr_theta
    self.noise_ratio = noise_ratio
    self.noise = noise
    self.base_model_type = base_model_type
    self.demo_baseline = demo_baseline
    self.set_paths()
    self.dataset_ref = pd.read_csv(self.dataset_path, index_col=0) #self.dataset_ref = pd.read_csv('dataset_ref.csv', index_col=0)
    self.num_of_attributs = self.dataset_ref.shape[1] - 1 # discard self.label
    self.model_params = None
    self.logi_params = {
        'C': 100,
        'penalty': 'l2',
        'solver': 'newton-cg',
        'max_iter': 1000
    }

  class data_demo:
    def __init__(self, train_x=None, test_x=None, train_y=None, test_y=None, train_A=None, test_A=None, train_A_str=None, test_A_str=None ,idx_train=None, idx_test=None):
        self.train_x = train_x
        self.test_x = test_x
        self.train_y = train_y
        self.test_y = test_y
        self.train_A = train_A
        self.test_A = test_A
        self.train_A_str = train_A_str
        self.test_A_str = test_A_str
        self.idx_train = idx_train
        self.idx_test = idx_test
        self.metric = {}

  def set_paths(self):
    if self.noise:
      root = "experiments/noise"
    else:
      root = "experiments"
    print("root: ", root)
    self.data_path = os.path.join(root,"data")
    self.model_path = os.path.join(root,"model")
    self.train_data_path = os.path.join(root, "train")
    self.test_data_path = os.path.join(root, "test")
    self.plots_path = os.path.join(root,"plots")
    self.dataset_path = os.path.join("dataset", self.dataset, "dataset_ref.csv")

  def base_model(self):
    
    self.model_name = "logistic-regression"
    train_data_filename = "train_data_" + make_experiment_filename(dataset = self.dataset, demo_baseline = self.demo_baseline, lr_theta = self.lr_theta, num_of_demos = self.num_of_demos, noise_ratio = self.noise_ratio) + ".csv"
    train_file_path = os.path.join(self.train_data_path, train_data_filename)
    self.train_data = pd.read_csv(train_file_path, index_col=0)

    A = self.train_data[self.sensitive_feature]
    A_str = A.map(self.dict_map)
    # Extract the target
    Y = self.train_data[self.label]
    X = self.train_data.drop(columns=[self.label])

    
    
    X_train, X_test, Y_train, Y_test, A_train, A_test, A_str_train, A_str_test = train_test_split(
        X,
        Y,
        A,
        A_str,
        test_size = 1 - alpha,
        random_state=12345,
        stratify=Y
        )
    # ###############################################
    # balanced_idx1 = X_train[Y_train==1].index
    # pp_train_idx = balanced_idx1.union(Y_train[Y_train==0].sample(n=balanced_idx1.size, random_state=1234).index)
    # X_train = X_train.loc[pp_train_idx, :]
    # Y_train = Y_train.loc[pp_train_idx]
    # A_train = A_train.loc[pp_train_idx]
    # #################################################

    self.model_obj = LR_base_superhuman_model() #LogisticRegression(**self.logi_params)
    print("self.model_obj.type: ", self.model_obj.type)
    self.model_obj.fit(X_train, Y_train)
    self.pred_scores = self.model_obj.predict_proba(X_test)

    if self.dataset == 'COMPAS':
      mode = 'demographic_parity' #'equalized_opportunity' # #'equalized_odds'
      C = .005
      if mode == 'demographic_parity':
        h = DP_fair_logloss_classifier(C=C, random_initialization=True, verbose=False)
      elif mode == 'equalized_opportunity':
        h = EOPP_fair_logloss_classifier(C=C, random_initialization=True, verbose=False)
      elif mode == 'equalized_odds':
        h = EODD_fair_logloss_classifier(C=C, random_initialization=True, verbose=False)    
      else:
        raise ValueError('Invalid second arg')
    
      Y_train = Y_train.astype('float64')
      Y_test = Y_test.astype('float64')
      A_train = A_train.astype('float64')
      A_test = A_test.astype('float64')
      if self.dataset == 'Adult':
        A_train = A_train - 1
        A_test = A_test - 1

      for c in list(X_train.columns):
          if X_train[c].min() < 0 or X_train[c].max() > 1:
              mu = X_train[c].mean()
              s = X_train[c].std(ddof=0)
              X_train.loc[:,c] = (X_train[c] - mu) / s
              X_train.loc[:,c] = (X_train[c] - mu) / s
      
      h.fit(X_train.values,Y_train.values,A_train.values)
      self.pred_scores = h.predict(X_test.values,A_test.values)
      
      print("h.theta: ", h.theta)
      self.model_obj.update_model_theta(h.theta[:-1])
      #self.model_obj.coef_ = np.asarray([h.theta[:-1]])
      print("self.model_obj.coef_: ", self.model_obj.get_model_theta())

    self.base_dict = {"model_obj": self.model_obj,
                 "pred_scores": self.pred_scores,
                 "model_name": self.model_name,
                 "logi_params": self.logi_params,
                 "X_train": X_train,
                 "Y_train": Y_train,
                 "X_test": X_test,
                 "Y_test": Y_test,
                 "A_str_train": A_str_train,
                 "A_str_test": A_str_test}
    
    model_file_dir = os.path.join(self.model_path, 'base_model_' + dataset + '.pickle') 
    with open(model_file_dir, 'wb') as handle:  #with open(f'base_model_{dataset}.pickle', 'wb') as handle:
        pickle.dump(self.base_dict, handle)

  def run_demo_baseline(self, data_demo = data_demo):
    X_train = data_demo.train_x
    Y_train = data_demo.train_y
    A_train = data_demo.train_A
    A_str_test = data_demo.test_A_str
    X_test = data_demo.test_x
    Y_test = data_demo.test_y
    A_test = data_demo.test_A
    #Super_human.model_name = model

    if self.demo_baseline == "pp":
      model_logi = LogisticRegression(**self.logi_params)
      model_logi.fit(X_train, Y_train)
      # Post-processing
      self.postprocess_est = ThresholdOptimizer(
          estimator=model_logi,
          constraints="demographic_parity", #"equalized_odds",
          predict_method='auto',
          prefit=True)
      # Balanced data set is obtained by sampling the same number of points from the majority class (Y=0)
      # as there are points in the minority class (Y=1)
      balanced_idx1 = X_train[Y_train==1].index
      pp_train_idx = balanced_idx1.union(Y_train[Y_train==0].sample(n=balanced_idx1.size, random_state=1234).index)
      X_train_balanced = X_train.loc[pp_train_idx, :]
      Y_train_balanced = Y_train.loc[pp_train_idx]
      A_train_balanced = A_train.loc[pp_train_idx]

      # Post-process fitting
      self.postprocess_est.fit(X_train_balanced, Y_train_balanced, sensitive_features=A_train_balanced)
      # Post-process preds
      baseline_preds = self.postprocess_est.predict(X_test, sensitive_features=A_test)

    elif self.demo_baseline == "fair_logloss":
      mode = 'equalized_opportunity' #'equalized_odds'
      C = .005
      if mode == 'demographic_parity':
        h = DP_fair_logloss_classifier(C=C, random_initialization=True, verbose=False)
      elif mode == 'equalized_opportunity':
        h = EOPP_fair_logloss_classifier(C=C, random_initialization=True, verbose=False)
      elif mode == 'equalized_odds':
        h = EODD_fair_logloss_classifier(C=C, random_initialization=True, verbose=False)    
      else:
        raise ValueError('Invalid second arg')
    
      Y_train = Y_train.astype('float64')
      Y_test = Y_test.astype('float64')
      A_train = A_train.astype('float64')
      A_test = A_test.astype('float64')
      A_train = A_train - 1
      A_test = A_test - 1

      for c in list(X_train.columns):
          if X_train[c].min() < 0 or X_train[c].max() > 1:
              mu = X_train[c].mean()
              s = X_train[c].std(ddof=0)
              X_train.loc[:,c] = (X_train[c] - mu) / s
              X_train.loc[:,c] = (X_train[c] - mu) / s
      
      h.fit(X_train.values,Y_train.values,A_train.values)
      baseline_preds = h.predict(X_test.values,A_test.values)
    

    ################################################################
    if self.noise == True:
        baseline_preds = self.add_noise(baseline_preds, protected=False) # add noise to the predicted label
        #A_test_noisy = self.add_noise(A_test, protected=True)    # add noise to the protected attribute
        #Y_test_noisy = self.add_noise(Y_test, protected=False)
    # Metrics
    models_dict = {
              self.demo_baseline : (baseline_preds, baseline_preds)} 
    ##############################################################
    # # Metrics
    # models_dict = {
    #           self.demo_baseline : (baseline_preds, baseline_preds)}          
    result = get_metrics_df(models_dict = models_dict, y_true = Y_test, group = A_str_test, feature = feature, is_demo = True)

    return result


    
  def split_data(self, model, alpha=0.5, dataset=None,  mode="post-processing"):
    # Assign dataset to temporary variable
    dataset_temp = dataset.copy(deep=True)
    # Extract the sensitive feature
    A = dataset_temp[self.sensitive_feature]
    A_str = A.map(self.dict_map)
    # Extract the target
    Y = dataset_temp[self.label]
    
    ####################################
    if mode == "post-processing":
      dataset_temp.drop(columns=[self.sensitive_feature])
      if 'prev_index' in dataset_temp.columns:
        idx = dataset_temp["prev_index"]
        dataset_temp = dataset_temp.drop(columns=['prev_index'])
    elif mode == "normal":
      idx = dataset_temp.index.tolist()
    ###################################
    X = dataset_temp.drop(columns=[self.label])
    # TRAIN TEST SPLIT
    # alpha is the percentage of (Train + Test of Post Processing)
    # beta is the percentage of train Post Processing and 1 - beta is percentage of Test Post Processing
    
    df_train, df_test, Y_train, Y_test, A_train, A_test, A_str_train, A_str_test, idx_train, idx_test = train_test_split(
        X,
        Y,
        A,
        A_str,
        idx,
        test_size = 1 - alpha,
        random_state=12345,
        stratify=Y
        )
    
    new_demo = self.data_demo(df_train, df_test, Y_train, Y_test, A_train, A_test, A_str_train, A_str_test, idx_train, idx_test)
    return new_demo

  def prepare_test_pp(self, model = "logistic_regression", alpha = 0.5, beta = 0.5):
  
    self.dataset_ref = pd.read_csv(self.dataset_path, index_col=0) #self.dataset_ref = pd.read_csv('dataset_ref.csv', index_col=0)
    dataset_ref = self.dataset_ref.copy(deep=True)
    # convert dataset_ref[senitive_feature] to int
    dataset_ref[self.sensitive_feature] = dataset_ref[self.sensitive_feature].astype(int)
    
    self.test_pp_logi = pd.DataFrame(index = [self.feature[i] for i in range(self.num_of_features)])
    self.demo_list = []
    r = random.randint(0, 10000000)
    dataset_ref = shuffle(dataset_ref, random_state=r)
    ####################################
    A = dataset_ref[self.sensitive_feature]
    A_str = A.map(self.dict_map)
    Y = dataset_ref[self.label]
    idx = dataset_ref.index.tolist()
    
    X = dataset_ref.drop(columns=[self.label])
    # print(X)
    # exit()
    df_train, df_test, Y_train, Y_test, A_train, A_test, A_str_train, A_str_test, idx_train, idx_test = train_test_split(
        X,
        Y,
        A,
        A_str,
        idx,
        test_size = 1 - alpha,
        random_state=12345,
        stratify=Y
        )
    dataset_pp = dataset_ref.iloc[idx_train].reset_index(drop=True) # use only pp portion of the data and leave SH Test portion
    dataset_sh = dataset_ref.iloc[idx_test].reset_index(drop=True)

    
    train_data_filename = "train_data_" + make_experiment_filename(dataset = self.dataset, demo_baseline = self.demo_baseline, lr_theta = self.lr_theta, num_of_demos = self.num_of_demos, noise_ratio = self.noise_ratio) + ".csv"
    train_file_path = os.path.join(self.train_data_path, train_data_filename)
    print("u running 5?")
    print(train_file_path)
    print(train_data_filename)
    test_data_filename = "test_data_" + make_experiment_filename(dataset = self.dataset, demo_baseline = self.demo_baseline, lr_theta = self.lr_theta, num_of_demos = self.num_of_demos, noise_ratio = self.noise_ratio) + ".csv"
    test_file_path = os.path.join(self.test_data_path, test_data_filename)
    

    dataset_pp.to_csv(train_file_path)
    dataset_sh.to_csv(test_file_path)
    
    dataset_pp_copy = dataset_pp.copy(deep=True)
    
    
    for i in range(self.num_of_demos):
      r = random.randint(0, 10000000)
      ##########################################################################
      index_list = dataset_pp_copy.index.tolist()
      dataset_temp = shuffle(dataset_pp_copy, random_state=r)
      dataset_temp['prev_index'] = index_list
      ##########################################################################
      new_demo = self.split_data(model, alpha=beta, dataset=dataset_temp,  mode="post-processing")
      if self.noise == True:
        new_demo = self.add_noise_new(new_demo)
      print("rnning metrics")
      metrics = self.run_demo_baseline(data_demo = new_demo) #self.run_logistic_pp(model = model, data_demo = new_demo)
      #if self.noise == True and metrics[self.demo_baseline]["Demographic parity difference"]
      # exit()
      print("demo metrics: ")
      print(metrics)
      print("-----------------------------------")
      self.test_pp_logi[i] = metrics
      new_demo.metric_df = metrics
      for k in range(self.num_of_features):
        new_demo.metric[k] = new_demo.metric_df.loc[self.feature[k]][self.demo_baseline]

      self.demo_list.append(new_demo)
      print("demo {}".format(i))

    file_dir = os.path.join(self.data_path)
    demo_list_filename = make_demo_list_filename(dataset = self.dataset, demo_baseline = self.demo_baseline, num_of_demos = self.num_of_demos, noise_ratio = self.noise_ratio)
    store_object(self.demo_list, file_dir, demo_list_filename, -1)

  def save_AXY(self, A_train, X_train, Y_train, A_test, X_test, Y_test):

    A_train.to_csv(os.path.join("dataset", self.dataset, 'A_train.csv'),  sep='\t')
    X_train.to_csv(os.path.join("dataset", self.dataset, 'X_train.csv'),  sep='\t')
    Y_train.to_csv(os.path.join("dataset", self.dataset, 'Y_train.csv'),  sep='\t')

    A_test.to_csv(os.path.join("dataset", self.dataset, 'A_test.csv'),  sep='\t')
    X_test.to_csv(os.path.join("dataset", self.dataset, 'X_test.csv'),  sep='\t')
    Y_test.to_csv(os.path.join("dataset", self.dataset, 'Y_test.csv'),  sep='\t')
    

  def add_noise(self, data, protected=False):
    if protected:
      name = self.sensitive_feature
      Y = data.to_frame(name=name).reset_index(drop=True)
    else: 
      name = self.label
      Y = pd.DataFrame(data, columns=[self.label]).reset_index(drop=True)
    
    n = len(Y)
    #print(n)
    noisy_Y = copy.deepcopy(Y)
    idx = np.random.permutation(range(n))[:int(self.noise_ratio*n)]
    #print(idx)
    if protected==True and (self.dataset == 'Adult' or dataset == 'Diabetes'): # checked! works well!
      Y['gender'].loc[idx] = Y['gender'].loc[idx] - 1   # change 1,2 to 0,1
      noisy_Y['gender'].loc[idx] = 1-Y['gender'].loc[idx] # change 0,1 to 1,0
      noisy_Y['gender'].loc[idx] = noisy_Y['gender'].loc[idx] + 1   # revert 1,0 to 2,1
    elif protected==True and self.dataset == 'COMPAS':
      noisy_Y['race'].loc[idx] = 1-Y['race'].loc[idx]
    elif protected==False:
      noisy_Y[self.label].loc[idx] = 1 - Y[self.label].loc[idx] # if adding noise to the label
    
    return noisy_Y

  def add_noise_new(self, data_demo):  # works fine!

    Y_train = data_demo.train_y
    A_str_test = data_demo.test_A_str
    A_test = data_demo.test_A

    n_Y = len(Y_train)
    n_A = len(A_test)

    idx_Y = np.random.permutation(range(n_Y))[:int(self.noise_ratio*n_Y)]
    idx_A = np.random.permutation(range(n_A))[:int(self.noise_ratio*n_A)]

    A_test_index = A_test.index
    Y_train_index = Y_train.index

    A_test = A_test.reset_index(drop=True)
    Y_train = Y_train.reset_index(drop=True)

    noisy_A_test = copy.deepcopy(A_test)
    noisy_Y_train = copy.deepcopy(Y_train)
    # flip protected attribute
    if self.dataset == 'Adult':
      A_test.loc[idx_A] = A_test.loc[idx_A] - 1   # change 1,2 to 0,1
      noisy_A_test.loc[idx_A] = 1 - A_test.loc[idx_A] # change 0,1 to 1,0
      noisy_A_test.loc[idx_A] = noisy_A_test.loc[idx_A] + 1   # revert 1,0 to 2,1
    elif self.dataset == 'COMPAS':
      noisy_A_test.loc[idx_A] = 1 - A_test.loc[idx_A] # change 0,1 to 1,0
    # flip label
    noisy_Y_train.loc[idx_Y] = 1 - Y_train.loc[idx_Y] # if adding noise to the label

    noisy_A_test.index = A_test_index
    noisy_Y_train.index = Y_train_index
    noisy_A_test_str = noisy_A_test.map(self.dict_map)
    noisy_A_test_str.index = A_test_index
    
    data_demo.test_A = noisy_A_test
    data_demo.test_A_str = noisy_A_test_str
    data_demo.train_y = noisy_Y_train

    return data_demo


  def read_demo_list(self):
    file_dir = os.path.join(self.data_path)
    print("file_dir in read_demo_list: ", file_dir)
    demo_list_filename = make_demo_list_filename(dataset = self.dataset, demo_baseline = self.demo_baseline, num_of_demos = self.num_of_demos, noise_ratio = self.noise_ratio)
    print("demo_list file name: ")
    print(demo_list_filename)
    self.demo_list = load_object(file_dir, demo_list_filename, -1)
    
    return self.demo_list

  def get_model_pred(self, item): # an item is one row of the dataset
    score = self.model_obj.predict_proba(item).squeeze() # [p(y = 0), p(y = 1)]
    return score

  def sample_from_prob(self, dist, size):

    preds = [0.0, 1.0]
    #print("dist: ", dist)
    #print(sum(dist))
    dist /= dist.sum()
    sample_preds = np.random.choice(preds, size, True, dist)
    return sample_preds

  def sample_superhuman(self):
    start_time = time.time()

    train_data_filename = "train_data_" + make_experiment_filename(dataset = self.dataset, demo_baseline = self.demo_baseline, lr_theta = self.lr_theta, num_of_demos = self.num_of_demos, noise_ratio = self.noise_ratio) + ".csv"
    train_file_path = os.path.join(self.train_data_path, train_data_filename)

    self.train_data = pd.read_csv(train_file_path, index_col=0)
    X = self.train_data.drop(columns=[self.label]).to_numpy()
    data_size, feature_size = self.train_data.shape
    self.sample_matrix = np.zeros((self.num_of_demos, data_size)) #np.array([[-1 for _ in range(data_size)] for _ in range(num_of_samples)]) # create a matrix of size [num_of_samples * data_set_size]. Each row is a sample from our model that predicts the self.label of dataset.
    for j in range(data_size):
      probs = self.get_model_pred(item = [X[j]])
      self.sample_matrix[:,j] = self.sample_from_prob(dist = probs, size = self.num_of_demos) # return a vector of size num_of_samples (50) with self.label prediction samples for j_th item of the dataset
          
    print("--- %s end of sample_superhuman ---" % (time.time() - start_time))
    return self.sample_matrix

  def get_samples_demo_indexed(self):
    start_time = time.time()
    sample_size = self.demo_list[0].idx_test.size
    self.sample_matrix_demo_indexed = np.zeros((self.num_of_demos, sample_size))
    
    for i in range(self.num_of_demos):
      demo = self.demo_list[i]
      self.sample_matrix_demo_indexed[i,:] = self.sample_matrix[i,:][demo.idx_test]
    return self.sample_matrix_demo_indexed
    

  def get_sample_loss(self):
    start_time = time.time()
    self.sample_loss = np.zeros((self.num_of_demos, self.num_of_features))
    for demo_index, x in enumerate(tqdm(self.demo_list)):
      demo = self.demo_list[demo_index]
      sample_preds = self.sample_matrix_demo_indexed[demo_index,:]
      # Metrics
      models_dict = {"Super_human": (sample_preds, sample_preds)}
      y = self.train_data.loc[demo.idx_test][self.label] # we use true_y from original dataset since y_true in demo can be noisy (in noise setting)
      A = self.train_data.loc[demo.idx_test][self.sensitive_feature]
      A_str = A.map(self.dict_map)
      metric_df = get_metrics_df(models_dict = models_dict, y_true = y, group = A_str, feature = feature, is_demo = False) #### takes so much time!!! #metric_df = get_metrics_df(models_dict = models_dict, y_true = demo.test_y, group = demo.test_A_str)
      for feature_index in range(self.num_of_features):
        self.sample_loss[demo_index, feature_index] = metric_df.loc[self.feature[feature_index]]["Super_human"] #metric[feature_index]
    
    print("--- %s end of get_sample_loss ---" % (time.time() - start_time))

  def get_demo_loss(self, demo_index, feature_index):
    demo_loss =self.demo_list[demo_index].metric[feature_index]
    return demo_loss

  def get_subdom_constant(self):
    if self.c == None:  # only update it in the first iteration with the initial sample values
      subdom_constant = np.mean(self.subdom_tensor)
    return subdom_constant

  def get_subdom_tensor(self):
    self.subdom_tensor = np.load('subdom_tensor.npy')
    return self.subdom_tensor

  def compute_exp_phi_X_Y(self):
    
    train_data_filename = "train_data_" + make_experiment_filename(dataset = self.dataset, demo_baseline = self.demo_baseline, lr_theta = self.lr_theta, num_of_demos = self.num_of_demos, noise_ratio = self.noise_ratio) + ".csv"
    train_file_path = os.path.join(self.train_data_path, train_data_filename)

    self.train_data = pd.read_csv(train_file_path, index_col=0)
    X = self.train_data.drop(columns=[self.label]) #self.dataset_ref.drop(columns=['self.label'])
    self.exp_phi_X_Y = [0 for _ in range(self.num_of_attributs)]
    self.phi_X_Y = []
    for i in range(self.num_of_demos):
      demo = self.demo_list[i]
      sample_Y = self.sample_matrix_demo_indexed[i,:]
      phi_X_Y_temp = np.reshape(sample_Y, (-1, 1)) * X.loc[demo.idx_test]
      phi_X_Y_temp = np.sum(phi_X_Y_temp, axis=0) / X.shape[0]
      self.phi_X_Y.append(phi_X_Y_temp)
      self.exp_phi_X_Y += phi_X_Y_temp
    self.exp_phi_X_Y /= self.num_of_demos  # get the average

  def feature_matching(self, demo_index):
    demo = self.demo_list[demo_index]
    X_demoIndexed = self.train_data.drop(columns=[self.label]).loc[demo.idx_test] #self.dataset_ref.drop(columns=['self.label']).loc[demo.idx_test]
    sample_Y_demoIndexed = self.sample_matrix_demo_indexed[demo_index,:]#[demo.idx_test]
    phi_X_Y = np.reshape(sample_Y_demoIndexed, (-1, 1)) * X_demoIndexed
    phi_X_Y = np.sum(phi_X_Y, axis=0) / X_demoIndexed.shape[0]

    return phi_X_Y - self.exp_phi_X_Y


  def compute_grad_theta(self):
    start_time = time.time()
    self.subdom_tensor = np.zeros((self.num_of_demos, self.num_of_features)) 
    self.compute_exp_phi_X_Y() 
    grad_theta = [0.0 for _ in range(self.num_of_attributs)]
    for j, x in enumerate(tqdm(self.demo_list)):
      if j == 0: self.subdom_constant = 0
      else: self.subdom_constant = self.get_subdom_constant()
      for k in range(self.num_of_features):
        sample_loss = self.sample_loss[j, k] #self.get_sample_loss(j, k)
        demo_loss = self.demo_list[j].metric[k]
        self.subdom_tensor[j, k] = max(self.alpha[k]*(sample_loss - demo_loss) + 1, 0) - self.subdom_constant     # subtract constant c to optimize for useful demonstation instead of avoiding from noisy ones
        grad_theta += self.subdom_tensor[j, k] * self.feature_matching(j)
        
    print("--- %s end of compute_grad_theta ---" % (time.time() - start_time))
    subdom_tensor_sum = np.sum(self.subdom_tensor)
    print("subdom tensor sum: ", subdom_tensor_sum)
    #np.save('subdom_tensor.npy', self.subdom_tensor)
    return subdom_tensor_sum, grad_theta

  # def compute_alpha(self):
  #   start_time = time.time()
  #   alpha = np.ones(self.num_of_features)
  
  #   for k in range(self.num_of_features):
  #     sorted_demos = []
  #     alpha_candidate = []
  #     for j in range(self.num_of_demos):
  #       sample_loss = self.sample_loss[j, k]
  #       demo_loss = self.demo_list[j].metric[k] 
  #       sorted_demos.append((demo_loss, sample_loss))
      
  #     sorted_demos.sort(key = lambda x: x[0]) #dominated_demos.sort(key = lambda x: x[0], reverse=True)   # sort based on demo loss
  #     sorted_demos = np.array(sorted_demos)
  #     alpha[k] = np.mean(self.alpha) #100 #max(self.alpha) #np.mean(self.alpha) # default value in case it didn't change using previous alpha values
  #     for m, demo in enumerate(sorted_demos):
  #       avg_sample_loss = np.mean([demo[1] for demo in sorted_demos])
  #       if (demo[0] > demo[1]):
  #         alpha[k] = min(100, 1.0/(demo[0] - demo[1]))  ### limit max alpha to 100
  #       if (avg_sample_loss + self.lamda) <= np.mean([x[0] for x in sorted_demos[0:m+1]]): #if (demo[2]) <= np.mean([x[1] for x in dominated_demos[0:m+1]] and demo[0] > 0):
  #         break

  #   print("--- %s end of compute_alpha ---" % (time.time() - start_time))
  #   print("alpha : ")
  #   print(alpha)
  #   #model_params = {"eval": self.eval}
  #   find_gamma_superhuman(self.demo_list, self.model_params)
  #   return alpha

  def compute_alpha(self):
    start_time = time.time()
    alpha = np.ones(self.num_of_features)
  
    for k in range(self.num_of_features):
      sorted_demos = []
      alpha_candidate = []
      for j in range(self.num_of_demos):
        sample_loss = self.sample_loss[j, k]
        demo_loss = self.demo_list[j].metric[k] 
        sorted_demos.append((demo_loss, sample_loss))
      
      sorted_demos.sort(key = lambda x: x[0]) #dominated_demos.sort(key = lambda x: x[0], reverse=True)   # sort based on demo loss
      #print(self.feature[k])
      #print("demo_loss, sample_loss: ")
      #print(sorted_demos)
      sorted_demos = np.array(sorted_demos)
      alpha[k] = 100 #max(self.alpha) #np.mean(self.alpha) # default value in case it didn't change using previous alpha values
      # print("alpha {}", k)
      # print(alpha)
      for m, demo in enumerate(sorted_demos):
        if (demo[0] > demo[1]):
          alpha[k] = min(100, 1.0/(demo[0] - demo[1]))  ### limit max alpha to 100
        if (demo[1] + self.lamda) <= np.mean([x[0] for x in sorted_demos[0:m+1]]): #if (demo[2]) <= np.mean([x[1] for x in dominated_demos[0:m+1]] and demo[0] > 0):
          break

    print("--- %s end of compute_alpha ---" % (time.time() - start_time))
    print("alpha : ")
    print(alpha)
    #model_params = {"eval": self.eval}
    find_gamma_superhuman(self.demo_list, self.model_params)
    return alpha

  def eval_model_baseline(self, baseline="pp", mode="demographic_parity"):

    train_data_filename = "train_data_" + make_experiment_filename(dataset = self.dataset, demo_baseline = self.demo_baseline, lr_theta = self.lr_theta, num_of_demos = self.num_of_demos, noise_ratio = self.noise_ratio) + ".csv"
    train_file_path = os.path.join(self.train_data_path, train_data_filename)
    test_data_filename = "test_data_" + make_experiment_filename(dataset = self.dataset, demo_baseline = self.demo_baseline, lr_theta = self.lr_theta, num_of_demos = self.num_of_demos, noise_ratio = self.noise_ratio) + ".csv"
    test_file_path = os.path.join(self.test_data_path, test_data_filename)

    self.train_data = pd.read_csv(train_file_path, index_col=0)
    self.test_data = pd.read_csv(test_file_path, index_col=0)
    A_train = self.train_data[self.sensitive_feature]
    A_test = self.test_data[self.sensitive_feature]
    A_str_train = A_train.map(self.dict_map)
    A_str_test = A_test.map(self.dict_map)
    # Extract the target
    Y_train = self.train_data[self.label]
    Y_test = self.test_data[self.label]
    X_train = self.train_data.drop(columns=[self.label])
    X_test = self.test_data.drop(columns=[self.label])
    ### Add noise
    if self.noise == True:
      demo = self.data_demo(X_train, X_train, Y_train, Y_train, A_train, A_train, A_str_train, A_str_train, X_train.index, X_train.index)
      demo = self.add_noise_new(demo) # in this function we only add noise to train_Y and test_A: so instead of test_A we use train_A in the input
      Y_train, A_train = demo.train_y, demo.test_A
      if self.dataset == 'Adult': 
        self.save_AXY(A_train - 1, X_train, Y_train, A_test - 1, X_test, Y_test)
      elif self.dataset == 'COMPAS':
        self.save_AXY(A_train, X_train, Y_train, A_test, X_test, Y_test)


    if baseline == "pp":

      model_logi = LogisticRegression(**self.logi_params)
      model_logi.fit(X_train, Y_train)

      # Post-processing
      postprocess_est = ThresholdOptimizer(
          estimator=model_logi,
          constraints=mode,
          predict_method='auto',
          prefit=True)
      # Balanced data set is obtained by sampling the same number of points from the majority class (Y=0)
      # as there are points in the minority class (Y=1)
      balanced_idx1 = X_train[Y_train==1].index
      pp_train_idx = balanced_idx1.union(Y_train[Y_train==0].sample(n=balanced_idx1.size, random_state=1234).index)
      X_train_balanced = X_train.loc[pp_train_idx, :]
      Y_train_balanced = Y_train.loc[pp_train_idx]
      A_train_balanced = A_train.loc[pp_train_idx]
      # Post-process fitting

      postprocess_est.fit(X_train_balanced, Y_train_balanced, sensitive_features=A_train_balanced)
      # Post-process preds
      baseline_preds = postprocess_est.predict(X_test, sensitive_features=A_test)
       # Metrics
      models_dict = {
                baseline+"_"+mode : (baseline_preds, baseline_preds)}
      return get_metrics_df(models_dict = models_dict, y_true = Y_test, group = A_str_test, feature = feature, is_demo = False)
    
    elif baseline == "fair_logloss":
      C = .005
      if mode == 'demographic_parity':
        h = DP_fair_logloss_classifier(C=C, random_initialization=True, verbose=False)
      elif mode == 'equalized_opportunity':
        h = EOPP_fair_logloss_classifier(C=C, random_initialization=True, verbose=False)
      elif mode == 'equalized_odds':
        h = EODD_fair_logloss_classifier(C=C, random_initialization=True, verbose=False)    
      else:
        raise ValueError('Invalid second arg')
    
      if dataset == 'Adult' or dataset == 'Diabetes':
        A_train = A_train - 1
        A_test = A_test - 1

      h.fit(X_train.values,Y_train.values,A_train.values)
      baseline_preds = h.predict(X_test.values,A_test.values)
      baseline_scores = h.predict_proba(X_test.values,A_test.values) 
      baseline_preds[np.isnan(baseline_preds)] = 1
      violation = h.fairness_violation(X_test.values, Y_test.values, A_test.values)
      accuracy = h.score(X_test.values, Y_test.values, A_test.values) 
      err, exp_zeroone = compute_error(baseline_preds, baseline_scores, Y_test.values)
      print(baseline+" "+mode+" violation: ")
      print(violation)
      print("expected_error: ")
      print(exp_zeroone)
      print()
       # Metrics
      models_dict = {
                baseline+"_"+mode : (baseline_preds, baseline_preds)}
      metrics = get_metrics_df(models_dict = models_dict, y_true = Y_test, group = A_str_test, feature = feature, is_demo = False)
      # Since fair logloss uses expected violation, we use metrics from their code
      metrics[baseline+"_"+mode]['ZeroOne'] = exp_zeroone
      if mode == "demographic_parity":
        metrics[baseline+"_"+mode]['Demographic parity difference'] = violation
      elif mode == "equalized_odds":
        metrics[baseline+"_"+mode]['Equalized odds difference'] = violation

      return metrics
    
    elif baseline == "MFOpt":
      test_data_filename = "MFOpt_" + dataset + ".csv"
      if self.noise == True: 
        test_data_filename = "MFOpt_" + dataset +"_0-2"+ ".csv"
      test_file_path = os.path.join(self.test_data_path, test_data_filename)
      #self.train_data = pd.read_csv(train_file_path, index_col=0)
      self.test_data = pd.read_csv(test_file_path, index_col=0)
      #A_train = self.train_data[self.sensitive_feature]
      A_test = self.test_data['g']
      #A_str_train = A_train.map(self.dict_map)
      A_str_test = A_test.map(self.dict_map)
      # Extract the target
      #Y_train = self.train_data['bin']
      Y_test = self.test_data['y']
      baseline_preds = self.test_data['bin'] - 1
      baseline_scores = self.test_data.index
      err, exp_zeroone = compute_error(baseline_preds, baseline_scores, Y_test.values)
      print("expected_error: ")
      print(exp_zeroone)
      #X_train = self.train_data.drop(columns=[self.label])
      #X_test = self.test_data.drop(columns=[self.label])
       # Metrics
      models_dict = {
                baseline : (baseline_preds, baseline_preds)}
      metrics = get_metrics_df(models_dict = models_dict, y_true = Y_test, group = A_str_test, feature = feature, is_demo = False)
      metrics[baseline]['ZeroOne'] = exp_zeroone
      return metrics


  def eval_model(self, mode):
    if mode == "train":
      train_data_filename = "train_data_" + make_experiment_filename(dataset = self.dataset, demo_baseline = self.demo_baseline, lr_theta = self.lr_theta, num_of_demos = self.num_of_demos, noise_ratio = self.noise_ratio) + ".csv"
      train_file_path = os.path.join(self.train_data_path, train_data_filename)

      self.train_data = pd.read_csv(train_file_path, index_col=0)
      A = self.train_data[self.sensitive_feature]
      A_str = A.map(self.dict_map)
      # Extract the target
      Y_train = self.train_data[self.label]
      Y_test = Y_train
      X = self.train_data.drop(columns=[self.label])
      
    elif mode == "test-sh" or mode == "test-pp":
      # read train data: we need Y_train to predict test data
      train_data_filename = "train_data_" + make_experiment_filename(dataset = self.dataset, demo_baseline = self.demo_baseline, lr_theta = self.lr_theta, num_of_demos = self.num_of_demos, noise_ratio = self.noise_ratio) + ".csv"
      train_file_path = os.path.join(self.train_data_path, train_data_filename)
      self.train_data = pd.read_csv(train_file_path, index_col=0)
      Y_train = self.train_data[self.label]
      ## read test data
      test_data_filename = "test_data_" + make_experiment_filename(dataset = self.dataset, demo_baseline = self.demo_baseline, lr_theta = self.lr_theta, num_of_demos = self.num_of_demos, noise_ratio = self.noise_ratio) + ".csv"
      test_file_path = os.path.join(self.test_data_path, test_data_filename)
      self.test_data = pd.read_csv(test_file_path, index_col=0)
      A = self.test_data[self.sensitive_feature]
      A_str = A.map(self.dict_map)
      Y_test = self.test_data[self.label]
      X = self.test_data.drop(columns=[self.label])

    # Scores on train set
    #print("len(X): ", X.shape)
    #print("len(Y): ", Y_test.shape)
    #print("self.model_obj.predict_proba(X):", self.model_obj.predict_proba(X))
    scores = self.model_obj.predict_proba(X)[:, 1]
    # Predictions (0 or 1) on test set
    preds = (scores >= np.mean(Y_train)) * 1
    #preds = predict_nn(X, Y_train, A, X, Y_test, A, self)
    # Metrics
    eval = pd.DataFrame(index = [self.feature[i] for i in range(self.num_of_features)]) #['Demographic parity difference', 'False negative rate difference', 'ZeroOne']
    models_dict = {
              "Super_human": (preds, preds)}
    eval = get_metrics_df(models_dict = models_dict, y_true = Y_test, group = A_str, feature = feature, is_demo = False)
    return eval

  # def get_model_theta(self):
  #   return self.model_obj.get_model_theta()  #self.model_obj.coef_[0]

  # def update_model_theta(self, new_theta):
  #   self.model_obj.update_model_theta(new_theta)
  #   #self.model_obj.coef_ = np.asarray([new_theta]) # update the coefficient of our logistic regression model with the new theta
  
  def get_model_alpha(self):
    return self.alpha

  def update_model_alpha(self, new_alpha):
    self.alpha = new_alpha
  

  def update_model(self, lr_theta, iters):
    self.lr_theta = lr_theta
    self.grad_theta, subdom_tensor_sum_arr, self.eval, self.gamma_superhuman_arr = [], [], [], []
    Y = self.train_data[self.label]
    X = self.train_data.drop(columns=[self.label])
    print("self.base_model_type: ", self.model_obj.type)
    if self.base_model_type == 'NN':
      self.subdom_constant = 0
      self.sample_superhuman() # update self.sample_matrix with new samples from new theta
      self.get_samples_demo_indexed()
      self.get_sample_loss()
      self.model_obj = NN_base_superhuman_model(self.demo_list, self.num_of_demos, self.num_of_features, self.subdom_constant, self.alpha, self.sample_loss)
      self.model_obj.fit(X, Y)
    gamma_degrade = 0
    for i in tqdm(range(iters)):
      # find sample loss and store it, we will use it for computing grad_theta and grad_alpha
      self.sample_superhuman() # update self.sample_matrix with new samples from new theta
      self.get_samples_demo_indexed()
      self.get_sample_loss()
      alpha = self.get_model_alpha()

      if isinstance(self.model_obj, LR_base_superhuman_model):
        print("model is LR_base_superhuman_model")
        # get the current theta and alpha
        theta = self.model_obj.get_model_theta()
        
        # find new theta
        subdom_tensor_sum, grad_theta = self.compute_grad_theta() # computer gradient of loss w.r.t theta by sampling from our model
        new_theta = theta - self.lr_theta * grad_theta  # update theta using the gradinet values
        # find new alpha
        if i == 0:
          print("eval from first sample: ")
          print(self.eval_model(mode = "train"))
        new_alpha = self.compute_alpha()
        #update theta
        self.model_obj.update_model_theta(new_theta)
        self.grad_theta.append(grad_theta)
        #update alpha
        
      elif isinstance(self.model_obj, NN_base_superhuman_model):
        print("model is NN_base_superhuman_model")
         # get the current theta and alpha
        #theta = self.model_obj.get_model_theta()
        self.model_obj.fit(X, Y)
        # find new theta
        #subdom_tensor_sum, grad_theta = self.compute_grad_theta() # computer gradient of loss w.r.t theta by sampling from our model
        #new_theta = theta - self.lr_theta * grad_theta  # update theta using the gradinet values
        # find new alpha
        if i == 0:
          print("eval from first sample: ")
          print(self.eval_model(mode = "train"))
        new_alpha = self.compute_alpha()
        subdom_tensor_sum = 0 #########
        #update theta
        #self.model_obj.update_model_theta(new_theta)
        #update alpha

      self.update_model_alpha(new_alpha)
      # eval model
      eval_i = self.eval_model(mode = "train")
      print("eval_i:")
      print(eval_i)
      # store some stuff
      subdom_tensor_sum_arr.append(subdom_tensor_sum)
      self.eval.append(eval_i)
      model_params = {"model":self.model_obj, "theta": self.model_obj.get_model_theta(), "alpha":self.alpha, "eval": self.eval, "subdom_value": subdom_tensor_sum_arr, "lr_theta": self.lr_theta, "num_of_demos":self.num_of_demos, "iters": iters, "num_of_features": self.num_of_features, "demo_baseline": self.demo_baseline, "feature": self.feature}
      gamma_superhuman = find_gamma_superhuman(self.demo_list, model_params)
      self.gamma_superhuman_arr.append(gamma_superhuman)
      print("gamma_superhuman: ")
      print(sum(gamma_superhuman))
      if (sum(self.gamma_superhuman) == self.num_of_features and sum(gamma_superhuman) < self.num_of_features): # if last iter every feature was 1-superhuman and this iter changes --> break
        break
      if len(self.gamma_superhuman_arr) > 10 and sum(gamma_superhuman) < sum(self.gamma_superhuman_arr[-2]):  # look back if it has improved for the last 3 iterations, if not --> break
        gamma_degrade += 1
      else:
        gamma_degrade = 0
      if gamma_degrade == 3: # peformance degrades for 3 cosecutive iterations.
        break
      self.model_params = model_params

      
    #model_params = {"model":self.model_obj, "theta": self.model_obj.coef_, "alpha":self.alpha, "eval": self.eval, "subdom_value": subdom_tensor_sum_arr, "lr_theta": self.lr_theta, "num_of_demos":self.num_of_demos, "iters": iters, "num_of_features": self.num_of_features}
    experiment_filename = make_experiment_filename(dataset = self.dataset, demo_baseline = self.demo_baseline, lr_theta = self.lr_theta, num_of_demos = self.num_of_demos, noise_ratio = self.noise_ratio)
    file_dir = os.path.join(self.train_data_path)
    store_object(self.model_params, file_dir, experiment_filename, -1)


  def read_model_from_file(self):
    experiment_filename = make_experiment_filename(dataset = self.dataset, demo_baseline = self.demo_baseline, lr_theta = self.lr_theta, num_of_demos = self.num_of_demos, noise_ratio = self.noise_ratio)
    file_dir = os.path.join(self.train_data_path)
    self.model_params = load_object(file_dir,experiment_filename, -1)
    self.model_obj = self.model_params["model"]
    self.theta = self.model_params["theta"]
    self.train_eval = self.model_params["eval"]

    model_file_dir = os.path.join(self.model_path, 'base_model_' + dataset + '.pickle') 
    try:
      with open(model_file_dir, 'rb') as handle: #with open(f'base_model_{dataset}.pickle', 'rb') as handle:
        self.base_dict = pickle.load(handle)
        self.X_test = self.base_dict["X_test"]
        self.Y_test = self.base_dict["Y_test"]
        self.A_str_test = self.base_dict["A_str_test"]
        self.logi_params = self.base_dict["logi_params"]
    except Exception:
      self.base_model()


  def test_model(self, exp_idx):
    eval_sh = self.eval_model(mode = "test-sh")
    print()
    print(eval_sh)
    eval_pp_dp = self.eval_model_baseline(baseline = "pp", mode = "demographic_parity")
    print()
    print(eval_pp_dp)
    eval_pp_eqodds = self.eval_model_baseline(baseline = "pp", mode = "equalized_odds")
    print()
    print(eval_pp_eqodds)
    eval_fairll_dp = self.eval_model_baseline(baseline = "fair_logloss", mode = "demographic_parity")
    print()
    print(eval_fairll_dp)
    eval_fairll_eqodds = self.eval_model_baseline(baseline = "fair_logloss", mode = "equalized_odds")
    print()
    print(eval_fairll_eqodds)
    eval_fairll_eqopp = self.eval_model_baseline(baseline = "fair_logloss", mode = "equalized_opportunity")
    print()
    print(eval_fairll_eqopp)
    eval_MFOpt = self.eval_model_baseline(baseline = "MFOpt")
    print()
    print(eval_MFOpt)
    
    self.model_params["eval_sh"]= eval_sh
    self.model_params["eval_pp_dp"]= eval_pp_dp
    self.model_params["eval_pp_eq_odds"] = eval_pp_eqodds
    self.model_params["eval_fairll_dp"] = eval_fairll_dp
    self.model_params["eval_fairll_eqodds"] = eval_fairll_eqodds
    self.model_params["eval_fairll_eqopp"] = eval_fairll_eqopp
    self.model_params["eval_MFOpt"]= eval_MFOpt
    experiment_filename = make_experiment_filename(dataset = self.dataset, demo_baseline = self.demo_baseline, lr_theta = self.lr_theta, num_of_demos = self.num_of_demos, noise_ratio = self.noise_ratio)
    file_dir = os.path.join(self.test_data_path)
    store_object(self.model_params, file_dir, experiment_filename, exp_idx)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Description of your program')
  parser.add_argument('-t','--task', help='enter the task to do', required=True)
  parser.add_argument('-n','--noise', help='noisy demos used if True', default = default_args['noise'])
  parser.add_argument('-d', '--dataset', help="dataset name", default = default_args['dataset'])
  parser.add_argument('-b','--demo_baseline', help='model for creating demos', default = default_args['demo_baseline'])
  parser.add_argument('-f', '--features', help="features list", nargs='+', default = default_args['features'])
  parser.add_argument('-m', '--base_model_type', help="model type", default = default_args['base_model_type'])
  
  args = vars(parser.parse_args())

  print(args)
  
  dataset = args['dataset']
  feature_list = args['features']
  demo_baseline = args['demo_baseline']
  base_model_type = args['base_model_type']
  feature, num_of_features = create_features_dict(feature_list)
  print("features: ", feature_list)
  noise = eval(args['noise'])
  print("dataset: ", dataset)
  if noise==False:
    noise_ratio = 0.0
  
  if args['task'] == 'prepare-demos':
    sh_obj = Super_human(dataset = dataset, num_of_demos = num_of_demos, feature = feature, num_of_features = num_of_features, lr_theta = lr_theta, noise = noise, noise_ratio = noise_ratio, demo_baseline= demo_baseline, base_model_type = base_model_type)
    #sh_obj.base_model()
    sh_obj.prepare_test_pp(model = model, alpha = alpha, beta = beta) # this alpha is different from self.alpha

  elif args['task'] == 'train':
    print("lr_theta: ", lr_theta)
    sh_obj = Super_human(dataset = dataset, num_of_demos = num_of_demos, feature = feature, num_of_features = num_of_features, lr_theta = lr_theta, noise = noise, noise_ratio = noise_ratio, demo_baseline= demo_baseline, base_model_type = base_model_type)
    sh_obj.base_model()
    sh_obj.read_demo_list()
    sh_obj.update_model(lr_theta, iters)

  elif args['task'] == 'test':
    sh_obj = Super_human(dataset = dataset, num_of_demos = num_of_demos, feature = feature, num_of_features = num_of_features, lr_theta = lr_theta, noise = noise, noise_ratio = noise_ratio, demo_baseline= demo_baseline, base_model_type = base_model_type)
    #sh_obj.base_model()
    sh_obj.read_model_from_file()
    sh_obj.test_model(-1)
  
  elif args['task'] == 'noise-test':
    noise = True
    for noise_ratio in noise_list:
      sh_obj = Super_human(dataset = dataset, num_of_demos = num_of_demos, feature = feature, num_of_features = num_of_features, lr_theta = lr_theta, noise = noise, noise_ratio = noise_ratio, demo_baseline= demo_baseline, base_model_type = base_model_type)
      sh_obj.prepare_test_pp(model = model, alpha = alpha, beta = beta)
      sh_obj.base_model()
      sh_obj.read_demo_list()
      sh_obj.update_model(lr_theta, iters)
      sh_obj.read_model_from_file()
      sh_obj.test_model(-1)
  
  elif args['task'] == 'test-errorbars':
    for exp_idx in range(default_args['num_experiment']):
      print("exp_idx: ", exp_idx)
      sh_obj = Super_human(dataset = dataset, num_of_demos = num_of_demos, feature = feature, num_of_features = num_of_features, lr_theta = lr_theta, noise = noise, noise_ratio = noise_ratio, demo_baseline= demo_baseline, base_model_type = base_model_type)
      sh_obj.prepare_test_pp(model = model, alpha = alpha, beta = beta)
      sh_obj.base_model()
      sh_obj.read_demo_list()
      sh_obj.update_model(lr_theta, iters)
      sh_obj.read_model_from_file()
      sh_obj.test_model(exp_idx)