import os
import sys
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
warnings.filterwarnings('ignore')

# Data processing
from sklearn.model_selection import train_test_split

# Models
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

# Fairlearn algorithms and utils
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import GridSearch, EqualizedOdds

from sklearn.datasets import fetch_openml
from util import get_metrics_df
# Metrics
from fairlearn.metrics import (
    MetricFrame,
    selection_rate, demographic_parity_difference, demographic_parity_ratio,
    false_positive_rate, false_negative_rate,
    false_positive_rate_difference, false_negative_rate_difference,
    equalized_odds_difference)
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, zero_one_loss


sample_record_filename_template = "{}_{}_{}_{}"


def make_experiment_filename(**kwargs):
    return sample_record_filename_template.format(kwargs['dataset'], kwargs['lr_theta'], kwargs['lr_alpha'],  kwargs['num_of_demos']).replace('.','-')

def store_object(obj,path, name):
    filepath = os.path.join(path,name)
    with open(filepath, 'wb') as file:
        pickle.dump(obj,file)
    print("Record wrote to {}".format(filepath))

def load_object(path,name):
    with open(os.path.join(path,name), 'rb') as file:
        return pickle.load(file)


class Super_human:

  def __init__(self, dataset, num_of_demos, num_of_features, lr_alpha = 0.05, lr_theta = 0.05, noise=False, noise_ratio = 0.2):
    self.dataset = dataset
    self.num_of_demos = num_of_demos
    #self.num_of_samples = num_of_samples
    self.num_of_features = num_of_features
    self.feature = {0: "ZeroOne", 1: "Demographic parity difference", 2: "False negative rate difference", 3: "False positive rate difference", 4: "Equalized odds difference"}
    self.alpha = [1.0 for _ in range(self.num_of_features)]
    self.dataset_ref = pd.read_csv('dataset_ref.csv', index_col=0)
    self.num_of_attributs = self.dataset_ref.shape[1] - 1 # discard label
    self.lamda = 1.0
    self.c = None
    self.lr_theta = lr_theta
    self.lr_alpha = lr_alpha
    self.noise_ratio = noise_ratio
    self.noise = noise
    self.set_paths()
    """
    try:
      with open('base_model.pickle', 'rb') as handle:
          self.base_dict = pickle.load(handle)
          self.model_name = self.base_dict["model_name"]
          self.logi_params = self.base_dict["logi_params"]
          self.model_obj = self.base_dict["model_obj"]
          self.pred_scores = self.base_dict["pred_scores"]
    except Exception:
      self.base_model()
    """

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
    self.train_data_path = os.path.join(root, "train")
    self.test_data_path = os.path.join(root, "test")
    self.plots_path = os.path.join(root,"plots")

  def base_model(self):
    
    self.model_name = "logistic-regression"
    self.logi_params = {
        'C': 100,
        'penalty': 'l2',
        'solver': 'newton-cg',
        'max_iter': 1000
    }

    train_file_path = os.path.join(self.train_data_path, "train_data.csv")
    self.train_data = pd.read_csv(train_file_path, index_col=0)
    A = self.train_data["gender"]
    A_str = A.map({ 2:"Female", 1:"Male"})
    # Extract the target
    Y = self.train_data["label"]
    X = self.train_data.drop(columns=['label'])

    #model_logi = LogisticRegression(**self.logi_params)
    #model_logi.fit(X_train, Y_train)

    
    #self.train_data = pd.read_csv(train_file_path, index_col=0).drop(columns=['label'])
    # Extract the sensitive feature
    #dataset_ref = self.train_data.copy(deep=True).reset_index(drop=True)
    #dataset_ref = self.add_noise(dataset_ref)
    #A = dataset_ref["gender"]
    #A_str = A.map({ 2:"Female", 1:"Male"})
    # Extract the target
    #Y = dataset_ref["label"]
    
    
    X_train, X_test, Y_train, Y_test, A_train, A_test, A_str_train, A_str_test = train_test_split(
        X,
        Y,
        A,
        A_str,
        test_size = 1 - alpha,
        random_state=12345,
        stratify=Y
        )
    
    self.model_obj = LogisticRegression(**self.logi_params)
    self.model_obj.fit(X_train, Y_train)
    # Scores on test set
    self.pred_scores = self.model_obj.predict_proba(X_test) #self.pred_scores = self.model_obj.predict_proba(self.dataset_ref.drop(columns=['gender', 'label']))#[:, 1]
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

    with open('base_model.pickle', 'wb') as handle:
        pickle.dump(self.base_dict, handle)

  def run_logistic_pp(self, model = "logistic_regression", data_demo = data_demo): #df_train=None, Y_train=None, A_train=None, A_str_test=None, df_test=None,\
                    #Y_test=None, A_test=None):
                    #train_x, test_x, train_y, test_y, train_A, test_A, train_A_str, test_A_str
    df_train = data_demo.train_x
    Y_train = data_demo.train_y
    A_train = data_demo.train_A
    A_str_test = data_demo.test_A_str
    df_test = data_demo.test_x
    Y_test = data_demo.test_y
    A_test = data_demo.test_A
    Super_human.model_name = model

    if model == "logistic_regression":
      logi_param = {
          'C': 100,
          'penalty': 'l2',
          'solver': 'newton-cg'
      }
      model_logi = LogisticRegression(**logi_param)
      

    elif model == "lgbm_classifier":
      lgb_params = {
          'objective' : 'binary',
          'metric' : 'auc',
          'learning_rate': 0.03,
          'num_leaves' : 10,
          'max_depth' : 3
      }
      model_logi = lgb.LGBMClassifier(**lgb_params)
      
    model_logi.fit(df_train, Y_train)

    # Scores on test set
    test_scores_logi = model_logi.predict_proba(df_test)[:, 1]
    # Train AUC
    #roc_auc_score(Y_train, model_logi.predict_proba(df_train)[:, 1])
    # Predictions (0 or 1) on test set
    test_preds_logi = (test_scores_logi >= np.mean(Y_train)) * 1

    mf = MetricFrame({
        'FPR': false_positive_rate,
        'FNR': false_negative_rate},
        Y_test, test_preds_logi, sensitive_features=A_str_test)
    mf.by_group

    # Post-processing
    self.postprocess_est = ThresholdOptimizer(
        estimator=model_logi,
        constraints="demographic_parity", #"equalized_odds",
        predict_method='auto',
        prefit=True)
    # Balanced data set is obtained by sampling the same number of points from the majority class (Y=0)
    # as there are points in the minority class (Y=1)
    balanced_idx1 = df_train[Y_train==1].index
    pp_train_idx = balanced_idx1.union(Y_train[Y_train==0].sample(n=balanced_idx1.size, random_state=1234).index)
    df_train_balanced = df_train.loc[pp_train_idx, :]
    Y_train_balanced = Y_train.loc[pp_train_idx]
    A_train_balanced = A_train.loc[pp_train_idx]
    # Post-process fitting

    self.postprocess_est.fit(df_train_balanced, Y_train_balanced, sensitive_features=A_train_balanced)
    # Post-process preds
    postprocess_preds = self.postprocess_est.predict(df_test, sensitive_features=A_test)

    # Metrics
    models_dict = {
              "ThresholdOptimizer": (postprocess_preds, postprocess_preds)}
    return get_metrics_df(models_dict = models_dict, y_true = Y_test, group = A_str_test)

  def split_data(self, model, alpha=0.5, dataset=None,  mode="post-processing"):
    # Assign dataset to temporary variable
    dataset_temp = dataset.copy(deep=True)
    # Extract the sensitive feature
    #dataset_temp.reset_index(inplace=True, drop=True)
    A = dataset_temp["gender"]
    A_str = A.map({ 2:"Female", 1:"Male"})
    # Extract the target
    Y = dataset_temp["label"]
    
    ####################################
    if mode == "post-processing":
      dataset_temp.drop(columns=['gender'])
      if 'prev_index' in dataset_temp.columns:
        idx = dataset_temp["prev_index"]
        dataset_temp = dataset_temp.drop(columns=['prev_index'])
    elif mode == "normal":
      idx = dataset_temp.index.tolist()
    ###################################
    X = dataset_temp.drop(columns=['label'])
    #idx = dataset_temp["index"]
    
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
    self.dataset_ref = pd.read_csv('dataset_ref.csv', index_col=0)
    dataset_ref = self.dataset_ref.copy(deep=True)
    #self.test_pp_logi = pd.DataFrame(index = ['Demographic parity difference', 'False negative rate difference', 'ZeroOne'])
    self.test_pp_logi = pd.DataFrame(index = [self.feature[i] for i in range(self.num_of_features)])
    self.demo_list = []
    r = random.randint(0, 10000000)
    #index_list = dataset_ref.index.tolist()
    dataset_ref = shuffle(dataset_ref, random_state=r)
    ####################################
    A = dataset_ref["gender"]
    A_str = A.map({ 2:"Female", 1:"Male"})
    Y = dataset_ref["label"]
    idx = dataset_ref.index.tolist()
    X = dataset_ref.drop(columns=['label'])
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

    dataset_pp = dataset_ref.loc[idx_train].reset_index(drop=True) # use only pp portion of the data and leave SH Test portion
    dataset_sh = dataset_ref.loc[idx_test].reset_index(drop=True)
    train_file_path = os.path.join(self.train_data_path, "train_data.csv")
    test_file_path = os.path.join(self.test_data_path, "test_data.csv")
    dataset_pp.to_csv(train_file_path)
    dataset_sh.to_csv(test_file_path)
    dataset_pp_copy = dataset_pp.copy(deep=True)
    """
    sh_demo = self.split_data(model, alpha=alpha, dataset=dataset_ref, mode="normal") # To get the Test data for SuperHuman approach evaluation we split data once and store the test data for evaluation: (1-alpha)% for TEST SH
    dataset_pp = dataset_ref.loc[sh_demo.idx_train].reset_index(drop=True) # use only pp portion of the data and leave SH Test portion
    dataset_sh = dataset_ref.loc[sh_demo.idx_test].reset_index(drop=True)
    train_file_path = os.path.join(self.train_data_path, "train_data.csv")
    test_file_path = os.path.join(self.test_data_path, "test_data.csv")
    dataset_pp.to_csv(train_file_path)
    dataset_sh.to_csv(test_file_path)
    dataset_pp_copy = dataset_pp.copy(deep=True)
    """
    for i in range(self.num_of_demos):
      r = random.randint(0, 10000000)
      #dataset_temp = shuffle(dataset_pp, random_state = r)
      ##########################################################################

      index_list = dataset_pp_copy.index.tolist()
      dataset_temp = shuffle(dataset_pp_copy, random_state=r)
      if self.noise == True:
        dataset_temp = self.add_noise(dataset_temp)
      dataset_temp['prev_index'] = index_list
      ##########################################################################
      new_demo = self.split_data(model, alpha=beta, dataset=dataset_temp,  mode="post-processing")
      #if self.noise == True:
      #  new_demo.Y_test = self.add_noise(new_demo.Y_test)
      metrics = self.run_logistic_pp(model = model, data_demo = new_demo)
      self.test_pp_logi[i] = metrics
      new_demo.metric_df = metrics
      #print(new_demo.metric_df)
      #print()
      for k in range(self.num_of_features):
        new_demo.metric[k] = new_demo.metric_df.loc[self.feature[k]]["ThresholdOptimizer"]
      #new_demo.metric = {0: new_demo.metric_df.loc['ZeroOne']["ThresholdOptimizer"], 1: new_demo.metric_df.loc['Demographic parity difference']["ThresholdOptimizer"], 2: new_demo.metric_df.loc['False negative rate difference']["ThresholdOptimizer"]}
      self.demo_list.append(new_demo)
      print(i)

    file_dir = os.path.join(self.data_path)
    store_object(self.demo_list, file_dir, 'demo_list')
    #with open('demo_list.pickle', 'wb') as handle:
    #    pickle.dump(self.demo_list, handle)

  def add_noise(self, dataset):
    Y = dataset["label"].to_numpy()
    n = len(Y)
    noisy_Y = copy.deepcopy(Y)
    idx = np.random.permutation(range(len(Y)))[:int(self.noise_ratio*n)]
    noisy_Y[idx] = 1-Y[idx]
    dataset["label"] = noisy_Y
    return dataset
  
  
  def read_sample_matrix(self):
    self.sample_matrix = np.load('sample_matrix.npy')

  def read_demo_list(self):
    file_dir = os.path.join(self.data_path)
    print("file_dir in read_demo_list: ", file_dir)
    self.demo_list = load_object(file_dir, 'demo_list')
    #with open('demo_list.pickle', 'rb') as handle:
    #  self.demo_list = pickle.load(handle)
    return self.demo_list

  def get_model_pred(self, item): # an item is one row of the dataset
    score = self.model_obj.predict_proba(item).squeeze() # [p(y = 0), p(y = 1)]
    return score

  def sample_from_prob(self, dist, size):

    preds = [0.0, 1.0]
    sample_preds = np.random.choice(preds, size, True, dist)
    return sample_preds

  def sample_superhuman(self):
    start_time = time.time()
    train_file_path = os.path.join(self.train_data_path, "train_data.csv")
    self.train_data = pd.read_csv(train_file_path, index_col=0)
    X = self.train_data.drop(columns=['label']).to_numpy()

    #X = self.dataset_ref.drop(columns=['label']).to_numpy() #X = self.train_data.to_numpy()
    data_size, feature_size = self.train_data.shape
    self.sample_matrix = np.zeros((self.num_of_demos, data_size)) #np.array([[-1 for _ in range(data_size)] for _ in range(num_of_samples)]) # create a matrix of size [num_of_samples * data_set_size]. Each row is a sample from our model that predicts the label of dataset.
    #print("--- %s in sample_superhuman ---" % (time.time() - start_time))
    for j in range(data_size):
      probs = self.get_model_pred(item = [X[j]]) #probs = self.get_model_pred(item = [self.dataset_ref.drop(columns=['label']).loc[j]])
      self.sample_matrix[:,j] = self.sample_from_prob(dist = probs, size = self.num_of_demos) # return a vector of size num_of_samples (100) with label prediction samples for j_th item of the dataset
      
    
    print("sample_matrix size: ", self.sample_matrix.shape)
    print("self.sample_matrix[:,0]", self.sample_matrix[:,0])
    print("self.dataset_ref['label]", self.dataset_ref["label"].loc[0])
    print("--- %s end of sample_superhuman ---" % (time.time() - start_time))
    return self.sample_matrix

  def get_samples_demo_indexed(self):
    start_time = time.time()
    sample_size = self.demo_list[0].idx_test.size
    self.sample_matrix_demo_indexed = np.zeros((self.num_of_demos, sample_size))
    

    #train_file_path = os.path.join(self.train_data_path, "train_data.csv")
    #self.train_data = pd.read_csv(train_file_path, index_col=0).drop(columns=['label'])
    #X = self.train_data.drop(columns=['label'])
    for i in range(self.num_of_demos):
      demo = self.demo_list[i]
      #len(demo.idx_test.merge(B)) == len(A)
      #print("self.train_data['prev_index']: ", self.train_data['prev_index']).to_numpy())
      #print("demo.idx_test: ", demo.idx_test.to_numpy())############################################################################################################
      self.sample_matrix_demo_indexed[i,:] = self.sample_matrix[i,:][demo.idx_test]
    return self.sample_matrix_demo_indexed
    

  def get_sample_loss(self):
    start_time = time.time()
    #print("--- %s in get_sample_loss ---" % (time.time() - start_time))
    self.sample_loss = np.zeros((self.num_of_demos, self.num_of_features))
    for demo_index, x in enumerate(tqdm(self.demo_list)):
      demo = self.demo_list[demo_index]
      sample_preds = self.sample_matrix_demo_indexed[demo_index,:]
      #print("sample_preds", sample_preds.shape)
      #print("sample_preds: ", sample_preds)
      # Metrics
      models_dict = {"Super_human": (sample_preds, sample_preds)}
      y = self.train_data.loc[demo.idx_test]['label'] # we use true_y from original dataset since y_true in demo can be noisy (in noise setting)
      A = self.train_data.loc[demo.idx_test]["gender"]
      A_str = A.map({ 2:"Female", 1:"Male"})
      
      metric_df = get_metrics_df(models_dict = models_dict, y_true = y, group = A_str) #### takes so much time!!! #metric_df = get_metrics_df(models_dict = models_dict, y_true = demo.test_y, group = demo.test_A_str)
      for feature_index in range(self.num_of_features):
        self.sample_loss[demo_index, feature_index] = metric_df.loc[self.feature[feature_index]]["Super_human"] #metric[feature_index]
    
    print("sample_loss for 0-1: ", self.sample_loss[:,0])
    print("sample_loss for dp: ", self.sample_loss[:,1])
    print("sample_loss for EqOdds: ", self.sample_loss[:,4])
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
    train_file_path = os.path.join(self.train_data_path, "train_data.csv")
    self.train_data = pd.read_csv(train_file_path, index_col=0)
    X = self.train_data.drop(columns=['label']) #self.dataset_ref.drop(columns=['label'])
    self.exp_phi_X_Y = [0 for _ in range(self.num_of_attributs)]
    self.phi_X_Y = []
    #print("X: ", X.head())
    #print("exp_phi_X_Y: ", self.exp_phi_X_Y.shape)
    for i in range(self.num_of_demos):
      demo = self.demo_list[i]
      sample_Y = self.sample_matrix_demo_indexed[i,:]
      phi_X_Y_temp = np.reshape(sample_Y, (-1, 1)) * X.loc[demo.idx_test]
      phi_X_Y_temp = np.sum(phi_X_Y_temp, axis=0) / X.shape[0]
      #print("phi_X_Y_temp: ", np.shape(phi_X_Y_temp))
      #print("self.exp_phi_X_Y: ", np.shape(self.exp_phi_X_Y))
      self.phi_X_Y.append(phi_X_Y_temp)
      self.exp_phi_X_Y += phi_X_Y_temp
    self.exp_phi_X_Y /= self.num_of_demos  # get the average

  def feature_matching(self, demo_index):
    demo = self.demo_list[demo_index]
    X_demoIndexed = self.train_data.drop(columns=['label']).loc[demo.idx_test] #self.dataset_ref.drop(columns=['label']).loc[demo.idx_test]
    sample_Y_demoIndexed = self.sample_matrix_demo_indexed[demo_index,:]#[demo.idx_test]
    phi_X_Y = np.reshape(sample_Y_demoIndexed, (-1, 1)) * X_demoIndexed
    phi_X_Y = np.sum(phi_X_Y, axis=0) / X_demoIndexed.shape[0]

    return phi_X_Y - self.exp_phi_X_Y


  def compute_grad_theta(self):
    #print(self.sample_matrix_demo_indexed)
    start_time = time.time()
    #print("--- %s in compute_grad_theta ---" % (time.time() - start_time))
    self.subdom_tensor = np.zeros((self.num_of_demos, self.num_of_features)) 
    self.compute_exp_phi_X_Y() 
    grad_theta = [0.0 for _ in range(self.num_of_attributs)]
    #print("--- %s before for loop j ---" % (time.time() - start_time))
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

  def compute_alpha(self):
    start_time = time.time()
    #print("--- %s in compute_alpha ---" % (time.time() - start_time))
    alpha = np.ones(self.num_of_features)
  
    for k in range(self.num_of_features):
      dominated_demos = []
      alpha_candidate = []
      for j in range(self.num_of_demos):
        sample_loss = self.sample_loss[j, k]
        demo_loss = self.demo_list[j].metric[k] 
        #if True:#sample_loss <= demo_loss:
        #  alpha_candidate = 1.0/(demo_loss - sample_loss)
        #  if not math.isinf(alpha_candidate):
        #    dominated_demos.append((alpha_candidate, demo_loss, sample_loss))
        dominated_demos.append((demo_loss, sample_loss))
      
      dominated_demos.sort(key = lambda x: x[0]) #dominated_demos.sort(key = lambda x: x[0], reverse=True)   # sort based on demo loss
      dominated_demos = np.array(dominated_demos)
      avg_sample_loss = np.mean([x[1] for x in dominated_demos])
      # print()
      # print(self.feature[k])
      # print()
      # print("demo loss:")
      # print([x[0] for x in dominated_demos])
      # print("avg_sample_loss:")
      # print(avg_sample_loss)
      for m, demo in enumerate(dominated_demos):
        # print("in for: ")
        # print("m: ", m)
        # print("sample loss:", demo[1])
        if (demo[1]) <= np.mean([x[0] for x in dominated_demos[0:m+1]]): #if (demo[2]) <= np.mean([x[1] for x in dominated_demos[0:m+1]] and demo[0] > 0):
          # print("in if: ")
          # print("1.0/(demo_loss - sampe_loss)", 1.0/(demo[0] - demo[1]))
          alpha[k] = 1.0/(demo[0] - demo[1]) # 1/(y_tilde - y_hat)
          break
    for i in range(self.num_of_features):     
      if alpha[i] == 1:
        alpha[i] = max(alpha)

    print("--- %s end of compute_alpha ---" % (time.time() - start_time))
    return alpha



  def eval_model_pp(self, mode="demographic_parity"):
    train_file_path = os.path.join(self.train_data_path, "train_data.csv")
    test_file_path = os.path.join(self.test_data_path, "test_data.csv")
    self.train_data = pd.read_csv(train_file_path, index_col=0)#.drop(columns=['prev_index'])
    self.test_data = pd.read_csv(test_file_path, index_col=0)#.drop(columns=['prev_index'])
    A_train = self.train_data["gender"]
    A_test = self.test_data["gender"]
    A_str_train = A_train.map({ 2:"Female", 1:"Male"})
    A_str_test = A_test.map({ 2:"Female", 1:"Male"})
    # Extract the target
    Y_train = self.train_data["label"]
    Y_test = self.test_data["label"]
    X_train = self.train_data.drop(columns=['label'])
    X_test = self.test_data.drop(columns=['label'])

    model_logi = LogisticRegression(**self.logi_params)
    model_logi.fit(X_train, Y_train)

    # Post-processing
    postprocess_est = ThresholdOptimizer(
        estimator=model_logi,
        constraints=mode, #"equalized_odds",
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
    postprocess_preds = postprocess_est.predict(X_test, sensitive_features=A_test)

    # Metrics
    models_dict = {
              "Post_processing": (postprocess_preds, postprocess_preds)}
    return get_metrics_df(models_dict = models_dict, y_true = Y_test, group = A_str_test)



  def eval_model(self, mode):
    if mode == "train":
      train_file_path = os.path.join(self.train_data_path, "train_data.csv")
      self.train_data = pd.read_csv(train_file_path, index_col=0)#.drop(columns=['prev_index'])
      A = self.train_data["gender"]
      A_str = A.map({ 2:"Female", 1:"Male"})
      # Extract the target
      Y = self.train_data["label"]
      X = self.train_data.drop(columns=['label'])
      #X = self.base_dict["X_" + mode]
      #Y = self.base_dict["Y_" + mode]
      #A_str = self.base_dict["A_str_" + mode]
    elif mode == "test-sh" or mode == "test-pp":
      test_file_path = os.path.join(self.test_data_path, "test_data.csv")
      self.test_data = pd.read_csv(test_file_path, index_col=0)#.drop(columns=['prev_index'])
      A = self.test_data["gender"]
      A_str = A.map({ 2:"Female", 1:"Male"})
      # Extract the target
      Y = self.test_data["label"]
      X = self.test_data.drop(columns=['label'])

    # Scores on train set
    scores = self.model_obj.predict_proba(X)[:, 1]
    # Train AUC
    #roc_auc_score(Y_train, self.model_obj.predict_proba(X_train)[:, 1])
    # Predictions (0 or 1) on test set
    preds = (scores >= np.mean(Y)) * 1
    # Metrics
    eval = pd.DataFrame(index = [self.feature[i] for i in range(self.num_of_features)]) #['Demographic parity difference', 'False negative rate difference', 'ZeroOne']
    models_dict = {
              "Super_human": (preds, preds)}
    eval = get_metrics_df(models_dict = models_dict, y_true = Y, group = A_str)
    return eval

  def get_model_thetha(self):
    return self.model_obj.coef_[0]
  
  def get_model_alpha(self):
    return self.alpha

  def update_model_theta(self, new_theta):
    self.model_obj.coef_ = np.asarray([new_theta]) # update the coefficient of our logistic regression model with the new theta

  def update_model_alpha(self, new_alpha):
    self.alpha = new_alpha
  

  def update_model(self, lr_theta, lr_alpha, iters):
    self.lr_theta = lr_theta
    self.lr_alpha = lr_alpha
    self.grad_theta, subdom_tensor_sum_arr, eval = [], [], []
    for i in tqdm(range(iters)):
      # find sample loss and store it, we will use it for computing grad_theta and grad_alpha
      self.sample_superhuman() # update self.sample_matrix with new samples from new theta
      self.get_samples_demo_indexed()
      self.get_sample_loss()
      
      # get the current theta and alpha
      theta = self.get_model_thetha()
      alpha = self.get_model_alpha()
      # find new theta
      subdom_tensor_sum, grad_theta = self.compute_grad_theta() # computer gradient of loss w.r.t theta by sampling from our model
      new_theta = theta - self.lr_theta * grad_theta  # update theta using the gradinet values
      # find new alpha
      if i == 0:
        print("eval from first sample: ")
        print(self.eval_model(mode = "train"))
      new_alpha = self.compute_alpha()
      # grad_alpha = self.compute_grad_alpha()
      # new_alpha = alpha - self.lr_alpha * grad_alpha
      # print("grad_alpha: ", grad_alpha)
      print("new alpha: ", new_alpha)
      #update theta
      self.update_model_theta(new_theta)
      #update alpha
      self.update_model_alpha(new_alpha)
      # eval model
      eval_i = self.eval_model(mode = "train")
      print("eval_i:")
      print(eval_i)
      # store some stuff
      self.grad_theta.append(grad_theta)
      subdom_tensor_sum_arr.append(subdom_tensor_sum)
      eval.append(eval_i)
    model_params = {"model":self.model_obj, "theta": self.model_obj.coef_, "alpha":self.alpha, "eval": eval, "subdom_value": subdom_tensor_sum_arr, "lr_theta": self.lr_theta, "lr_alpha": self.lr_alpha, "num_of_demos":self.num_of_demos, "iters": iters, "num_of_features": self.num_of_features}
    experiment_filename = make_experiment_filename(dataset = self.dataset, lr_theta = self.lr_theta, lr_alpha = self.lr_alpha, num_of_demos = self.num_of_demos) # only lr_theta!!!
    file_dir = os.path.join(self.train_data_path)
    store_object(model_params, file_dir, experiment_filename)
    #np.save('eval_model.npy', eval)
    #np.save('subdom_value-'+self.learning_rate+'.npy', subdom_tensor_sum_arr)

  def read_model_from_file(self):
    experiment_filename = make_experiment_filename(dataset = self.dataset, lr_theta = self.lr_theta, lr_alpha = self.lr_alpha, num_of_demos = self.num_of_demos)
    file_dir = os.path.join(self.train_data_path)
    self.model_params = load_object(file_dir,experiment_filename)
    self.model_obj = self.model_params["model"]
    self.theta = self.model_params["theta"]
    self.train_eval = self.model_params["eval"]
    try:
      with open('base_model.pickle', 'rb') as handle:  # Be careful of which base_model you are reading!! We need to read base model corresponding to the training data
        self.base_dict = pickle.load(handle)
        self.X_test = self.base_dict["X_test"]
        self.Y_test = self.base_dict["Y_test"]
        self.A_str_test = self.base_dict["A_str_test"]
        self.logi_params = self.base_dict["logi_params"]
        #self.model_obj = self.base_dict["model_obj"]
        #self.pred_scores = self.base_dict["pred_scores"]
    except Exception:
      self.base_model()

    

    
    # try:
    #   with open('model-pp.pickle', 'wb') as handle:
    #     self.postprocess_est = pickle.load(handle)
    #     print("self.postprocess_est: ", self.postprocess_est)
    # except:
    #   print("No post-processing model saved!")
    


  def test_model(self):
    eval_sh = self.eval_model(mode = "test-sh")
    eval_pp_dp = self.eval_model_pp(mode = "demographic_parity")
    eval_pp_eq_odds = self.eval_model_pp(mode = "equalized_odds")
    print()
    print(eval_sh)
    print()
    print(eval_pp_dp)
    self.model_params["eval_sh"]= eval_sh
    self.model_params["eval_pp_dp"]= eval_pp_dp
    self.model_params["eval_pp_eq_odds"] = eval_pp_eq_odds
    experiment_filename = make_experiment_filename(dataset = self.dataset, lr_theta = self.lr_theta, lr_alpha = self.lr_alpha, num_of_demos = self.num_of_demos)
    file_dir = os.path.join(self.test_data_path)
    store_object(self.model_params, file_dir, experiment_filename)



lr_theta_list = [0.05] #[0.05, 0.1, 0.5, 1.0]
lr_theta = 0.01
lr_alpha = 0.05
iters = 10
dataset = "Adult"
num_of_demos = 100
num_of_features = 5
alpha = 0.5
beta = 0.5
model = "logistic_regression"
noise_ratio = 0.2
#noise = False

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Description of your program')
  parser.add_argument('-t','--task', help='enter the task to do', required=True)
  parser.add_argument('-n','--noise', help='noisy demos used if True', default=False)
  args = vars(parser.parse_args())

  noise = eval(args['noise'])
  print("noise: ", noise)
  print("noise type: ", type(noise))

  if args['task'] == 'prepare-demos':
    sh_obj = Super_human(dataset = dataset, num_of_demos = num_of_demos, num_of_features = num_of_features, noise = noise, noise_ratio = noise_ratio)
    #sh_obj.base_model()
    sh_obj.prepare_test_pp(model = model, alpha = alpha, beta = beta) # this alpha is different from self.alpha

  elif args['task'] == 'train':
    sh_obj = Super_human(dataset = dataset, num_of_demos = num_of_demos, num_of_features = num_of_features, noise = noise, noise_ratio = noise_ratio)
    sh_obj.base_model()
    sh_obj.read_demo_list()
    sh_obj.update_model(lr_theta, lr_alpha, iters)

  elif args['task'] == 'test':
    sh_obj = Super_human(dataset = dataset, num_of_demos = num_of_demos, num_of_features = num_of_features, lr_alpha = lr_alpha, lr_theta = lr_theta, noise = noise, noise_ratio = noise_ratio)
    #sh_obj.base_model()
    sh_obj.read_model_from_file()
    sh_obj.test_model()