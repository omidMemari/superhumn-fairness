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


root = "experiments"


#sample_path = os.path.join(root,"samples")
#results_path = os.path.join(root,"results")
data_path = os.path.join(root,"data")
train_data_path = os.path.join(root, "train")
test_data_path = os.path.join(root, "test")
plot_path = os.path.join(root,"results","plot")
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

  def __init__(self, dataset, num_of_demos, num_of_features, lr_alpha = 0.05, lr_theta = 0.05):
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
    try:
      with open('base_model.pickle', 'rb') as handle:
          self.base_dict = pickle.load(handle)
          self.model_name = self.base_dict["model_name"]
          self.logi_params = self.base_dict["logi_params"]
          self.model_obj = self.base_dict["model_obj"]
          self.pred_scores = self.base_dict["pred_scores"]
    except Exception:
      self.base_model()
    
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

  def base_model(self):
    
    self.model_name = "logistic-regression"
    self.logi_params = {
        'C': 100,
        'penalty': 'l2',
        'solver': 'newton-cg',
        'max_iter': 1000
    }

    # Extract the sensitive feature
    self.dataset_ref = self.dataset_ref.reset_index(drop=True)
    A = self.dataset_ref["gender"]
    A_str = A.map({ 2:"Female", 1:"Male"})
    # Extract the target
    Y = self.dataset_ref["label"]
    df_train, df_test, Y_train, Y_test, A_train, A_test, A_str_train, A_str_test = train_test_split(
        self.dataset_ref.drop(columns=['label']), ####self.dataset_ref.drop(columns=['gender', 'label']),
        Y,
        A,
        A_str,
        test_size = 1 - alpha,
        random_state=12345,
        stratify=Y
        )

    self.model_obj = LogisticRegression(**self.logi_params)
    self.model_obj.fit(df_train, Y_train)
    # Scores on test set
    self.pred_scores = self.model_obj.predict_proba(self.dataset_ref.drop(columns=['label'])) #self.pred_scores = self.model_obj.predict_proba(self.dataset_ref.drop(columns=['gender', 'label']))#[:, 1]
    self.base_dict = {"model_obj": self.model_obj,
                 "pred_scores": self.pred_scores,
                 "model_name": self.model_name,
                 "logi_params": self.logi_params,
                 "X_train": df_train,
                 "Y_train": Y_train,
                 "X_test": df_test,
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

  def split_data(self, model, alpha=0.5, dataset=None):
    # Assign dataset to temporary variable
    dataset_temp = dataset
    # Extract the sensitive feature
    #dataset_temp.reset_index(inplace=True, drop=True)
    A = dataset_temp["gender"]
    A_str = A.map({ 2:"Female", 1:"Male"})
    # Extract the target
    Y = dataset_temp["label"]
    #################################### CHANGED HERE
    idx = dataset_temp["prev_index"]
    ###################################
    #idx = dataset_temp["index"]
    
    # TRAIN TEST SPLIT
    # alpha is the percentage of (Train + Test of Post Processing)
    # beta is the percentage of train Post Processing and 1 - beta is percentage of Test Post Processing
    
    df_train, df_test, Y_train, Y_test, A_train, A_test, A_str_train, A_str_test, idx_train, idx_test = train_test_split(
        dataset_temp.drop(columns=['prev_index', 'gender', 'label']),
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
    dataset_ref = self.dataset_ref.copy(deep=True)
    #self.test_pp_logi = pd.DataFrame(index = ['Demographic parity difference', 'False negative rate difference', 'ZeroOne'])
    self.test_pp_logi = pd.DataFrame(index = [self.feature[i] for i in range(self.num_of_features)])
    self.demo_list = []
    r = random.randint(0, 10000000)
    #dataset_ref = shuffle(dataset_ref, random_state = r)
    #################################### CHANGED HERE
    index_list = dataset_ref.index.tolist()
    dataset_ref = shuffle(dataset_ref, random_state=r)
    dataset_ref['prev_index'] = index_list
    ####################################
    #dataset_ref = dataset_ref.reset_index() # leave index as a column to keep track of index change
    sh_demo = self.split_data(model, alpha=alpha, dataset=dataset_ref) # To get the Test data for SuperHuman approach evaluation we split data once and store the test data for evaluation: (1-alpha)% for TEST SH
    dataset_pp = dataset_ref.loc[sh_demo.idx_train] # use only pp portion of the data and leave SH Test portion
    dataset_sh = dataset_ref.loc[sh_demo.idx_test]
    train_file_path = os.path.join(train_data_path, "train_data.csv")
    test_file_path = os.path.join(test_data_path, "test_data.csv")
    dataset_pp.to_csv(train_file_path)
    dataset_sh.to_csv(test_file_path)
    for i in range(self.num_of_demos):
      r = random.randint(0, 10000000)
      #dataset_temp = shuffle(dataset_pp, random_state = r)
      ##########################################################################
      index_list = dataset_pp.index.tolist()
      dataset_temp = shuffle(dataset_pp, random_state=r)
      dataset_temp['prev_index'] = index_list
      ##########################################################################
      new_demo = self.split_data(model, alpha=beta, dataset=dataset_temp)
      #dataset_ref_sorted.loc[new_demo.idx_test] #works fine!
      #print(dataset_ref.loc[new_demo.test_x["index"]]) #works fine!
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
      
    with open('demo_list.pickle', 'wb') as handle:
        pickle.dump(self.demo_list, handle)
        
  
  def read_sample_matrix(self):
    self.sample_matrix = np.load('sample_matrix.npy')

  def read_demo_list(self):
    with open('demo_list.pickle', 'rb') as handle:
      self.demo_list = pickle.load(handle)
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
    X = self.dataset_ref.drop(columns=['label']).to_numpy()
    data_size, feature_size = self.dataset_ref.shape
    self.sample_matrix = np.zeros((self.num_of_demos, data_size)) #np.array([[-1 for _ in range(data_size)] for _ in range(num_of_samples)]) # create a matrix of size [num_of_samples * data_set_size]. Each row is a sample from our model that predicts the label of dataset.
    print("--- %s in sample_superhuman ---" % (time.time() - start_time))
    for j in range(data_size):
      probs = self.get_model_pred(item = [X[j]]) #probs = self.get_model_pred(item = [self.dataset_ref.drop(columns=['label']).loc[j]])
      self.sample_matrix[:,j] = self.sample_from_prob(dist = probs, size = self.num_of_demos) # return a vector of size num_of_samples (100) with label prediction samples for j_th item of the dataset
    
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
    print("--- %s in get_sample_loss ---" % (time.time() - start_time))
    self.sample_loss = np.zeros((self.num_of_demos, self.num_of_features))
    for demo_index, x in enumerate(tqdm(self.demo_list)):
      for feature_index in range(self.num_of_features):
        demo = self.demo_list[demo_index]
        sample_preds = self.sample_matrix_demo_indexed[demo_index,:]
        # Metrics
        models_dict = {"ThresholdOptimizer": (sample_preds, sample_preds)}
        metric_df = get_metrics_df(models_dict = models_dict, y_true = demo.test_y, group = demo.test_A_str) #### takes so much time!!!
        self.sample_loss[demo_index, feature_index] = metric_df.loc[self.feature[feature_index]]["ThresholdOptimizer"] #metric[feature_index]
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
      X = self.dataset_ref.drop(columns=['label'])
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
    X_demoIndexed = self.dataset_ref.drop(columns=['label']).loc[demo.idx_test]
    sample_Y_demoIndexed = self.sample_matrix_demo_indexed[demo_index,:]#[demo.idx_test]
    phi_X_Y = np.reshape(sample_Y_demoIndexed, (-1, 1)) * X_demoIndexed
    phi_X_Y = np.sum(phi_X_Y, axis=0) / X_demoIndexed.shape[0]

    return phi_X_Y - self.exp_phi_X_Y


  def compute_grad_theta(self):
    print(self.sample_matrix_demo_indexed)
    start_time = time.time()
    print("--- %s in compute_grad_theta ---" % (time.time() - start_time))
    self.subdom_tensor = np.zeros((self.num_of_demos, self.num_of_features)) 
    self.compute_exp_phi_X_Y() 
    grad_theta = [0.0 for _ in range(self.num_of_attributs)]
    print("--- %s before for loop j ---" % (time.time() - start_time))
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

  # def compute_grad_alpha(self):
  #   start_time = time.time()
  #   print("--- %s in compute_grad_alpha ---" % (time.time() - start_time))
  #   grad_alpha = np.zeros(self.num_of_features)
  #   for j, x in enumerate(self.demo_list):
  #     for k in range(self.num_of_features):
  #       sample_loss = self.sample_loss[j, k] #self.get_sample_loss(j, k)
  #       demo_loss = self.demo_list[j].metric[k]
  #       if self.alpha[k]*(sample_loss - demo_loss) + 1 > 0:     # subtract constant c to optimize for useful demonstation instead of avoiding from noisy ones
  #         grad_alpha[k] += sample_loss - demo_loss
  #   #denum = -2*self.lamda*self.num_of_demos
  #   grad_alpha /= -2*self.lamda*self.num_of_demos
  #   print("--- %s end of compute_grad_alpha ---" % (time.time() - start_time))
  #   return grad_alpha

  def compute_alpha(self):
    start_time = time.time()
    print("--- %s in compute_alpha ---" % (time.time() - start_time))
    alpha = np.zeros(self.num_of_features)
  
    for k in range(self.num_of_features):
      dominated_demos = []
      alpha_candidate = []
      for j in range(self.num_of_demos):
        sample_loss = self.sample_loss[j, k]
        demo_loss = self.demo_list[j].metric[k] 
        if sample_loss <= demo_loss:
          alpha_candidate = 1.0/(demo_loss - sample_loss)
          dominated_demos.append((alpha_candidate, demo_loss, sample_loss))
      
      avg_inverse_demo_loss = np.mean([1.0/x[1] for x in dominated_demos])
      dominated_demos.sort(key = lambda x: x[0])    # sort based on demo loss

      for dominated_demo in dominated_demos:
        if (1.0/dominated_demo[2]) >= avg_inverse_demo_loss:
          alpha[k] = dominated_demo[0]
          break

    print("--- %s end of compute_alpha ---" % (time.time() - start_time))
    return alpha



  def eval_model_pp(self):
    train_file_path = os.path.join(train_data_path, "train_data.csv")
    test_file_path = os.path.join(test_data_path, "test_data.csv")
    self.train_data = pd.read_csv(train_file_path, index_col=0).drop(columns=['prev_index'])
    self.test_data = pd.read_csv(test_file_path, index_col=0).drop(columns=['prev_index'])
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

    postprocess_est.fit(X_train_balanced, Y_train_balanced, sensitive_features=A_train_balanced)
    # Post-process preds
    postprocess_preds = postprocess_est.predict(X_test, sensitive_features=A_test)

    # Metrics
    models_dict = {
              "Post_processing": (postprocess_preds, postprocess_preds)}
    return get_metrics_df(models_dict = models_dict, y_true = Y_test, group = A_str_test)



  def eval_model(self, mode):
    if mode == "train":
      X = self.base_dict["X_" + mode]
      Y = self.base_dict["Y_" + mode]
      A_str = self.base_dict["A_str_" + mode]
    elif mode == "test-sh" or mode == "test-pp":
      test_file_path = os.path.join(test_data_path, "test_data.csv")
      self.test_data = pd.read_csv(test_file_path, index_col=0).drop(columns=['prev_index'])
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
    for i in range(iters):
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
      # store some stuff
      self.grad_theta.append(grad_theta)
      subdom_tensor_sum_arr.append(subdom_tensor_sum)
      eval.append(eval_i)
    model_params = {"model":self.model_obj, "theta": self.model_obj.coef_, "alpha":self.alpha, "eval": eval, "subdom_value": subdom_tensor_sum_arr, "lr_theta": self.lr_theta, "lr_alpha": self.lr_alpha, "num_of_demos":self.num_of_demos, "iters": iters, "num_of_features": self.num_of_features}
    experiment_filename = make_experiment_filename(dataset = self.dataset, lr_theta = self.lr_theta, lr_alpha = self.lr_alpha, num_of_demos = self.num_of_demos) # only lr_theta!!!
    file_dir = os.path.join(train_data_path)
    store_object(model_params, file_dir, experiment_filename)
    #np.save('eval_model.npy', eval)
    #np.save('subdom_value-'+self.learning_rate+'.npy', subdom_tensor_sum_arr)

  def read_model_from_file(self):
    experiment_filename = make_experiment_filename(dataset = self.dataset, lr_theta = self.lr_theta, lr_alpha = self.lr_alpha, num_of_demos = self.num_of_demos)
    file_dir = os.path.join(train_data_path)
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
    eval_pp = self.eval_model_pp()
    print()
    print(eval_sh)
    print()
    print(eval_pp)
    self.model_params["eval_sh"]= eval_sh
    self.model_params["eval_pp"]= eval_pp
    experiment_filename = make_experiment_filename(dataset = self.dataset, lr_theta = self.lr_theta, lr_alpha = self.lr_alpha, num_of_demos = self.num_of_demos)
    file_dir = os.path.join(test_data_path)
    store_object(self.model_params, file_dir, experiment_filename)



  




#lr_list = [-1.0, -0.5, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.5, 1.0]
lr_theta_list = [0.05] #[0.05, 0.1, 0.5, 1.0]
lr_theta = 0.05
lr_alpha = 0.05
iters = 10
dataset = "Adult"
num_of_demos = 100
num_of_features = 5
alpha = 0.5
beta = 0.5
model = "logistic_regression"
        

if __name__ == '__main__':
  if sys.argv[1] == 'train':
    sh_obj = Super_human(dataset = dataset, num_of_demos = num_of_demos, num_of_features = num_of_features)
    #sh_obj.base_model()
    sh_obj.read_demo_list()
    for lr_theta in lr_theta_list:
      sh_obj.update_model(lr_theta, lr_alpha, iters)
  
  elif sys.argv[1] == 'prepare-demos':
    sh_obj = Super_human(dataset = dataset, num_of_demos = num_of_demos, num_of_features = num_of_features)
    sh_obj.base_model()
    sh_obj.prepare_test_pp(model = model, alpha = alpha, beta = beta) # this alpha is different from self.alpha

  elif sys.argv[1] == "test":
    sh_obj = Super_human(dataset = dataset, num_of_demos = num_of_demos, num_of_features = num_of_features, lr_alpha = lr_alpha, lr_theta = lr_theta)
    #sh_obj.base_model()
    sh_obj.read_model_from_file()
    sh_obj.test_model()
    # write test results in a file 

    # Be careful! we have to make sure that we use alpha percent for both base model and demo list/
    # and test set in completely unseen!! ---> solved this problem by saving the test data and reading it back in test time.



