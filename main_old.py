# Commented out IPython magic to ensure Python compatibility.
import os
import sys
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

# Data processing
from sklearn.model_selection import train_test_split

# Models
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

# Fairlearn algorithms and utils
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import GridSearch, EqualizedOdds

# Metrics
from fairlearn.metrics import (
    MetricFrame,
    selection_rate, demographic_parity_difference, demographic_parity_ratio,
    false_positive_rate, false_negative_rate,
    false_positive_rate_difference, false_negative_rate_difference,
    equalized_odds_difference)
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, zero_one_loss
from sklearn.datasets import fetch_openml

# CURRENT_DIR = os.path.abspath(os.path.dirname(__name__))
# DATA_DIR = os.path.join(CURRENT_DIR, 'data')
# names = [
#     'age',
#     'workclass',
#     'fnlwgt',
#     'education',
#     'education-num',
#     'marital-status',
#     'occupation',
#     'relationship',
#     'race',
#     'gender',
#     'capital-gain',
#     'capital-loss',
#     'hours-per-week',
#     'native-country',
#     'label'
# ]

# relevant = [
#     'marital-status',
#     'occupation',
#     'relationship',
#     'race',
#     'gender',
#     'over_25',
#     'age',
#     'education-num',
#     'capital-gain',
#     'capital-loss',
#     'hours-per-week',
#     'label']

# positive_label=1
# negative_label=0

# def read_data(filename, specifier):
#     data = pd.read_csv(filename, names=names, \
#     sep=r'\s*,\s*',engine='python',na_values='?')
#     # 0 is train
#     if specifier == 0: 
#         data['label'] = \
#         data['label'].map({'<=50K': negative_label,'>50K': positive_label})
#     else:
#         data['label'] = \
#         data['label'].map({'<=50K.': negative_label,'>50K.': positive_label})
    
#     return data
# TRAIN_DATA_FILE = os.path.join(DATA_DIR, 'adult.data')
# TEST_DATA_FILE = os.path.join(DATA_DIR, 'adult.test')

# # Reading train and test data
# train_data = read_data(TRAIN_DATA_FILE, 0)
# train_data['over_25'] = np.where(train_data['age']>=25,'yes','no')
# test_data = read_data(TEST_DATA_FILE, 1)
# test_data.drop(0, inplace=True)
# test_data.reset_index(drop=True, inplace=True)
# test_data['age'] = test_data['age'].astype(int)
# test_data['over_25'] = np.where(test_data['age']>=25,'yes','no')

# data = pd.concat([test_data, train_data])
# data = data.fillna(value="NA")

# from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

# from sklearn.compose import ColumnTransformer, make_column_transformer
# data['gender'].replace({"Female":2, "Male":1}, inplace=True)
# preprocess = make_column_transformer(
#     (StandardScaler(), ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']),
#     (OneHotEncoder(sparse=False),['race', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'native-country', 'over_25']),
#     remainder = 'passthrough', # leave
#     verbose_feature_names_out=False
# )

# mat = pd.DataFrame(preprocess.fit_transform(data),  columns=preprocess.get_feature_names_out())


# data_testing = pd.get_dummies(mat)
# ## ADDED
# data_testing.reset_index().drop('index', axis=1, inplace=True)

# Helper functions
def get_metrics_df(models_dict, y_true, group):
    
    metrics_dict = {
        #"Overall selection rate": (
        #    lambda x: selection_rate(y_true, x), True),
        "Demographic parity difference": (
            lambda x: demographic_parity_difference(y_true, x, sensitive_features=group), True),
        #"Demographic parity ratio": (
        #    lambda x: demographic_parity_ratio(y_true, x, sensitive_features=group), True),
        #"Overall balanced error rate": (
        #    lambda x: 1-balanced_accuracy_score(y_true, x), True),
        #"Balanced error rate difference": (
        #    lambda x: MetricFrame(metrics=balanced_accuracy_score, y_true=y_true, y_pred=x, sensitive_features=group).difference(method='between_groups'), True),
        #"False positive rate difference": (
        #    lambda x: false_positive_rate_difference(y_true, x, sensitive_features=group), True),
        "False negative rate difference": (
            lambda x: false_negative_rate_difference(y_true, x, sensitive_features=group), True),
        #"Equalized odds difference": (
        #    lambda x: equalized_odds_difference(y_true, x, sensitive_features=group), True),
        "ZeroOne": (
            lambda x: zero_one_loss(y_true, x), True)
        #"Overall AUC": (
        #    lambda x: roc_auc_score(y_true, x), False),
        #"AUC difference": (
        #    lambda x: MetricFrame(metrics=roc_auc_score, y_true=y_true, y_pred=x, sensitive_features=group).difference(method='between_groups'), False),
    }
    df_dict = {}
    for metric_name, (metric_func, use_preds) in metrics_dict.items():
        df_dict[metric_name] = [metric_func(preds) if use_preds else metric_func(scores) 
                                for model_name, (preds, scores) in models_dict.items()]
    return pd.DataFrame.from_dict(df_dict, orient="index", columns=models_dict.keys())

from pandas.core.strings import base
import random
import pickle
import time
import os
from sklearn.utils import shuffle
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

  
class Super_human:

  def __init__(self, num_of_demos, num_of_samples, num_of_features):
    #self.model_name = None
    #self.model_obj = None
    #self.model_params = None
    #self.dataset_ref = None
    #self.pred_scores = None
    self.num_of_demos = num_of_demos
    self.num_of_samples = num_of_samples
    self.num_of_features = num_of_features
    self.feature = {0: "ZeroOne", 1: "Demographic parity difference", 2: "False negative rate difference", 3: "False positive rate difference", 4: "Equalized odds difference"}
    self.alpha = [1.0 for _ in range(self.num_of_features)]
    #self.dataset_ref = data_testing
    self.dataset_ref = pd.read_csv('dataset_ref.csv', index_col=0)
    self.num_of_attributs = self.dataset_ref.shape[1] - 1 # discard label
    self.c = None
    try:
      with open('base_model.pickle', 'rb') as handle:
          self.base_dict = pickle.load(handle)
          self.model_name = self.base_dict["model_name"]
          self.model_params = self.base_dict["model_params"]
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
    self.model_params = {
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
    alpha = 0.5
    df_train, df_test, Y_train, Y_test, A_train, A_test, A_str_train, A_str_test = train_test_split(
        self.dataset_ref.drop(columns=['label']), ####self.dataset_ref.drop(columns=['gender', 'label']),
        Y,
        A,
        A_str,
        test_size = 1 - alpha,
        random_state=12345,
        stratify=Y
        )

    self.model_obj = LogisticRegression(**self.model_params)
    self.model_obj.fit(df_train, Y_train)
    # Scores on test set
    self.pred_scores = self.model_obj.predict_proba(self.dataset_ref.drop(columns=['label'])) #self.pred_scores = self.model_obj.predict_proba(self.dataset_ref.drop(columns=['gender', 'label']))#[:, 1]
    self.base_dict = {"model_obj": self.model_obj,
                 "pred_scores": self.pred_scores,
                 "model_name": self.model_name,
                 "model_params": self.model_params,
                 "X_train": df_train,
                 "Y_train": Y_train,
                 "X_test": df_test,
                 "Y_test": Y_test,
                 "A_str_train": A_str_train,
                 "A_str_test": A_str_test}

    with open('base_model.pickle', 'wb') as handle:
        pickle.dump(self.base_dict, handle)



  def get_model_pred(self, item): # an item is one row of the dataset
    #print("item: ", item)
    score = self.model_obj.predict_proba(item).squeeze() # [p(y = 0), p(y = 1)]
    #print("score", score)
    return score

  def sample_from_prob(self, dist, size):

    preds = [0.0, 1.0]
    sample_preds = np.random.choice(preds, size, True, dist)
    #print("sample_preds", sample_preds)

    return sample_preds


  def sample_superhuman(self):

    data_size, feature_size = self.dataset_ref.shape
    self.sample_matrix = np.zeros((self.num_of_samples, data_size)) #np.array([[-1 for _ in range(data_size)] for _ in range(num_of_samples)]) # create a matrix of size [num_of_samples * data_set_size]. Each row is a sample from our model that predicts the label of dataset.

    for j in range(data_size):
      probs = self.get_model_pred(item = [self.dataset_ref.drop(columns=['label']).loc[j]])
      self.sample_matrix[:,j] = self.sample_from_prob(dist = probs, size = self.num_of_samples) # return a vector of size num_of_samples (100) with label prediction samples for j_th item of the dataset
    
    # write sample_matrix in a file
    np.save('sample_matrix.npy', self.sample_matrix)
    
    return self.sample_matrix

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
      #Super_human.model_params = logi_params
      

    elif model == "lgbm_classifier":
      lgb_params = {
          'objective' : 'binary',
          'metric' : 'auc',
          'learning_rate': 0.03,
          'num_leaves' : 10,
          'max_depth' : 3
      }
      model_logi = lgb.LGBMClassifier(**lgb_params)
      #Super_human.model_params = lgb_params
      
    model_logi.fit(df_train, Y_train)

    #Super_human.model_obj = model_logi

    # Scores on test set
    test_scores_logi = model_logi.predict_proba(df_test)[:, 1]
    # Train AUC
    roc_auc_score(Y_train, model_logi.predict_proba(df_train)[:, 1])
    # Predictions (0 or 1) on test set
    test_preds_logi = (test_scores_logi >= np.mean(Y_train)) * 1

    mf = MetricFrame({
        'FPR': false_positive_rate,
        'FNR': false_negative_rate},
        Y_test, test_preds_logi, sensitive_features=A_str_test)
    mf.by_group

    # Post-processing
    postprocess_est = ThresholdOptimizer(
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

    postprocess_est.fit(df_train_balanced, Y_train_balanced, sensitive_features=A_train_balanced)
    # Post-process preds
    postprocess_preds = postprocess_est.predict(df_test, sensitive_features=A_test)

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
    self.test_pp_logi = pd.DataFrame(index = ['Demographic parity difference', 'False negative rate difference', 'ZeroOne'])
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
    for i in range(0, self.num_of_demos):
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
        for i in range(self.num_of_features):
          new_demo.metric[i] = new_demo.metric_df.loc[self.feature[i]]["ThresholdOptimizer"]
        #new_demo.metric = {0: new_demo.metric_df.loc['ZeroOne']["ThresholdOptimizer"], 1: new_demo.metric_df.loc['Demographic parity difference']["ThresholdOptimizer"], 2: new_demo.metric_df.loc['False negative rate difference']["ThresholdOptimizer"]}
        self.demo_list.append(new_demo)
    with open('demo_list.pickle', 'wb') as handle:
        pickle.dump(self.demo_list, handle)


  
  def read_sample_matrix(self):
    self.sample_matrix = np.load('sample_matrix.npy')
    #return self.sample_matrix

  def read_demo_list(self):
    with open('demo_list.pickle', 'rb') as handle:
      self.demo_list = pickle.load(handle)

    #return self.demo_list

  def get_sample_loss(self, sample_index, demo_index, feature_index):
    demo = self.demo_list[demo_index]
    sample_preds = self.sample_matrix[sample_index,:]
    sample_preds_demoIndexed = sample_preds[demo.idx_test] ## need to be tested
    # Metrics
    models_dict = {"ThresholdOptimizer": (sample_preds_demoIndexed, sample_preds_demoIndexed)}
    metric_df = get_metrics_df(models_dict = models_dict, y_true = demo.test_y, group = demo.test_A_str) #### takes so much time!!!
    #metric = [metric_df.loc['ZeroOne']["ThresholdOptimizer"], metric_df.loc['Demographic parity difference']["ThresholdOptimizer"], metric_df.loc['False negative rate difference']["ThresholdOptimizer"]]

    return metric_df.loc[self.feature[feature_index]]["ThresholdOptimizer"] #metric[feature_index]

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
      
      for i in range(self.num_of_samples):
        sample_Y = self.sample_matrix[i,:]
        phi_X_Y_temp = np.reshape(sample_Y, (-1, 1)) * X
        phi_X_Y_temp = np.sum(phi_X_Y_temp, axis=0) / X.shape[0]

        self.phi_X_Y.append(phi_X_Y_temp)
        self.exp_phi_X_Y += phi_X_Y_temp
      self.exp_phi_X_Y /= self.num_of_samples  # get the average

  def feature_matching(self, sample_index, demo_index):
    demo = self.demo_list[demo_index]
    X_demoIndexed = self.dataset_ref.drop(columns=['label']).loc[demo.idx_test]
    sample_Y_demoIndexed = self.sample_matrix[sample_index,:][demo.idx_test]
    phi_X_Y = np.reshape(sample_Y_demoIndexed, (-1, 1)) * X_demoIndexed
    phi_X_Y = np.sum(phi_X_Y, axis=0) / X_demoIndexed.shape[0]

    return phi_X_Y - self.exp_phi_X_Y


  def compute_grad_theta(self):
    #self.read_sample_matrix()
    self.sample_superhuman() # update self.sample_matrix with new samples from new theta
    self.read_demo_list()
    self.subdom_tensor = np.zeros((self.num_of_samples, self.num_of_demos, self.num_of_features)) 
    start_time = time.time()
    #theta = np.hstack((self.model_obj.intercept_[:,None], self.model_obj.coef_))[0]
    self.compute_exp_phi_X_Y() 
    #self.phi = []
    grad_theta = [0.0 for _ in range(self.num_of_attributs)]
    self.subdom_constant = self.get_subdom_constant()
    for i, x in enumerate(tqdm(self.sample_matrix)):
    #for i in range(self.num_of_samples):
      #self.phi.append(self.feature_matching(i))
      if i == 0: self.subdom_constant = 0
      else: self.subdom_constant = self.get_subdom_constant()
      #print(i)
      #print("--- %s after for loop i ---" % (time.time() - start_time))
      for j in range(self.num_of_demos):
        for k in range(self.num_of_features):
          sample_loss = self.get_sample_loss(i, j, k)
          #print("sample_loss: ",  sample_loss)
          demo_loss = self.demo_list[j].metric[k]
          #print("demo_loss: ", demo_loss)
          #print("self.alpha[k]: ", self.alpha[k])
          #print("self.subdom_constant: ", self.subdom_constant)
          #print("self.phi[i]: ", self.phi[i])
          self.subdom_tensor[i, j, k] = max(self.alpha[k]*(sample_loss - demo_loss) + 1, 0) - self.subdom_constant     # subtract constant c to optimize for useful demonstation instead of avoiding from noisy ones
          #print("subdom_tensor[i, j, k]: ", self.subdom_tensor[i, j, k])
          grad_theta += self.subdom_tensor[i, j, k] * self.feature_matching(i, j) #self.phi[i]
          #print("grad_theta in subdom fucn: ", grad_theta)

    # write sample_matrix in a file
    np.save('subdom_tensor.npy', self.subdom_tensor)
    return self.subdom_tensor, grad_theta

  def eval_model(self):
    X_train = self.base_dict["X_train"]
    Y_train = self.base_dict["Y_train"]
    A_str_train = self.base_dict["A_str_train"]
    
    # Scores on train set
    train_scores = self.model_obj.predict_proba(X_train)[:, 1]
    # Train AUC
    roc_auc_score(Y_train, self.model_obj.predict_proba(X_train)[:, 1])
    # Predictions (0 or 1) on test set
    train_preds = (train_scores >= np.mean(Y_train)) * 1
    # Metrics
    train_eval = pd.DataFrame(index = [self.feature[i] for i in range(self.num_of_features)]) #['Demographic parity difference', 'False negative rate difference', 'ZeroOne']
    models_dict = {
              "sh_model": (train_preds, train_preds)}
    train_eval = get_metrics_df(models_dict = models_dict, y_true = Y_train, group = A_str_train)
    #display(train_eval)
    return train_eval




  def update_theta(self):
    self.learning_rate = 0.1
    #print("old coef_: ",   self.model_obj.coef_)
    theta = self.model_obj.coef_[0]
    iters = 3
    self.grad_theta, subdom_tensor_arr = [], []
    eval = []
    for i in range(iters):
      subdom_tensor, grad_theta = self.compute_grad_theta() # computer gradient of loss w.r.t theta by sampling from our model
      new_theta = theta - self.learning_rate * grad_theta  # update theta using the gradinet values
      self.grad_theta.append(grad_theta)
      subdom_tensor_arr.append(subdom_tensor)
      self.model_obj.coef_ = np.asarray([new_theta]) # update the coefficient of our logistic regression model with the new theta
      #print("new coef_ : ",  self.model_obj.coef_)
      #print("subdom_tensor")
      #display(subdom_tensor)
      print("grad_theta")
      print(grad_theta)
      print("theta")
      print(theta)
      print("new_theta")
      print(new_theta)
      if(np.isnan(new_theta).any()):
        print("The Array contain NaN values")
      else:
        print("The Array does not contain NaN values")
      eval_i = self.eval_model()
      eval.append(eval_i)
    np.save('eval_model.npy', eval)
    np.save('subdom_tensor.npy', subdom_tensor_arr)

sh_obj = Super_human(num_of_demos = 100, num_of_samples = 100, num_of_features = 3)
#print(sh_obj.dataset_ref)
sh_obj.base_model()

#sample_matrix = sh_obj.sample_superhuman()

sh_obj.prepare_test_pp(model = "logistic_regression", alpha = 0.5, beta = 0.5)

#eval_model = np.load('eval_model.npy')
#eval_model



eval = sh_obj.update_theta()

print(eval)