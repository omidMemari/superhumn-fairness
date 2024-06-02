import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from fairlearn.postprocessing import ThresholdOptimizer
from fair_logloss.fair_logloss import DP_fair_logloss_classifier, EOPP_fair_logloss_classifier, EODD_fair_logloss_classifier
from data_demo import data_demo
from util import make_demo_list_filename, load_object, make_experiment_filename, store_object, get_metrics_df
import pandas as pd
import random
import numpy as np
import copy
from sklearn.utils import shuffle

def split_data(alpha=0.5, dataset=None, mode="post-processing", sensitive_feature = None, label = None, dict_map = None):
    # Assign dataset to temporary variable
    dataset_temp = dataset.copy(deep=True)
    # Extract the sensitive feature
    A = dataset_temp[sensitive_feature]
    A_str = A.map(dict_map)
    # Extract the target
    Y = dataset_temp[label]

    ####################################
    if mode == "post-processing":
        dataset_temp.drop(columns=[sensitive_feature])
        if 'prev_index' in dataset_temp.columns:
            idx = dataset_temp["prev_index"]
            dataset_temp = dataset_temp.drop(columns=['prev_index'])
    elif mode == "normal":
        idx = dataset_temp.index.tolist()
    ###################################
    X = dataset_temp.drop(columns=[label])
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

    new_demo = data_demo(df_train, df_test, Y_train, Y_test, A_train, A_test, A_str_train, A_str_test, idx_train, idx_test)
    return new_demo

def read_demo_list(data_path, dataset, demo_baseline, num_of_demos, noise_ratio):
    file_dir = os.path.join(data_path)
    print("file_dir in read_demo_list: ", file_dir, flush=True)
    
    demo_list_filename = make_demo_list_filename(dataset=dataset, demo_baseline=demo_baseline, num_of_demos=num_of_demos, noise_ratio=noise_ratio)
    print("demo_list file name: ", flush=True)
    print(demo_list_filename)
    
    demo_list = load_object(file_dir, demo_list_filename, -1)
    
    # print demo_list attributes
    print("demo_list attributes: ", flush=True)
    print(demo_list[0].__dict__.keys())
    
    return demo_list

def save_AXY(dataset, A_train, X_train, Y_train, A_test, X_test, Y_test):

    A_train.to_csv(os.path.join("dataset", dataset, 'A_train.csv'),  sep='\t')
    X_train.to_csv(os.path.join("dataset", dataset, 'X_train.csv'),  sep='\t')
    Y_train.to_csv(os.path.join("dataset", dataset, 'Y_train.csv'),  sep='\t')

    A_test.to_csv(os.path.join("dataset", dataset, 'A_test.csv'),  sep='\t')
    X_test.to_csv(os.path.join("dataset", dataset, 'X_test.csv'),  sep='\t')
    Y_test.to_csv(os.path.join("dataset", dataset, 'Y_test.csv'),  sep='\t')    
    
def add_noise(data, dataset, sensitive_feature, label, noise_ratio, protected=False):
  if protected:
    name = sensitive_feature
    Y = data.to_frame(name=name).reset_index(drop=True)
  else: 
    name = label
    Y = pd.DataFrame(data, columns=[label]).reset_index(drop=True)
  
  n = len(Y)
  #print(n)
  noisy_Y = copy.deepcopy(Y)
  idx = np.random.permutation(range(n))[:int(noise_ratio*n)]
  #print(idx)
  if protected==True and (dataset == 'Adult' or dataset == 'Diabetes'): # checked! works well!
    Y['gender'].loc[idx] = Y['gender'].loc[idx] - 1   # change 1,2 to 0,1
    noisy_Y['gender'].loc[idx] = 1-Y['gender'].loc[idx] # change 0,1 to 1,0
    noisy_Y['gender'].loc[idx] = noisy_Y['gender'].loc[idx] + 1   # revert 1,0 to 2,1
  elif protected==True and dataset == 'COMPAS':
    noisy_Y['race'].loc[idx] = 1-Y['race'].loc[idx]
  elif protected==False:
    noisy_Y[label].loc[idx] = 1 - Y[label].loc[idx] # if adding noise to the label
  
  return noisy_Y
    
def run_demo_baseline(data_demo, demo_baseline, noise, feature, dataset, sensitive_feature, label, noise_ratio):
    X_train = data_demo.train_x
    Y_train = data_demo.train_y
    A_train = data_demo.train_A
    A_str_test = data_demo.test_A_str
    X_test = data_demo.test_x
    Y_test = data_demo.test_y
    A_test = data_demo.test_A
    
    
    logi_params = {'C': 100,
        'penalty': 'l2',
        'solver': 'newton-cg',
        'max_iter': 1000}
    #Super_human.model_name = model

    if demo_baseline == "pp":
      model_logi = LogisticRegression(**logi_params)
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
      baseline_preds = postprocess_est.predict(X_test, sensitive_features=A_test)

    elif demo_baseline == "fair_logloss":
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
    if noise == True:
      # add_noise(self, data, dataset, sensitive_feature, label, noise_ratio, protected=False)
        baseline_preds = add_noise(data = baseline_preds, dataset = dataset, sensitive_feature = sensitive_feature, 
                                   label = label, noise_ratio = noise_ratio, protected=False)
    # Metrics
    models_dict = {
              demo_baseline : (baseline_preds, baseline_preds)} 
    ##############################################################    
    result = get_metrics_df(models_dict = models_dict, y_true = Y_test, group = A_str_test, feature = feature, is_demo = True)

    return result
    
def add_noise_new(data_demo, dataset, noise_ratio, dict_map):  # works fine!

  Y_train = data_demo.train_y
  A_str_test = data_demo.test_A_str
  A_test = data_demo.test_A

  n_Y = len(Y_train)
  n_A = len(A_test)

  idx_Y = np.random.permutation(range(n_Y))[:int(noise_ratio*n_Y)]
  idx_A = np.random.permutation(range(n_A))[:int(noise_ratio*n_A)]

  A_test_index = A_test.index
  Y_train_index = Y_train.index

  A_test = A_test.reset_index(drop=True)
  Y_train = Y_train.reset_index(drop=True)

  noisy_A_test = copy.deepcopy(A_test)
  noisy_Y_train = copy.deepcopy(Y_train)
  # flip protected attribute
  if dataset == 'Adult':
    A_test.loc[idx_A] = A_test.loc[idx_A] - 1   # change 1,2 to 0,1
    noisy_A_test.loc[idx_A] = 1 - A_test.loc[idx_A] # change 0,1 to 1,0
    noisy_A_test.loc[idx_A] = noisy_A_test.loc[idx_A] + 1   # revert 1,0 to 2,1
  elif dataset == 'COMPAS':
    noisy_A_test.loc[idx_A] = 1 - A_test.loc[idx_A] # change 0,1 to 1,0
  # flip label
  noisy_Y_train.loc[idx_Y] = 1 - Y_train.loc[idx_Y] # if adding noise to the label

  noisy_A_test.index = A_test_index
  noisy_Y_train.index = Y_train_index
  noisy_A_test_str = noisy_A_test.map(dict_map)
  noisy_A_test_str.index = A_test_index
  
  data_demo.test_A = noisy_A_test
  data_demo.test_A_str = noisy_A_test_str
  data_demo.train_y = noisy_Y_train

  return data_demo    

def prepare_test_pp(dataset, dataset_path, sensitive_feature, label, feature, num_of_features, dict_map, demo_baseline, lr_theta, num_of_demos, noise_ratio, train_data_path, test_data_path, data_path, noise=False, model="logistic_regression", alpha=0.5, beta=0.5):
    
    dataset_ref = pd.read_csv(dataset_path, index_col=0) 
    dataset_ref = dataset_ref.copy(deep=True)
    # convert dataset_ref[senitive_feature] to int
    dataset_ref[sensitive_feature] = dataset_ref[sensitive_feature].astype(int)
    
    test_pp_logi = pd.DataFrame(index = [feature[i] for i in range(num_of_features)])
    demo_list = []
    r = random.randint(0, 10000000)
    dataset_ref = shuffle(dataset_ref, random_state=r)
    ####################################
    A = dataset_ref[sensitive_feature]
    A_str = A.map(dict_map)
    Y = dataset_ref[label]
    idx = dataset_ref.index.tolist()
    X = dataset_ref.drop(columns=[label])
    # print(X)

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

    # dataset name, baseline, lr_theta, num_of_demos, noise_ratio
    train_data_filename = "train_data_" + make_experiment_filename(dataset = dataset, demo_baseline = demo_baseline,\
      lr_theta = lr_theta, num_of_demos = num_of_demos, noise_ratio = noise_ratio) + ".csv"
    train_file_path = os.path.join(train_data_path, train_data_filename)
    print(train_file_path)
    print(train_data_filename)
    test_data_filename = "test_data_" + make_experiment_filename(dataset = dataset, demo_baseline = demo_baseline, 
                                                                 lr_theta = lr_theta, num_of_demos = num_of_demos, noise_ratio = noise_ratio) + ".csv"
    test_file_path = os.path.join(test_data_path, test_data_filename)
    
    if not os.path.exists(train_file_path):
        os.makedirs(train_data_path)
    if not os.path.exists(test_file_path):
        os.makedirs(test_data_path)    

    dataset_pp.to_csv(train_file_path)
    dataset_sh.to_csv(test_file_path)
    
    dataset_pp_copy = dataset_pp.copy(deep=True)
    
    
    for i in range(num_of_demos):
      r = random.randint(0, 10000000)
      ##########################################################################
      index_list = dataset_pp_copy.index.tolist()
      dataset_temp = shuffle(dataset_pp_copy, random_state=r)
      dataset_temp['prev_index'] = index_list
      ##########################################################################
      new_demo = split_data(alpha=beta, dataset=dataset_temp,  mode="post-processing", sensitive_feature = sensitive_feature, label = label, dict_map = dict_map)
      print("before:", new_demo)
      if noise == True:
        # add_noise_new(self, data_demo, dataset, noise_ratio, dict_map):
        new_demo = add_noise_new(data_demo = new_demo, dataset = dataset, noise_ratio = noise_ratio, dict_map = dict_map)
      print("running metrics")
      #run_demo_baseline(self, data_demo, demo_baseline, noise, feature, dataset, sensitive_feature, label, noise_ratio):
      metrics = run_demo_baseline(data_demo = new_demo, demo_baseline = demo_baseline, noise = noise,
                                       feature = feature, dataset = dataset, sensitive_feature = sensitive_feature,
                                       label = label, noise_ratio = noise_ratio)
      #if self.noise == True and metrics[self.demo_baseline]["Demographic parity difference"]
      
      print("demo metrics: ")
      print(metrics)
      print("-----------------------------------")
      test_pp_logi[i] = metrics
      new_demo.metric_df = metrics
      for k in range(num_of_features):
        new_demo.metric[k] = new_demo.metric_df.loc[feature[k]][demo_baseline]
      print(new_demo)
      demo_list.append(new_demo)
      print("demo {}".format(i))

    file_dir = os.path.join(data_path)
    demo_list_filename = make_demo_list_filename(dataset = dataset, demo_baseline = demo_baseline, num_of_demos = num_of_demos, noise_ratio = noise_ratio)
    print(file_dir)
    print(demo_list_filename)
    store_object(demo_list, file_dir, demo_list_filename, -1)