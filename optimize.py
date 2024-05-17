import os
import time
import pickle
import copy

# Data Manipulation and Visualization
import numpy as np
import pandas as pd

# Machine Learning and Model Evaluation
import torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Fairness Algorithms and Utilities
from fairlearn.postprocessing import ThresholdOptimizer
from fair_logloss.fair_logloss import DP_fair_logloss_classifier, EOPP_fair_logloss_classifier, EODD_fair_logloss_classifier
from model import LR_base_superhuman_model, LogisticRegression_pytorch

# Data Utilities and Custom Modules
from data_util import read_demo_list, save_AXY, add_noise_new, run_demo_baseline
from util import compute_error, get_metrics_df, find_gamma_superhuman, load_object, store_object, make_experiment_filename
from data_demo import data_demo

# Progress Bar
from tqdm import tqdm

default_args = {'dataset': 'Adult', 'iters': 50, 'num_of_demos': 50, 'num_of_features': 4, 'lr_theta': 0.01, 'noise': 'False', 'noise_ratio': 0.2, 'demo_baseline': 'pp', 'features': ['inacc', 'dp', 'eqodds', 'prp'], 'base_model_type': 'LR', 'num_experiment': 10}
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

model = "logistic_regression"
noise_list = [0.2]


class Super_human:

  def __init__(self, dataset, num_of_demos, feature, num_of_features, lr_theta, noise, noise_ratio, demo_baseline, base_model_type, model_obj):
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
    self.dataset_ref = pd.read_csv(self.dataset_path, index_col=0)
    self.num_of_attributs = self.dataset_ref.shape[1] - 1 # discard self.label
    self.model_params = None
    self.logi_params = {
        'C': 100,
        'penalty': 'l2',
        'solver': 'newton-cg',
        'max_iter': 1000
    }
    self.predss = []
    self.model_obj = model_obj

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
  
  def subdom_loss(self, demo_list, num_of_demos, num_of_features, subdom_constant, alpha, sample_loss):
      subdom_tensor = torch.empty([num_of_demos, num_of_features]).cuda()
      sample_loss_arr = torch.tensor(sample_loss, requires_grad=True).cuda()
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
  
  def base_model(self):
    self.model_name = "LR_pytorch"
    train_data_filename = "train_data_" + make_experiment_filename(dataset = self.dataset, demo_baseline = self.demo_baseline, lr_theta = self.lr_theta, num_of_demos = self.num_of_demos, noise_ratio = self.noise_ratio) + ".csv"
    train_file_path = os.path.join(self.train_data_path, train_data_filename)
    self.train_data = pd.read_csv(train_file_path, index_col=0)

    A = self.train_data[self.sensitive_feature]
    A_str = A.map(self.dict_map)
    
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

    if self.model_obj == 'omid':
      self.model_obj = LR_base_superhuman_model()
    elif self.model_obj == 'nn':
      self.model_obj = LogisticRegression_pytorch(X_train.shape[1], pd.unique(Y).size).cuda()
      
      
    print("self.model_obj.type: ", self.model_obj.type)
    # X_train = torch.from_numpy(X_train.to_numpy(dtype=np.float32)).cuda()
    # Y_train = torch.from_numpy(Y_train.to_numpy(dtype=np.float32)).cuda()
    # X_test = torch.from_numpy(X_test.to_numpy(dtype=np.float32)).cuda()
    # Y_test = torch.from_numpy(Y_test.to_numpy(dtype=np.float32)).cuda()
    
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
    print(self.dataset)
    model_file_dir = os.path.join(self.model_path, 'base_model_' + self.dataset + '.pickle') 
    with open(model_file_dir, 'wb') as handle:
        pickle.dump(self.base_dict, handle)
    
  def add_noise(self, data, dataset, sensitive_feature, label, noise_ratio, protected=False):
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

  def add_noise_new(self, data_demo, dataset, noise_ratio, dict_map):  # works fine!

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

  def get_model_pred(self, item): # an item is one row of the dataset
    if isinstance(self.model_obj, LogisticRegression_pytorch):
      item = np.asarray(item, dtype=np.float32)
      # convert to tensor
      item = torch.from_numpy(item)
      score = self.model_obj(item.cuda()).squeeze()
      return score
    score = self.model_obj.predict_proba(item).squeeze() # [p(y = 0), p(y = 1)]
    return score

  def sample_from_prob(self, dist, size):
    preds = [0.0, 1.0]
    dist /= dist.sum()
    if isinstance(self.model_obj, LogisticRegression_pytorch):
      dist_t = dist.cpu().detach().numpy()
      sample_preds = np.random.choice(preds, size, True, dist_t)
      return sample_preds
    sample_preds = np.random.choice(preds, size, True, dist)
    return sample_preds

  def sample_superhuman(self):
    start_time = time.time()
    print("")
    train_data_filename = "train_data_" + make_experiment_filename(dataset = self.dataset, demo_baseline = self.demo_baseline, lr_theta = self.lr_theta, num_of_demos = self.num_of_demos, noise_ratio = self.noise_ratio) + ".csv"
    train_file_path = os.path.join(self.train_data_path, train_data_filename)
    print(train_file_path)
    self.train_data = pd.read_csv(train_file_path, index_col=0)
    X = self.train_data.drop(columns=[self.label]).to_numpy(dtype=np.float32)
    data_size, feature_size = self.train_data.shape
    self.sample_matrix = np.zeros((self.num_of_demos, data_size)) #np.array([[-1 for _ in range(data_size)] for _ in range(num_of_samples)]) # create a matrix of size [num_of_samples * data_set_size]. Each row is a sample from our model that predicts the self.label of dataset.
    for j in range(data_size):
      # print("check X[j] here", type(X[j]))
      probs = self.get_model_pred(item = [X[j]] )
      self.sample_matrix[:,j] = self.sample_from_prob(dist = probs, size = self.num_of_demos) # return a vector of size num_of_samples (50) with self.label prediction samples for j_th item of the dataset
    print("--- %s end of sample_superhuman ---" % (time.time() - start_time))
    return self.sample_matrix

  def get_samples_demo_indexed(self, demo_list):
    sample_size = demo_list[0].idx_test.size
    self.sample_matrix_demo_indexed = np.zeros((self.num_of_demos, sample_size))
    for i in range(self.num_of_demos):
      demo = demo_list[i]
      self.sample_matrix_demo_indexed[i,:] = self.sample_matrix[i,:][demo.idx_test]
    return self.sample_matrix_demo_indexed
    
  def get_sample_loss(self, demo_list):
    """
      get feature cost fk
    """
    start_time = time.time()
    self.sample_loss = np.zeros((self.num_of_demos, self.num_of_features))
    
    # self.sample_loss = torch.from_numpy(self.sample_loss).cuda()
    for demo_index, x in enumerate(tqdm(demo_list)):
      demo = demo_list[demo_index] # get demonstrator
      sample_preds = self.sample_matrix_demo_indexed[demo_index,:]
      # Metrics
      models_dict = {"Super_human": (sample_preds, sample_preds)}
      # indices of students that demonstrator has seen
      y = self.train_data.loc[demo.idx_test][self.label] # we use true_y from original dataset since y_true in demo can be noisy (in noise setting)
      A = self.train_data.loc[demo.idx_test][self.sensitive_feature]
      A_str = A.map(self.dict_map)
      metric_df = get_metrics_df(models_dict = models_dict, y_true = y, group = A_str, feature = self.feature, is_demo = False)
      for feature_index in range(self.num_of_features):
        self.sample_loss[demo_index, feature_index] = metric_df.loc[self.feature[feature_index]]["Super_human"] #metric[feature_index]
    print("--- %s end of get_sample_loss ---" % (time.time() - start_time))

  def get_demo_loss(self, demo_index, feature_index):
    demo_loss = self.demo_list[demo_index].metric[feature_index]
    return demo_loss

  def get_subdom_constant(self, subdom_tensor_param):
    if isinstance(self.model_obj, LogisticRegression_pytorch):
      if self.c == None:
        #convert self.subdom_tensor to tensor
        subdom_tensor = torch.from_numpy(subdom_tensor_param)
        subdom_constant = torch.mean(subdom_tensor)
        return subdom_constant
      
    if self.c == None:  # only update it in the first iteration with the initial sample values
      subdom_constant = np.mean(subdom_tensor_param)
    return subdom_constant

  def compute_exp_phi_X_Y(self, demo_list):
    
    train_data_filename = "train_data_" + make_experiment_filename(dataset = self.dataset, demo_baseline = self.demo_baseline, lr_theta = self.lr_theta, num_of_demos = self.num_of_demos, noise_ratio = self.noise_ratio) + ".csv"
    train_file_path = os.path.join(self.train_data_path, train_data_filename)

    self.train_data = pd.read_csv(train_file_path, index_col=0)
    X = self.train_data.drop(columns=[self.label])
    self.exp_phi_X_Y = [0 for _ in range(self.num_of_attributs)]
    self.phi_X_Y = []
    for i in range(self.num_of_demos):
      demo = demo_list[i]
      sample_Y = self.sample_matrix_demo_indexed[i,:]
      phi_X_Y_temp = np.reshape(sample_Y, (-1, 1)) * X.loc[demo.idx_test]
      phi_X_Y_temp = np.sum(phi_X_Y_temp, axis=0) / X.shape[0]
      self.phi_X_Y.append(phi_X_Y_temp)
      self.exp_phi_X_Y += phi_X_Y_temp
    self.exp_phi_X_Y /= self.num_of_demos  # get the average

  def feature_matching(self, demo_index, demo_list):
    demo = demo_list[demo_index]
    X_demoIndexed = self.train_data.drop(columns=[self.label]).loc[demo.idx_test] 
    sample_Y_demoIndexed = self.sample_matrix_demo_indexed[demo_index,:]#[demo.idx_test]
    phi_X_Y = np.reshape(sample_Y_demoIndexed, (-1, 1)) * X_demoIndexed
    phi_X_Y = np.sum(phi_X_Y, axis=0) / X_demoIndexed.shape[0]

    return phi_X_Y - self.exp_phi_X_Y

  def compute_grad_theta(self, demo_list):
    start_time = time.time()
    self.subdom_tensor = np.zeros((self.num_of_demos, self.num_of_features)) 
    self.compute_exp_phi_X_Y(demo_list) 
    grad_theta = [0.0 for _ in range(self.num_of_attributs)]
    for j, x in enumerate(tqdm(demo_list)):
      if j == 0: self.subdom_constant = 0
      else: self.subdom_constant = self.get_subdom_constant(self.subdom_tensor)
      for k in range(self.num_of_features):
        sample_loss = self.sample_loss[j, k] #self.get_sample_loss(j, k)
        demo_loss = demo_list[j].metric[k]
        self.subdom_tensor[j, k] = max(self.alpha[k]*(sample_loss - demo_loss) + 1, 0) - self.subdom_constant     # subtract constant c to optimize for useful demonstation instead of avoiding from noisy ones
        grad_theta += self.subdom_tensor[j, k] * self.feature_matching(j, demo_list)
        
    print("--- %s end of compute_grad_theta ---" % (time.time() - start_time))
    subdom_tensor_sum = np.sum(self.subdom_tensor)
    print("subdom tensor sum: ", subdom_tensor_sum)
    return subdom_tensor_sum, grad_theta

  def compute_alpha(self, demo_list):
    start_time = time.time()
    alpha = np.ones(self.num_of_features)
  
    for k in range(self.num_of_features):
      sorted_demos = []
      alpha_candidate = []
      for j in range(self.num_of_demos):
        sample_loss = self.sample_loss[j, k]
        demo_loss = demo_list[j].metric[k] 

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
    find_gamma_superhuman(demo_list, self.model_params)
    return alpha

  def eval_model_baseline(self, dataset, noise_ratio, dict_map, baseline="pp", mode="demographic_parity"):
    # add_noise_new(self, data_demo, dataset, noise_ratio, dict_map)
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
      demo = data_demo(X_train, X_train, Y_train, Y_train, A_train, A_train, A_str_train, A_str_train, X_train.index, X_train.index)
      demo = add_noise_new(demo, self.dataset, self.noise_ratio, self.dict_map) # in this function we only add noise to train_Y and test_A: so instead of test_A we use train_A in the input
      Y_train, A_train = demo.train_y, demo.test_A
      if self.dataset == 'Adult': 
        save_AXY(self.dataset, A_train - 1, X_train, Y_train, A_test - 1, X_test, Y_test)
      elif self.dataset == 'COMPAS':
        save_AXY(self.dataset, A_train, X_train, Y_train, A_test, X_test, Y_test)


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
      return get_metrics_df(models_dict = models_dict, y_true = Y_test, group = A_str_test, feature = self.feature, is_demo = False)
    
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
      metrics = get_metrics_df(models_dict = models_dict, y_true = Y_test, group = A_str_test, feature = self.feature, is_demo = False)
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
      metrics = get_metrics_df(models_dict = models_dict, y_true = Y_test, group = A_str_test, feature = self.feature, is_demo = False)
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
    # test = self.model_obj.predict_proba(X)

    if isinstance(X, pd.DataFrame):
      X = X.to_numpy(dtype=np.float32)
      X = torch.from_numpy(X)
      
    print(self.model_obj)
    print(type(X))
    if isinstance(self.model_obj, LogisticRegression_pytorch):
      scores = self.model_obj(X.cuda())[:, 1]
    else:
      scores = self.model_obj.predict_proba(X)[:, 1]
    
    print(scores)
    print(np.mean(Y_train))
    # Predictions (0 or 1) on test set
    preds = (scores >= np.mean(Y_train)) * 1
    print(preds)
    #preds = predict_nn(X, Y_train, A, X, Y_test, A, self)
    # Metrics
    eval = pd.DataFrame(index = [self.feature[i] for i in range(self.num_of_features)]) #['Demographic parity difference', 'False negative rate difference', 'ZeroOne']
    
    if isinstance(self.model_obj, LogisticRegression_pytorch):
      models_dict = {
                "Super_human": (preds.cpu().data.numpy(), preds.cpu().data.numpy())}
    else:
      models_dict = {
              "Super_human": (preds, preds)}
    eval = get_metrics_df(models_dict = models_dict, y_true = Y_test, group = A_str, feature = self.feature, is_demo = False)
    return eval

  def update_model_alpha(self, new_alpha):
    self.alpha = new_alpha
  
  def update_model(self, lr_theta, iters):
    self.lr_theta = lr_theta
    self.grad_theta, subdom_tensor_sum_arr, self.eval, self.gamma_superhuman_arr = [], [], [], []
    Y = self.train_data[self.label]
    X = self.train_data.drop(columns=[self.label])
    X = torch.tensor(X.values, dtype=torch.float32)
    
    print(Y)
    print("self.base_model_type: ", self.model_obj.type)
    gamma_degrade = 0
    print("after it returned")
    demo_list = read_demo_list(data_path = self.data_path, dataset = self.dataset, demo_baseline = self.demo_baseline, num_of_demos = self.num_of_demos, noise_ratio = self.noise_ratio)
    for i in tqdm(range(iters)):
      # find sample loss and store it, we will use it for computing grad_theta and grad_alpha
      # if not isinstance(self.model_obj, LogisticRegression_pytorch):
      self.sample_superhuman() # update self.sample_matrix with new samples from new theta # sample_matrix
      self.get_samples_demo_indexed(demo_list) # sample_matrix_demo_indexed
      self.get_sample_loss(demo_list) # cost feature fk(xi)
      
      
      alpha = self.alpha
      if isinstance(self.model_obj, LogisticRegression_pytorch):
        print("model is LogisticRegression_pytorch")
        # get the current theta and alpha
        # theta = self.model_obj.get_model_theta()
        # print(len(theta))
        # find new theta
        # subdom_tensor_sum, grad_theta = self.compute_grad_theta() # computer gradient of loss w.r.t theta by sampling from our model

        # new_theta = theta - self.lr_theta * grad_theta  # update theta using the gradinet values
        
        # 1. compute sample_loss (sample loss contains f vectors)
        
        # with torch.no_grad():
        
        # pred = self.model_obj(X)
        # 2. compute subdom_tensor
        
        # pred = self.model_obj(X)
        # get the positive pred values
        # preds = pred[:, 1]
        self.model_obj.optimizer.zero_grad()
        self.subdom_tensor = np.zeros((self.num_of_demos, self.num_of_features))
        self.predss = []
        temp_subdom_sum = 0
        subdom_tensor_sum = 0
        loss = 0
        for j, x in enumerate(tqdm(demo_list)):
          X_test = X[x.idx_test.to_numpy(),:]
          
          if j == 0:
            self.subdom_constant = 0
          else:
            self.subdom_constant = self.get_subdom_constant(self.subdom_tensor)
          pred = self.model_obj(X_test.cuda())
          preds = pred[:, 1]
          for k in range(self.num_of_features):
            sample_loss = self.sample_loss[j, k]
            demo_loss = demo_list[j].metric[k]
            self.subdom_tensor[j, k] = max(alpha[k]*(sample_loss - demo_loss) + 1, 0) - self.subdom_constant
            # convert self.subdom_tensor[j, k] to tensor
            temp_subdom_sum = torch.sum(torch.tensor(self.subdom_tensor[j, k], dtype=torch.float32))
            loss += torch.sum(preds) * temp_subdom_sum / len(demo_list)
            subdom_tensor_sum += temp_subdom_sum
          # print(pred)
          # exit()
          # self.preds.append(pred)
        # subdom_tensor_sum = np.sum(self.subdom_tensor) / len(self.demo_list)
        # convert subdom_tensor_sum to tensor
        # self.subdom_tensor = torch.tensor(self.subdom_tensor, dtype=torch.float32)
        # 3. compute loss
        # loss = torch.sum(torch.sum(self.predss) * self.subdom_tensor)
        loss.backward()
        self.model_obj.optimizer.step()
  
        # self.model_obj.optimizer.zero_grad()
        # # loss = multiply(prob(y_hat given Xi)) * subdom_tensor
        # self.subdom_tensor = np.zeros((self.num_of_demos, self.num_of_features)) 
        # self.sample_loss = np.zeros((self.num_of_demos, self.num_of_features))
        # loss = 0
        # for j, demo in enumerate(tqdm(self.demo_list)):
        #   # breakpoint()
        #   X_test = X[demo.idx_test.to_numpy(),:]
        #   pred = self.model_obj(X_test)
        #   y_hat_probs, _ = torch.max(pred, dim=1)
        #   with torch.no_grad():
        #     y_hat = torch.argmax(pred,dim=1).cpu().detach().numpy()
        #   models_dict = {"Super_human": (y_hat, y_hat) }
        #   y = self.train_data.loc[demo.idx_test][self.label] # we use true_y
        #   A = self.train_data.loc[demo.idx_test][self.sensitive_feature]
        #   A_str = A.map(self.dict_map)
        #   try:
        #     metric_df = get_metrics_df(models_dict = models_dict, y_true = y, group = A_str,\
        #     feature = self. feature, is_demo = False)
        #   except:
        #     import ipdb; ipdb.set_trace()
        #   f_hat = []
        #   f_tilde = []
        #   #f_demo vector
        #   for feature_index in range(self.num_of_features):
        #     f_hat.append(metric_df.loc[self.feature[feature_index]]["Super_human"])
        #     f_tilde.append(demo.metric[feature_index])
        #   f_hat = np. asarray(f_hat)
        #   self.sample_loss [j, :] = f_hat # update sample loss matrix
        #   f_tilde = np.asarray(f_tilde)
        #   subdom = np.maximum(np.zeros(self.num_of_features), alpha*(f_hat - f_tilde) + beta).sum()
        #   # (1, )
        #   log_prob_sum = torch.log(y_hat_probs).sum()
        #   loss += log_prob_sum * subdom
          
          # if j == 0:
          #   self.subdom_constant = 0
          # else:
          #   self.subdom_constant = self.get_subdom_constant()
          
          # for k in range(self.num_of_features):
          #   sample_loss = self.sample_loss[j, k]
          #   demo_loss = self.demo_list[j].metric[k]
          #   self.subdom_tensor[j, k] = max(alpha[k]*(sample_loss - demo_loss) + 1, 0) - self.subdom_constant
            
        # subdom_tensor_sum = np.sum(self.subdom_tensor) / len(self.demo_list)
        # convert subdom_tensor_sum to tensor
        # self.subdom_tensor = torch.tensor(self.subdom_tensor, dtype=torch.float32)
        
        # breakpoint()
        # loss = torch.sum(torch.sum(pred, dim=0) * self.subdom_tensor)
        # loss.backward()
        # self.model_obj.optimizer.step()
        # subdom_tensor_sum = loss.detach().cpu().item()
        # find new alpha
        if i == 0:
          print("eval from first sample: ")
          print(self.eval_model(mode = "train"))
        new_alpha = self.compute_alpha(demo_list)
        #update theta
        # self.model_obj.update_model_theta(new_theta)
        # self.grad_theta.append(grad_theta)
        #update alpha
      elif isinstance(self.model_obj, LR_base_superhuman_model):
        print("model is LR_base")
        # get the current theta and alpha
        theta = self.model_obj.get_model_theta()
        print(len(theta))
        # find new theta
        subdom_tensor_sum, grad_theta = self.compute_grad_theta(demo_list) # computer gradient of loss w.r.t theta by sampling from our model

        new_theta = theta - self.lr_theta * grad_theta  # update theta using the gradinet values
        # find new alpha
        if i == 0:
          print("eval from first sample: ")
          print(self.eval_model(mode = "train"))
        new_alpha = self.compute_alpha(demo_list)
        #update theta
        self.model_obj.update_model_theta(new_theta)
        self.grad_theta.append(grad_theta)

      self.update_model_alpha(new_alpha)
      # eval model
      eval_i = self.eval_model(mode = "train")
      print("eval_i:")
      print(eval_i)
      
      # store some stuff
      subdom_tensor_sum_arr.append(subdom_tensor_sum)
      self.eval.append(eval_i)
      model_params = {"model":self.model_obj, "theta": self.model_obj.get_model_theta(), "alpha":self.alpha, "eval": self.eval, "subdom_value": subdom_tensor_sum_arr, "lr_theta": self.lr_theta, "num_of_demos":self.num_of_demos, "iters": iters, "num_of_features": self.num_of_features, "demo_baseline": self.demo_baseline, "feature": self.feature}
      gamma_superhuman = find_gamma_superhuman(demo_list, model_params)
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

    model_file_dir = os.path.join(self.model_path, 'base_model_' + self.dataset + '.pickle') 
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
    # eval_model_baseline(self, dataset, noise_ratio, dict_map, baseline="pp", mode="demographic_parity")
    eval_pp_dp = self.eval_model_baseline(self.dataset, self.noise_ratio, self.dict_map, baseline = "pp", mode = "demographic_parity")
    print()
    print(eval_pp_dp)
    eval_pp_eqodds = self.eval_model_baseline(self.dataset, self.noise_ratio, self.dict_map, baseline = "pp", mode = "equalized_odds")
    print()
    print(eval_pp_eqodds)
    eval_fairll_dp = self.eval_model_baseline(self.dataset, self.noise_ratio, self.dict_map, baseline = "fair_logloss", mode = "demographic_parity")
    print()
    print(eval_fairll_dp)
    eval_fairll_eqodds = self.eval_model_baseline(self.dataset, self.noise_ratio, self.dict_map, baseline = "fair_logloss", mode = "equalized_odds")
    print()
    print(eval_fairll_eqodds)
    eval_fairll_eqopp = self.eval_model_baseline(self.dataset, self.noise_ratio, self.dict_map, baseline = "fair_logloss", mode = "equalized_opportunity")
    print()
    print(eval_fairll_eqopp)
    eval_MFOpt = self.eval_model_baseline(self.dataset, self.noise_ratio, self.dict_map, baseline = "MFOpt")
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