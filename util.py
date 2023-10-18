import numpy as np
import pandas as pd
import pickle
import os

# Metrics
from fairlearn.metrics import (
    MetricFrame,
    selection_rate, demographic_parity_difference, demographic_parity_ratio,
    false_positive_rate, false_negative_rate, true_negative_rate, true_positive_rate,
    false_positive_rate_difference, false_negative_rate_difference,
    equalized_odds_difference)
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, zero_one_loss

sample_record_filename_template = "{}_{}_{}_{}_{}"

feature_expand_dict = {'inacc': "ZeroOne", 'dp': "Demographic parity difference", 'eqodds': "Equalized odds difference", 'prp': "Predictive value difference", 'eqopp': "False negative rate difference",  'fnr': "False negative rate difference", 'fpr': "False positive rate difference", 'ppv': "Positive predictive value difference", 'npv': "Negative predictive value difference", 'auc': "Overall AUC", 'auc_diff': "AUC difference", 'error_rate_diff': "Balanced error rate difference"}

def compute_error(Yhat,proba,Y):
    err = 1 - np.sum(Yhat == Y) / Y.shape[0] 
    exp_zeroone = np.mean(np.where(Y == 1 , 1 - proba, proba))
    return err, exp_zeroone

def create_features_dict(feature_list):
  num_of_features = len(feature_list)
  feature = {}
  for i, f in enumerate(feature_list):
        print(i, f)
        feature[i] = feature_expand_dict[f]
  return feature, num_of_features

def make_experiment_filename(**kwargs):
    return sample_record_filename_template.format(kwargs['dataset'], kwargs['demo_baseline'], kwargs['lr_theta'],  kwargs['num_of_demos'], kwargs['noise_ratio']).replace('.','-')

def make_demo_list_filename(**kwargs):
    return "demo_list_{}_{}_{}_{}".format(kwargs['dataset'], kwargs['demo_baseline'],  kwargs['num_of_demos'], kwargs['noise_ratio']).replace('.','-')


def store_object(obj, path, name, exp_idx):
    if not os.path.exists(path):
        os.makedirs(path)
    filepath = os.path.join(path,name)
    from main import default_args
    if exp_idx == -1 or exp_idx == 0:
        with open(filepath, 'wb') as file:
            pickle.dump(obj,file)
        print("exp {} wrote to {}".format(exp_idx, filepath))
    elif exp_idx > 0 and exp_idx <  default_args['num_experiment'] - 1:
        with open(filepath, 'ab+') as file:
            #file = open(filepath, 'wb')
            pickle.dump(obj,file)
        print("exp {} wrote to {}".format(exp_idx, filepath))
    elif exp_idx == default_args['num_experiment'] - 1:
        with open(filepath, 'ab+') as file:
            #file = open(filepath, 'ab+')
            pickle.dump(obj, file)
        #file.close()
        print("exp {} wrote to {}".format(exp_idx, filepath))

        

def load_object(path, name, exp_num):
    if exp_num == -1:
        with open(os.path.join(path,name), 'rb') as file:
            return pickle.load(file)
    elif exp_num > -1:
        data = []
        with open(os.path.join(path, name), 'rb') as file:
            try:
                while True:
                    data.append(pickle.load(file))
            except EOFError:
                pass
        return data


def find_gamma_superhuman_all(demo_list, model_params):
  if not model_params: return
  feature = model_params["feature"]
  num_of_features = model_params["num_of_features"]
  print("gamma-superhuman: ")
  gamma_superhuman_arr = []
  baseline = {0: 'eval_pp_dp', 1:'eval_pp_eq_odds', 2:'eval_fairll_dp', 3:'eval_fairll_eqodds', 4:'eval_MFOpt', 5: 'superhuman'}
  baseline_loss = np.zeros(len(baseline))
  dominated = np.zeros(len(baseline))
  for j in range(len(demo_list)):
    count_baseline = np.zeros(len(baseline))
    for i in range(num_of_features):
      demo_loss = demo_list[j].metric[i] #for z in range(len(demo_list))]
      model_loss = model_params['eval'][-1].loc[feature[i]][0]
      baseline_loss[-1] = model_loss
      for k in range(len(baseline)-1):
        baseline_loss[k] = model_params[baseline[k]].loc[feature[i]][0]
      for k in range(len(baseline)):
        if baseline_loss[k] <= demo_loss:
          count_baseline[k] += 1
          if count_baseline[k] == num_of_features:
            dominated[k] += 1
  dominated = dominated/len(demo_list)
  print(baseline)
  print("dominated:")
  print(dominated)

def find_gamma_superhuman(demo_list, model_params):
  if not model_params: return
  feature = model_params["feature"]
  num_of_features = model_params["num_of_features"]
  print("gamma-superhuman: ")
  gamma_superhuman_arr = []
  for i in range(num_of_features):
      demo_loss = [demo_list[z].metric[i] for z in range(len(demo_list))]
      model_loss = model_params['eval'][-1].loc[feature[i]][0]
      f = feature[i]
      n = len(demo_loss)
      count = 0
      for j in range(n):
          if model_loss <= demo_loss[j]:
              count += 1
      gamma_superhuman = count/n
      print(gamma_superhuman, f)
      gamma_superhuman_arr.append(gamma_superhuman)
  return gamma_superhuman_arr



def true_positives(y_true, y_pred):
    tp = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            tp += 1
    return tp

def true_negatives(y_true, y_pred):
    tn = 0
    for i in range(len(y_true)):
        if y_true[i] == 0 and y_pred[i] == 0:
            tn += 1
    return tn


def false_positives(y_true, y_pred):
    fp = 0
    for i in range(len(y_true)):
        if y_true[i] == 0 and y_pred[i] == 1:
            fp += 1
    return fp


def false_negatives(y_true, y_pred):
    fn = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 0:
            fn += 1
    return fn



def positive_predictive_value_helper(y_true, y_pred):
    tp = true_positives(y_true, y_pred)
    fp = false_positives(y_true, y_pred)
    if tp == 0 and fp == 0: return 0
    return tp / (tp + fp)

def negative_predictive_value_helper(y_true, y_pred):
    tn = true_negatives(y_true, y_pred)
    fn = false_negatives(y_true, y_pred)
    if tn == 0 and fn == 0: return 0
    return tn / (tn + fn)

def predictive_value_helper(y_true, y_pred):
    ppv = positive_predictive_value_helper(y_true, y_pred)
    npv = negative_predictive_value_helper(y_true, y_pred)
    return max(ppv, npv)

def positive_predictive_value(y_true, y_pred, group):
    return MetricFrame(metrics=positive_predictive_value_helper, y_true=y_true, y_pred=y_pred, sensitive_features=group).difference(method='between_groups')

def negative_predictive_value(y_true, y_pred, group):
    return MetricFrame(metrics=negative_predictive_value_helper, y_true=y_true, y_pred=y_pred, sensitive_features=group).difference(method='between_groups')

def predictive_value(y_true, y_pred, group) -> float:
    fns = {"ppv": positive_predictive_value_helper, "npv": negative_predictive_value_helper}
    prp = MetricFrame(
        metrics=fns,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=group,
    )

    return max(prp.difference(method="between_groups"))

def true_positives(y_true, y_pred):
    tp = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            tp += 1
    return tp

def true_negatives(y_true, y_pred):
    tn = 0
    for i in range(len(y_true)):
        if y_true[i] == 0 and y_pred[i] == 0:
            tn += 1
    return tn


def false_positives(y_true, y_pred):
    fp = 0
    for i in range(len(y_true)):
        if y_true[i] == 0 and y_pred[i] == 1:
            fp += 1
    return fp


def false_negatives(y_true, y_pred):
    fn = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 0:
            fn += 1
    return fn


def positive_predictive_value_helper(y_true, y_pred):
    tp = true_positives(y_true, y_pred)
    fp = false_positives(y_true, y_pred)
    if tp == 0 and fp == 0: return 0
    return tp / (tp + fp)

def negative_predictive_value_helper(y_true, y_pred):
    tn = true_negatives(y_true, y_pred)
    fn = false_negatives(y_true, y_pred)
    if tn == 0 and fn == 0: return 0
    return tn / (tn + fn)

def predictive_value_helper(y_true, y_pred):
    ppv = positive_predictive_value_helper(y_true, y_pred)
    npv = negative_predictive_value_helper(y_true, y_pred)
    return max(ppv, npv)

def positive_predictive_value(y_true, y_pred, group):
    return MetricFrame(metrics=positive_predictive_value_helper, y_true=y_true, y_pred=y_pred, sensitive_features=group).difference(method='between_groups')

def negative_predictive_value(y_true, y_pred, group):
    return MetricFrame(metrics=negative_predictive_value_helper, y_true=y_true, y_pred=y_pred, sensitive_features=group).difference(method='between_groups')

def predictive_value(y_true, y_pred, group) -> float:
    fns = {"ppv": positive_predictive_value_helper, "npv": negative_predictive_value_helper}
    prp = MetricFrame(
        metrics=fns,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=group,
    )

    return max(prp.difference(method="between_groups"))


# Helper functions
def get_metrics_df(models_dict, y_true, group, feature, is_demo = False):
    metrics_dict = {

         "ZeroOne": (
            lambda x: zero_one_loss(y_true, x), True),
        "Demographic parity difference": (
            lambda x: demographic_parity_difference(y_true, x, sensitive_features=group), True),
        "Equalized odds difference": (
            lambda x: equalized_odds_difference(y_true, x, sensitive_features=group), True),
        "Predictive value difference": (
            lambda x: predictive_value(y_true, x, group), True),
        "Overall selection rate": (
           lambda x: selection_rate(y_true, x), True),
        "Demographic parity ratio": (
           lambda x: demographic_parity_ratio(y_true, x, sensitive_features=group), True),
        "Overall balanced error rate": (
           lambda x: 1-balanced_accuracy_score(y_true, x), True),
        "Balanced error rate difference": (
           lambda x: MetricFrame(metrics=balanced_accuracy_score, y_true=y_true, y_pred=x, sensitive_features=group).difference(method='between_groups'), True),
        "False positive rate difference": (
            lambda x: false_positive_rate_difference(y_true, x, sensitive_features=group), True),
        "False negative rate difference": (
            lambda x: false_negative_rate_difference(y_true, x, sensitive_features=group), True),       
        "Positive predictive value difference": (
            lambda x: positive_predictive_value(y_true, x, group), True),
        "Negative predictive value difference": (
            lambda x: negative_predictive_value(y_true, x, group), True),
        "Overall AUC": (
           lambda x: 1.0 - roc_auc_score(y_true, x), False),
        "AUC difference": (
           lambda x: MetricFrame(metrics=roc_auc_score, y_true=y_true, y_pred=x, sensitive_features=group).difference(method='between_groups'), False)
    }
    df_dict = {}
    if is_demo == True:     # if we are creating demos, let's store all the metrics
        metrics_dict_subset = metrics_dict
    else:                   # otherwise only store the metrics we care about
        metrics_dict_subset = {k: metrics_dict[k] for k in feature.values()}
    for metric_name, (metric_func, use_preds) in metrics_dict_subset.items():
        df_dict[metric_name] = [metric_func(preds) if use_preds else metric_func(scores) 
                                for model_name, (preds, scores) in models_dict.items()]
    return pd.DataFrame.from_dict(df_dict, orient="index", columns=models_dict.keys())

