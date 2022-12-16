import numpy as np
import pandas as pd
# Metrics
from fairlearn.metrics import (
    MetricFrame,
    selection_rate, demographic_parity_difference, demographic_parity_ratio,
    false_positive_rate, false_negative_rate, true_negative_rate, true_positive_rate,
    false_positive_rate_difference, false_negative_rate_difference,
    equalized_odds_difference)
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, zero_one_loss, confusion_matrix


def true_positives(y_true, y_pred):
    tp = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            tp += 1
    return tp


def false_positives(y_true, y_pred):
    fp = 0
    for i in range(len(y_true)):
        if y_true[i] == 0 and y_pred[i] == 1:
            fp += 1
    return fp

# backup
# def positive_predictive_value_helper(y_true, y_pred):
#     tp = np.sum(np.logical_and(y_true, y_pred))
#     fp = np.sum(np.logical_and(np.logical_not(y_true), y_pred))
#     return tp / (tp + fp)


# def positive_predictive_value_helper(y_true, y_pred):
#     confusion_matrix = confusion_matrix(y_true, y_pred)
#     tp = confusion_matrix[1][1]
#     fp = confusion_matrix[0][1]
#     return tp / (tp + fp)




def positive_predictive_value_helper(y_true, y_pred):
    tp = true_positives(y_true, y_pred)
    fp = false_positives(y_true, y_pred)
    return tp / (tp + fp)

def positive_predictive_value(y_true, y_pred, group):
    #tp = np.sum(np.logical_and(y_true, y_pred))
    #fp = np.sum(np.logical_and(np.logical_not(y_true), y_pred))
    #return tp / (tp + fp)
    return MetricFrame(metrics=positive_predictive_value_helper, y_true=y_true, y_pred=y_pred, sensitive_features=group).difference(method='between_groups')
    

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
        "False positive rate difference": (
            lambda x: false_positive_rate_difference(y_true, x, sensitive_features=group), True),
        "False negative rate difference": (
            lambda x: false_negative_rate_difference(y_true, x, sensitive_features=group), True),
        "Equalized odds difference": (
            lambda x: equalized_odds_difference(y_true, x, sensitive_features=group), True),
        "ZeroOne": (
            lambda x: zero_one_loss(y_true, x), True),
        "positive_predictive_value_difference": (
            lambda x: positive_predictive_value(y_true, x, group), True)
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




g = ["m", "m", "m","m", "m", "m","m", "m", "m","m", "m", "m","m", "m", "m","m", "m", "m","m", "m", "m","m", "m", "m","m", "m", "m","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f"] #np.concatenate((np.ones(27), np.zeros(18)), axis=0)
print(len(g))
y_true = list(np.concatenate((np.ones(9), np.zeros(18), np.ones(9), np.zeros(9)), axis=0))
y_pred = list(np.concatenate((np.ones(3), np.zeros(6), np.ones(3), np.zeros(15), np.ones(2), np.zeros(7), np.ones(2), np.zeros(7)), axis=0))


 

#df_g = pd.DataFrame(g, columns=['gender'])

ppv = positive_predictive_value(y_true, y_pred, g)

print(ppv)