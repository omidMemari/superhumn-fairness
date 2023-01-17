import numpy as np
import pandas as pd
# Metrics
from fairlearn.metrics import (
    MetricFrame,
    selection_rate, demographic_parity_difference, demographic_parity_ratio,
    false_positive_rate, false_negative_rate, true_negative_rate, true_positive_rate,
    false_positive_rate_difference, false_negative_rate_difference,
    equalized_odds_difference)
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, zero_one_loss

# def dp_difference(y_true, y_pred, group):
#     g0, g1, n0, n1 = 0, 0, 0, 0
#     print(len(y_true))
#     print(group)
#     for i in range(len(y_true)):
#         if group.loc[i] == 'Male':
#             n1 += 1
#             if y_pred[i] == 1:
#                 g1 += 1
#         else: 
#             n0 += 1
#             if y_pred[i] ==1:
#                 g0 += 1
#     dp1 = g1/n1
#     dp0 = g0/n0

#     return abs(dp1 - dp0)

#     #return abs(np.mean(proba[A == 1]) - np.mean(proba[A == 0]))


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

# backup
# def positive_predictive_value_helper(y_true, y_pred):
#     tp = np.sum(np.logical_and(y_true, y_pred))
#     fp = np.sum(np.logical_and(np.logical_not(y_true), y_pred))
#     return tp / (tp + fp)


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

# backup
# def positive_predictive_value_helper(y_true, y_pred):
#     tp = np.sum(np.logical_and(y_true, y_pred))
#     fp = np.sum(np.logical_and(np.logical_not(y_true), y_pred))
#     return tp / (tp + fp)


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
def get_metrics_df(models_dict, y_true, group):
    # print("models_dict: ")
    # print(models_dict)
    # print("y_true: ")
    # print(y_true)
    metrics_dict = {

         "ZeroOne": (
            lambda x: zero_one_loss(y_true, x), True),
        "Demographic parity difference": (
            lambda x: demographic_parity_difference(y_true, x, sensitive_features=group), True), #dp_difference(y_true, x, group), True),
        "Equalized odds difference": (
            lambda x: equalized_odds_difference(y_true, x, sensitive_features=group), True),
        "Predictive value difference": (
            lambda x: predictive_value(y_true, x, group), True)
        #"Overall selection rate": (
        #    lambda x: selection_rate(y_true, x), True),

        
        #"Demographic parity ratio": (
        #    lambda x: demographic_parity_ratio(y_true, x, sensitive_features=group), True),
        #"Overall balanced error rate": (
        #    lambda x: 1-balanced_accuracy_score(y_true, x), True),
        #"Balanced error rate difference": (
        #    lambda x: MetricFrame(metrics=balanced_accuracy_score, y_true=y_true, y_pred=x, sensitive_features=group).difference(method='between_groups'), True),
        # "False positive rate difference": (
        #     lambda x: false_positive_rate_difference(y_true, x, sensitive_features=group), True),
        # "False negative rate difference": (
        #     lambda x: false_negative_rate_difference(y_true, x, sensitive_features=group), True),
        
        # "Positive predictive value difference": (
        #     lambda x: positive_predictive_value(y_true, x, group), True),
        # "Negative predictive value difference": (
        #     lambda x: negative_predictive_value(y_true, x, group), True),
        #"Overall AUC": (
        #    lambda x: 1.0 - roc_auc_score(y_true, x), False),
        #"AUC difference": (
        #    lambda x: MetricFrame(metrics=roc_auc_score, y_true=y_true, y_pred=x, sensitive_features=group).difference(method='between_groups'), False),
    }
    df_dict = {}
    for metric_name, (metric_func, use_preds) in metrics_dict.items():
        df_dict[metric_name] = [metric_func(preds) if use_preds else metric_func(scores) 
                                for model_name, (preds, scores) in models_dict.items()]
    return pd.DataFrame.from_dict(df_dict, orient="index", columns=models_dict.keys())

