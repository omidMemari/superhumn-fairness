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
    return tp / (tp + fp)

def negative_predictive_value_helper(y_true, y_pred):
    tn = true_negatives(y_true, y_pred)
    fn = false_negatives(y_true, y_pred)
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
    #return MetricFrame(metrics=predictive_value_helper, y_true=y_true, y_pred=y_pred, sensitive_features=group)
    fns = {"ppv": positive_predictive_value_helper, "npv": negative_predictive_value_helper}
    #sw_dict = {"sample_weight": None}
    #sp = {"tpr": sw_dict, "fpr": sw_dict}
    prp = MetricFrame(
        metrics=fns,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=group,
    )
    # print("prp:  ", prp)
    # print("prp.diff:   ", prp.difference(method="between_groups"))
    # print("max(prp.diff):   ", max(prp.difference(method="between_groups")))

    return max(prp.difference(method="between_groups"))


# def _get_eo_frame(y_true, y_pred, sensitive_features, sample_weight) -> MetricFrame:
#     fns = {"tpr": true_positive_rate, "fpr": false_positive_rate}
#     sw_dict = {"sample_weight": sample_weight}
#     sp = {"tpr": sw_dict, "fpr": sw_dict}
#     eo = MetricFrame(
#         metrics=fns,
#         y_true=y_true,
#         y_pred=y_pred,
#         sensitive_features=sensitive_features,
#         sample_params=sp,
#     )
#     return eo 

# def equalized_odds_difference(
#     y_true, y_pred, *, sensitive_features, method="between_groups", sample_weight=None
# ) -> float:
#     """Calculate the equalized odds difference.
#     The greater of two metrics: `true_positive_rate_difference` and
#     `false_positive_rate_difference`. The former is the difference between the
#     largest and smallest of :math:`P[h(X)=1 | A=a, Y=1]`, across all values :math:`a`
#     of the sensitive feature(s). The latter is defined similarly, but for
#     :math:`P[h(X)=1 | A=a, Y=0]`."""
#     eo = _get_eo_frame(y_true, y_pred, sensitive_features, sample_weight)

#     return max(eo.difference(method=method))

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
        "Positive predictive value difference": (
            lambda x: positive_predictive_value(y_true, x, group), True),
        "Negative predictive value difference": (
            lambda x: negative_predictive_value(y_true, x, group), True),
        "Predictive value difference": (
            lambda x: predictive_value(y_true, x, group), True)
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




# g = ["m", "m", "m","m", "m", "m","m", "m", "m","m", "m", "m","m", "m", "m","m", "m", "m","m", "m", "m","m", "m", "m","m", "m", "m","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f"] #np.concatenate((np.ones(27), np.zeros(18)), axis=0)
# print(len(g))
# y_true = list(np.concatenate((np.ones(9), np.zeros(18), np.ones(9), np.zeros(9)), axis=0))
# y_pred = list(np.concatenate((np.ones(3), np.zeros(6), np.ones(3), np.zeros(15), np.ones(2), np.zeros(7), np.ones(2), np.zeros(7)), axis=0))

# ppv = positive_predictive_value(y_true, y_pred, g)

# print(ppv)
