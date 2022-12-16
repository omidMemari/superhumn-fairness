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

def positive_predictive_value(y_true, y_pred, group):
    # PPV for every group
    for group_value in group.unique():
        # Get the indices for the group
        group_indices = group == group_value
        # Get the true positives
        true_positives = np.logical_and(y_true[group_indices], y_pred[group_indices])
        # Get the false positives
        false_positives = np.logical_and(np.logical_not(y_true[group_indices]), y_pred[group_indices])
        # Calculate the PPV
        ppv = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_positives))
        print(f"PPV for {group.name}={group_value}: {ppv}")
    # get difference in PPV
    ppv_diff = np.abs(ppv[0] - ppv[1])
    return ppv_diff
    

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