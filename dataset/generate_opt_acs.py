import numpy as np
import pandas as pd
import sys
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import folktables
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def validate_bins(df, n_bins):
    df = df.groupby(['bin','g']).agg({'s': ['mean', 'count']}).reset_index()
    nonoverlap_bins = []
    for i in range(1, n_bins+1):
        df_sub = df.loc[df['bin']==i]
        if df_sub.shape[0] < 2:
            nonoverlap_bins.append(i)
    return nonoverlap_bins

pathX = "/home/lvu5/temp-fair/superhumn-fairness/dataset/acs_west_income/m_X_data.csv"
pathY = "/home/lvu5/temp-fair/superhumn-fairness/dataset/acs_west_income/m_Y_data.csv"
pathA = "/home/lvu5/temp-fair/superhumn-fairness/dataset/acs_west_income/m_A_data.csv"
task_name = "acs_west_income"

dataX, dataY, dataA = pd.read_csv(pathX, sep='\t'), pd.read_csv(pathY), pd.read_csv(pathA)
dataX = dataX.to_numpy()
dataY = dataY.to_numpy()
dataA = dataA.to_numpy()
dataY = dataY.astype('float64')
dataA = dataA.astype('float64')
dataA = dataA - 1
dataA = dataA.ravel()
dataY = dataY.ravel()
print(dataX)
print(dataX.shape)
print("******************")
print(dataY)
print(dataY.shape)
print("******************")
print(dataA)
print(dataA.shape)

n_trials = 1
n_bins = 2
n_trials_success = 0
i = 0
# exit()
while n_trials_success < n_trials:
    i += 1
    seed = i*i*100
    
    X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
        dataX, dataY, dataA, test_size=0.3, random_state=seed)

    # also train a more powerful model based on ERM 
    tune_grid = {"n_estimators": [100],
                 "max_features":['auto',0.5],
                 "max_depth":[3, 4, 8, 10],
                 "min_samples_leaf":[1, 2, 4],
                 "min_samples_split":[2, 4, 6]}
    rf = RandomForestClassifier(random_state=42)
    rf_grid = RandomizedSearchCV(estimator = rf, param_distributions = tune_grid,
                                 n_iter = 10, cv = 2, verbose=2, random_state=seed, n_jobs = -1)
    rf_grid.fit(np.concatenate([X_train, A_train.reshape(-1,1)], axis=1), y_train)
    
    best_params = rf_grid.best_params_
    print(best_params)
    
    model = rf_grid.best_estimator_
    
    yhat_train_erm = model.predict_proba(np.concatenate([X_train, A_train.reshape(-1,1)], axis=1))[:,1]
    yhat_test_erm = model.predict_proba(np.concatenate([X_test, A_test.reshape(-1,1)], axis=1))[:,1]                          
    
    train_erm = pd.DataFrame({'s':yhat_train_erm, 'y': y_train.astype('int'), 'g': A_train+1})
    test_erm = pd.DataFrame({'s':yhat_test_erm, 'y': y_test.astype('int'), 'g': A_test+1})
    roc_auc_score(train_erm['y'], train_erm['s'])
    roc_auc_score(test_erm['y'], test_erm['s'])
    
    bins, cuts = pd.qcut(train_erm['s'], q=n_bins, retbins=True, duplicates='drop')
    cut_midpoints = [cuts[i]+(cuts[i+1] - cuts[i])/2 for i in range(len(cuts)-1)]
    cuts[0], cuts[-1] = 0.0, 1.0
    # Cut both train_erm and test_erm scores into bins based on cuts from the train_erming data
    train_erm['bin']=pd.cut(train_erm['s'], bins = cuts, include_lowest=True, labels = False)
    train_erm['bin']=train_erm['bin']+1
    test_erm['bin']=pd.cut(test_erm['s'], bins = cuts, include_lowest=True, labels = False)
    test_erm['bin']=test_erm['bin']+1
    print("Here")
    try:
        print("hello")
        assert(validate_bins(train_erm,n_bins) == [])
        assert(validate_bins(test_erm,n_bins) == [])
        print("passed assertion")
        folder = "./" + task_name + "/"
        print(folder)
        if not os.path.exists(folder):
            os.mkdir(folder)
        print(folder)
        pd.DataFrame(cut_midpoints, columns=['bin_midpoints']).to_csv(folder+"/"+task_name+'_bin_midpoints.csv', index=False)
        # to put bin midpoint, train test to a folder for opt
        train_erm.to_csv(folder+"/"+task_name+'_train.csv', index=False)
        test_erm.to_csv(folder+"/"+task_name+'_test.csv', index=False)
        n_trials_success += 1
    except:
        break
    
