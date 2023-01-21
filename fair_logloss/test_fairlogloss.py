from __future__ import print_function
from scipy.io import arff
from prepare_data import prepare_compas,prepare_IBM_adult, prepare_law

import functools
import numpy as np
import pandas as pd
import sys
from fair_logloss import DP_fair_logloss_classifier, EOPP_fair_logloss_classifier, EODD_fair_logloss_classifier


def compute_error(Yhat,proba,Y):
    err = 1 - np.sum(Yhat == Y) / Y.shape[0] 
    exp_zeroone = np.mean(np.where(Y == 1 , 1 - proba, proba))
    return err, exp_zeroone

if __name__ == '__main__':

    dataset = ""
    if sys.argv[1] == 'Adult':
        dataA,dataY,dataX,perm = prepare_IBM_adult()
        dataset = 'Adult'
    elif sys.argv[1] == 'compas':
        dataA,dataY,dataX,perm = prepare_compas()
        dataset = 'compas'
    elif sys.argv[1] == 'law':
        dataA,dataY,dataX,perm = prepare_law()
        dataset = 'law'
    else:
        raise ValueError('Invalid first arg')
    C = .005
    criteria = sys.argv[2]
    print("dataY:")
    print(dataY)
    print(type(dataY))
    print("dataX:")
    print(dataX)
    print(type(dataX))
    if criteria == 'dp':
        h = DP_fair_logloss_classifier(C=C, random_initialization=True, verbose=False)
    elif criteria == 'eqopp':
        h = EOPP_fair_logloss_classifier(C=C, random_initialization=True, verbose=False)
    elif criteria == 'eqodd':
        h = EODD_fair_logloss_classifier(C=C, random_initialization=True, verbose=False)    
    else:
        raise ValueError('Invalid second arg')
    filename_tr = "results/fairll_{}_{:.3f}_{}_tr.csv".format(dataset,C,criteria)
    filename_ts = "results/fairll_{}_{:.3f}_{}_ts.csv".format(dataset,C,criteria)
    
    outfile_tr = open(filename_tr,"w")
    outfile_ts = open(filename_ts,"w")

    for r in range(20):
        order = perm[r,:]
        tr_sz = int(np.floor(.7 * dataX.shape[0]))
        tr_idx = order[:tr_sz]
        ts_idx = order[tr_sz:]
        tr_X = dataX.reindex(tr_idx)
        ts_X = dataX.reindex(ts_idx)
        
        tr_A = dataA.reindex(tr_X.index)
        ts_A = dataA.reindex(ts_X.index)
        tr_Y = dataY.reindex(tr_X.index)
        ts_Y = dataY.reindex(ts_X.index)
        
        # Comment out to not include A in features
        tr_X = pd.concat([tr_X, tr_A], axis=1) 	
        ts_X = pd.concat([ts_X, ts_A], axis=1)
        # ---------

        for c in list(tr_X.columns):
            if tr_X[c].min() < 0 or tr_X[c].max() > 1:
                mu = tr_X[c].mean()
                s = tr_X[c].std(ddof=0)
                tr_X.loc[:,c] = (tr_X[c] - mu) / s
                ts_X.loc[:,c] = (ts_X[c] - mu) / s
        
        h.fit(tr_X.values,tr_Y.values,tr_A.values)
        exp_zo_tr = h.expected_error(tr_X.values, tr_Y.values, tr_A.values)
        exp_zo_ts = h.expected_error(ts_X.values, ts_Y.values, ts_A.values)
        err_tr = 1 - h.score(tr_X.values, tr_Y.values, tr_A.values)
        err_ts = 1 - h.score(ts_X.values, ts_Y.values, ts_A.values)
        violation_tr = h.fairness_violation(tr_X.values, tr_Y.values, tr_A.values)
        violation_ts = h.fairness_violation(ts_X.values, ts_Y.values, ts_A.values)

        print("---------------------------- Random Split %d ----------------------------------" % (r + 1))
        print("Train - predict_err : {:.3f} \t expected_err : {:.3f} \t fair_violation : {:.3f} ".format(err_tr, exp_zo_tr,violation_tr))
        print("Test  - predict_err : {:.3f} \t expected_err : {:.3f} \t fair_violation : {:.3f} ".format(err_ts, exp_zo_ts,violation_ts))
        print("")

        outfile_ts.write("{:.4f},{:.4f},{:.4f}\n".format(exp_zo_ts,err_ts, violation_ts))
        outfile_tr.write("{:.4f},{:.4f},{:.4f}\n".format(exp_zo_tr,err_tr, violation_tr))
        
    outfile_tr.close()
    outfile_ts.close()