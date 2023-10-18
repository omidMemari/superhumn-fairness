from abc import ABC,abstractmethod
from enum import Enum
from math import isclose
import numpy as np
from scipy.optimize import fmin_bfgs, minimize
import pdb
__all__ = ['DP_fair_logloss_classifier','EODD_fair_logloss_classifier','EOPP_fair_logloss_classifier']

def _log_logistic(X):
    """ This function is used from scikit-learn source code. Source link below """

    """Compute the log of the logistic function, ``log(1 / (1 + e ** -x))``.
    This implementation is numerically stable because it splits positive and
    negative values::
        -log(1 + exp(-x_i))     if x_i > 0
        x_i - log(1 + exp(x_i)) if x_i <= 0

    Parameters
    ----------
    X: array-like, shape (M, N)
        Argument to the logistic function

    Returns
    -------
    out: array, shape (M, N)
        Log of the logistic function evaluated at every point in x
    Notes
    -----
    Source code at:
    https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/extmath.py
    -----

    See the blog post describing this implementation:
    http://fa.bianp.net/blog/2013/numerical-optimizers-for-logistic-regression/
    """
    if X.ndim > 1: raise Exception("Array of samples cannot be more than 1-D!")
    out = np.empty_like(X) # same dimensions and data types

    idx = X>0
    out[idx] = -np.log(1.0 + np.exp(-X[idx]))
    out[~idx] = X[~idx] - np.log(1.0 + np.exp(X[~idx]))
    return out

def _dot_intercept(w, X):
    """ This function is used from scikit-learn source code. Source link below """

    """Computes y * np.dot(X, w).
    It takes into consideration if the intercept should be fit or not.
    Parameters
    ----------
    w : ndarray, shape (n_features,) or (n_features + 1,)
        Coefficient vector.
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.
    y : ndarray, shape (n_samples,)
        Array of labels.
    Returns
    -------
    w : ndarray, shape (n_features,)
        Coefficient vector without the intercept weight (w[-1]) if the
        intercept should be fit. Unchanged otherwise.
    c : float
        The intercept.
    yz : float
        y * np.dot(X, w).
    
    Notes
	-----
	Source code at:
    https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/linear_model/logistic.py

    """
    c = 0
    if w.size == X.shape[1] + 1:
        c = w[-1]
        w = w[:-1]

    z = np.dot(X, w) + c
    return z

def sum_truncated_loss_grad(theta, X, Y, idx1, idx0, _lambda ):
    z = _dot_intercept(theta, X)
    p = np.exp(_log_logistic(z))
    loss , grad = 0, 0
    if _lambda == 0:
        loss, grad = sum_logistic_loss_grad(theta,X,Y, np.logical_or(idx1 , idx0))
        cond_g1, cond_g0 = idx1, idx0
    else:
        p1 = np.mean(idx1)   # empirical probability of each group
        p0 = np.mean(idx0)
        if _lambda > 0:
            cond_g1 = np.logical_and( idx1 , p > p1 / _lambda)
            cond_g0 = np.logical_and( idx0 , p < (1 - p0 / _lambda))

            grad = np.sum(X[np.logical_and(cond_g1, 1 - Y)] ,0) + np.sum(-X[np.logical_and(cond_g0, Y)], 0)
            loss = (np.log(_lambda /p1)) * np.sum(cond_g1,0) + np.sum(_dot_intercept(theta,X[cond_g1]) * (1-Y[cond_g1]),0) \
                   + (np.log(_lambda/p0)) * np.sum(cond_g0,0) + np.sum(-_dot_intercept(theta,X[cond_g0]) * Y[cond_g0],0) 
        else:
            cond_g1 = np.logical_and( idx1 , p < (1 + p1 / _lambda))
            cond_g0 = np.logical_and( idx0 , p > - p0 / _lambda)

            grad = np.sum(X[np.logical_and(cond_g0, 1 - Y)],0) + np.sum(-X[np.logical_and(cond_g1, Y)], 0)
            loss = (np.log(-_lambda/p1)) * np.sum(cond_g1,0) + np.sum(-_dot_intercept(theta,X[cond_g1]) * Y[cond_g1],0) \
                   + (np.log(-_lambda/p0)) * np.sum(cond_g0,0) + np.sum(_dot_intercept(theta,X[cond_g0]) * (1-Y[cond_g0]),0) 
    return loss, grad, np.logical_or(cond_g1, cond_g0)             

def sum_logistic_loss_grad(theta, X, Y, idx):
    X, Y = X[idx,:], Y[idx]
    z = _dot_intercept(theta, X)
    p = np.exp(_log_logistic(z))
    grad = np.dot(p.T, X) - np.sum(X[Y == 1,:],0)

    logZ = z + np.log(1 + np.exp(-z))
    loss = np.sum(logZ,0) - np.sum(z * Y,0)
    
    return loss, grad 

def fairify(a,b):
    avgA, avgB = np.mean(a), np.mean(b)
    if a.size == 0:
        return np.Inf 
    elif b.size == 0:
        return np.NINF
    _lambda = 0
    if isclose(avgA,avgB):
        return _lambda  # already fair
    flipped = False
    if avgA < avgB:
        b, a = a, b
        avgA, avgB = avgB, avgA
        flipped = True
    diff = avgA - avgB
    if diff < 0:
        raise ValueError('_lambda is not supposed to be negative')
    
    a = - np.sort(-a)     # sort descending
    b.sort()                 # sort ascending

    idxA, idxB = 0, 0   # current index
    thrA, thrB = 1.0, 0.0   # current probability threshold
    gainA, gainB = 0, 0     # average gain in each group

    while True:
        if idxA < len(a):
            cd_thrA = a[idxA]
        else:
            cd_thrA = 0.0
        cd_thrB_ifA = 1 - (cd_thrA * len(b) / len(a))

        if idxB < len(b):
            cd_thrB = b[idxB]
        else:
            cd_thrB = 1.0
        
        if cd_thrB_ifA <= cd_thrB:
            next_thrA = cd_thrA
            next_thrB = cd_thrB_ifA
        else:
            next_thrA = (1 - cd_thrB) * len(a) / len(b)
            next_thrB = cd_thrB

        next_gainA = gainA + idxA * (thrA - next_thrA) / len(a)
        next_gainB = gainB + idxB * (next_thrB - thrB) / len(b)

        if isclose(next_gainA + next_gainB , diff):
            thrA = next_thrA
            _lambda = len(a) / thrA
            break
        elif next_gainA + next_gainB < diff:
            thrA = next_thrA
            thrB = next_thrB
            if cd_thrB_ifA <= cd_thrB:
                idxA += 1
            else:
                idxB += 1
            gainA = next_gainA
            gainB = next_gainB
        else:
            gain_needed = diff - gainA - gainB
            _lambda = (idxA + idxB) / (idxA * thrA / len(a) + idxB * (1 - thrB) / len(b) - gain_needed)
            break
    thrA = len(a) / _lambda
    thrB = 1 - len(b) / _lambda
    if flipped:
        _lambda = -_lambda
    avgA = np.mean(np.minimum(a, thrA))
    avgB = np.mean(np.maximum(b, thrB))

    if not isclose(avgA , avgB):
        raise ValueError('Averages not equalized %.3f vs %.3f, diff was %.3f' % (avgA, avgB, diff) )
    return _lambda 


class fair_logloss_classifier:
    def __init__(self, tol=1e-6, verbose=True, max_iter=10000, C = .001, random_initialization=False):
        self.tol = tol
        self.verbose = verbose
        self.max_iter = max_iter
        self.C = C
        self.random_start = random_initialization
        self.theta = None
        
    @abstractmethod
    def compute_loss_grad(self, theta,  X, Y):
        pass        
    @abstractmethod
    def _group_protected_attribute(self, Y, A):
        pass

    def fit(self,X,Y,A):
        n = np.size(Y)
        X = np.hstack((X,np.ones((n,1))))
        m = X.shape[1]
        
        self._group_protected_attribute(Y,A)

        if self.random_start:
            theta = np.random.random_sample(m) - .5 
        else:
            theta = np.zeros((m,))
        #f = lambda w : self.compute_loss_grad(w,X, Y)[0]
        #grad = lambda w : self.compute_loss_grad(w,X, Y)[1] 
        def callback(w):
            f, g = self.compute_loss_grad(w,X,Y)
            print("fun_value {:.4f} \t gnorm {:.4f}".format(f,np.linalg.norm(g)))
        #res = fmin_bfgs(f,theta, grad, gtol=self.tol, maxiter=self.max_iter,full_output=False,disp=True, retall=True, callback = callback)
        #res = minimize(self.compute_loss_grad, theta,args=(X, Y), method='L-BFGS-B',jac=True, tol=self.tol, options={'maxiter':self.max_iter, 'disp':False}, callback=callback)
        res = minimize(self.compute_loss_grad, theta,args=(X, Y), method='L-BFGS-B',jac=True, tol=self.tol, options={'maxiter':self.max_iter, 'disp':self.verbose})
         
        self.theta = res.x
        return self

    @abstractmethod
    def predict_proba_given_y(self,X,Y,A):
        pass
    @abstractmethod
    def predict_proba(self,X,A):
        pass

    def _fixed_point_proba(self,X,A):
        phat1, pcheck1 = self.predict_proba_given_y(X,np.ones_like(A),A)
        phat0, pcheck0 = self.predict_proba_given_y(X,np.zeros_like(A),A)
        prob = (phat1 * pcheck0 + phat0 * (1 - pcheck1)) / (1 - pcheck1 + pcheck0)
        return prob
         
    def _predict_proba_given_y(self, X, Y, _lambda, grp1, grp2, p1, p2):
        """
            Computes \hat{P}(\hat{Y}|X,A,Y) for two groups

        Parameters
        ----------
        X : ndarray, shape (n_samples,n_features + 1)
            Feature vector
        Y : ndarray, shape (n_samples,)
            Array of labels.
        _lambda : float 
            Fairness parameter for the given two groups
        grp1, grp2 : ndarray, shape(n_samples,)
            indices for group members
        p1, p2, : float
            empirical probability of each group from training data
        
        Returns
        -------
        phat: array, shape (n_samples,) 
            Prediction probability
        pcheck : array, shape (n_samples,)
            Adversarial estimation of the empirical distribution
        """
        phat = np.exp(_log_logistic(_dot_intercept(self.theta,X)))
        pcheck = np.copy(phat)
        
        if _lambda > 0:
            cond = (p1 / _lambda) < phat
            phat[grp1] = np.minimum( phat[grp1] , p1 /_lambda)
            pcheck[grp1] = np.where( cond[grp1], np.ones_like(pcheck[grp1]), phat[grp1] * (1 + (_lambda / p1) * (1 - phat[grp1])))
            
            cond = (1 - (p2 / _lambda)) > phat
            phat[grp2] = np.maximum( phat[grp2] , 1 - p2 /_lambda)
            pcheck[grp2] = np.where( cond[grp2], np.zeros_like(cond[grp2]), phat[grp2] * (1 - (_lambda / p2) * (1 - phat[grp2])))
        elif _lambda < 0:
            cond = (1 + (p1 / _lambda)) > phat
            phat[grp1] = np.maximum( phat[grp1] , 1 + p1 /_lambda)
            pcheck[grp1] = np.where( cond[grp1], np.zeros_like(cond[grp1]), phat[grp1] * (1 - (_lambda / p1) * (1 - phat[grp1])))
            
            cond = (- p2 / _lambda) < phat
            phat[grp2] = np.minimum( phat[grp2] , - p2 /_lambda)
            pcheck[grp2] = np.where( cond[grp2], np.ones_like(cond[grp2]), phat[grp2] * (1 + (_lambda / p2) * (1 - phat[grp2])))
        return phat, pcheck

    def predict(self,X,A):
        return np.round(self.predict_proba(X,A))

    @abstractmethod
    def fairness_violation(self,X,Y,A):
        pass
    def score(self,X,Y,A):
        return 1 - np.mean(abs(self.predict(X,A) - Y))
    def expected_error(self,X,Y,A):
        proba = self.predict_proba(X,A)
        return np.mean(np.where(Y == 1 , 1 - proba, proba))

class DP_fair_logloss_classifier(fair_logloss_classifier):
    def __init__(self, tol=1e-6, verbose=True, max_iter=10000, C = .1, random_initialization=False):
        super().__init__(tol = tol, verbose = verbose, max_iter = max_iter, C = C, random_initialization=random_initialization)

    def _group_protected_attribute(self, tr_Y, tr_A):
        self.grp1 = tr_A == 1
        self.grp2 = tr_A == 0

    def compute_loss_grad(self, theta, X, Y):
        p = np.exp(_log_logistic(_dot_intercept(theta,X)))
        idx1 = self.grp1 # A == 1
        idx0 = self.grp2 # A == 0
        n = X.shape[0]
        self._lambda = fairify(p[idx1], p[idx0]) / n
        loss, grad, trunc_idx = sum_truncated_loss_grad(theta, X, Y, idx1, idx0, self._lambda)
        loss_ow, grad_ow = sum_logistic_loss_grad(theta,X,Y, np.logical_not(trunc_idx))

        loss += loss_ow
        grad += grad_ow
        
        loss = loss / n + .5 * self.C * np.dot(theta, theta)
        grad = grad / n + self.C * theta
        return loss, grad
    
    def predict_proba_given_y(self,X,Y,A):
        grp1 = A == 1
        grp2 = A == 0
        p1, p2 = np.mean(self.grp1) , np.mean(self.grp2) # group empirical probability based on training data
        
        return self._predict_proba_given_y(X,Y, self._lambda, grp1, grp2, p1, p2)
                   
    def predict_proba(self,X,A):
        return self.predict_proba_given_y(X,np.empty_like(A),A)[0]

    def fairness_violation(self,X,Y,A):
        proba = self.predict_proba(X,A)
        return abs(np.mean(proba[A == 1]) - np.mean(proba[A == 0]))


class EOPP_fair_logloss_classifier(fair_logloss_classifier):
    def __init__(self, tol=1e-6, verbose=True, max_iter=10000, C = .1, random_initialization=False):
        super().__init__(tol = tol, verbose = verbose, max_iter = max_iter, C = C, random_initialization= random_initialization)

    def _group_protected_attribute(self, tr_Y, tr_A):
        self.grp1 = np.logical_and( tr_A == 1, tr_Y == 1)
        self.grp2 = np.logical_and( tr_A == 0, tr_Y == 1)

    def compute_loss_grad(self, theta, X, Y):
        p = np.exp(_log_logistic(_dot_intercept(theta,X)))
        idx1 = self.grp1 # np.logical_and(A == 1, Y == 1)
        idx0 = self.grp2 # np.logical_and(A == 0, Y == 1)
        n = X.shape[0]
        self._lambda = fairify(p[idx1], p[idx0]) / n 
        loss, grad, trunc_idx  = sum_truncated_loss_grad(theta, X, Y, idx1, idx0, self._lambda)
        loss_ow, grad_ow = sum_logistic_loss_grad(theta,X,Y, np.logical_not(trunc_idx))

        loss += loss_ow
        grad += grad_ow
        loss = loss / n + .5 * self.C * np.dot(theta, theta)
        grad = grad /n + self.C * theta
        return loss, grad

    def predict_proba_given_y(self,X,Y,A):
        grp1 = np.logical_and(A == 1, Y == 1)
        grp2 = np.logical_and(A == 0, Y == 1)
        p1, p2 = np.mean(self.grp1) , np.mean(self.grp2)
        return self._predict_proba_given_y(X,Y, self._lambda, grp1, grp2, p1, p2)
        
            
    def predict_proba(self,X,A):
        return self._fixed_point_proba(X,A)

    def fairness_violation(self,X,Y,A):
        proba = self.predict_proba(X,A)
        return abs(np.mean(proba[np.logical_and(Y == 1, A == 1)]) - np.mean(proba[np.logical_and(Y == 1, A == 0)]))  
    

class EODD_fair_logloss_classifier(fair_logloss_classifier):
    def __init__(self, tol=1e-6, verbose=True, max_iter=10000, C = .1, random_initialization=False):
        super().__init__(tol = tol, verbose = verbose, max_iter = max_iter, C = C, random_initialization=random_initialization)

    def _group_protected_attribute(self, tr_Y, tr_A):
        self.grp1 = np.logical_and( tr_A == 1, tr_Y == 1)
        self.grp2 = np.logical_and( tr_A == 0, tr_Y == 1)
        self.grp3 = np.logical_and( tr_A == 1, tr_Y == 0)
        self.grp4 = np.logical_and( tr_A == 0, tr_Y == 0)
         
    def compute_loss_grad(self, theta,X, Y):
        p = np.exp(_log_logistic(_dot_intercept(theta,X)))
        n = X.shape[0]
        idx11 = self.grp1 # np.logical_and(A == 1, Y == 1)
        idx01 = self.grp2 # np.logical_and(A == 0, Y == 1)
        self._lambda1 = fairify(p[idx11], p[idx01]) / n 

        loss1, grad1, trunc_idx1 = sum_truncated_loss_grad(theta, X, Y, idx11, idx01, self._lambda1)

        idx10 = self.grp3 # np.logical_and(A == 1, Y == 0)
        idx00 = self.grp4 # np.logical_and(A == 0, Y == 0)
        self._lambda0 = fairify(p[idx10], p[idx00]) / n 

        loss0, grad0, trunc_idx2 = sum_truncated_loss_grad(theta, X, Y, idx10, idx00, self._lambda0)
        
        loss_ow, grad_ow = sum_logistic_loss_grad(theta, X, Y, np.logical_not(np.logical_or(trunc_idx1, trunc_idx2)))

        loss = loss1 + loss0 + loss_ow
        grad = grad1 + grad0 + grad_ow
        loss = loss / n + .5 * self.C * np.dot(theta, theta)
        grad = grad /n + self.C * theta
        #pdb.set_trace()
        return loss, grad    
    
    def predict_proba_given_y(self,X,Y,A):
        grp1 = np.logical_and(A == 1, Y == 1)
        grp2 = np.logical_and(A == 0, Y == 1)
        p1, p2 = np.mean(self.grp1) , np.mean(self.grp2)
        phat, pcheck = self._predict_proba_given_y(X, Y, self._lambda1, grp1, grp2, p1, p2)
        
        grp3 = np.logical_and(A == 1, Y == 0)
        grp4 = np.logical_and(A == 0, Y == 0)
        p3, p4 = np.mean(self.grp3) , np.mean(self.grp4)
        phat_, pcheck_ = self._predict_proba_given_y(X, Y, self._lambda0, grp3, grp4, p3, p4)
        idx = np.logical_or(grp3, grp4)
        phat[idx], pcheck[idx] = phat_[idx], pcheck_[idx]

        return phat, pcheck
        

    def predict_proba(self,X,A):
        return self._fixed_point_proba(X,A)
        
    def fairness_violation(self,X,Y,A):
        proba = self.predict_proba(X,A)
        return  abs(np.mean(proba[np.logical_and(Y == 1, A == 1)]) - np.mean(proba[np.logical_and(Y == 1, A == 0)]))  \
            +   abs(np.mean(proba[np.logical_and(Y == 0, A == 1)]) - np.mean(proba[np.logical_and(Y == 0, A == 0)]))
   
     
            