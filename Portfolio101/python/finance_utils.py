#!/usr/bin/env python
import pandas as pd
import numpy as np
from scipy.stats import mstats
import copy

def interpolate_missing(x, mean_window=20, inplace=False):
    """
    """

    if not inplace:
        x = copy.deepcopy(x)

    for i in range(len(x)):
        if np.isnan(x[i]):
            low = int(i-mean_window) if i-mean_window>0 else 0
            avg = np.mean(x[low:i])
            x[i] = avg
            
    return x

def winsorize(x, winsor=5, inplace=False):
    """
    """

    if not inplace:
        x = copy.deepcopy(x) 

    low = np.percentile(x, winsor)
    hi = np.percentile(x, 100.-winsor)
    def select(xi, low, hi):
        if xi>=low and xi<=hi:
            return xi
        elif xi<low:
            return low
        else:
            return hi
    x = np.array( [select(xi,low,hi) for xi in x] )

    return x

def compute_signal(x, mean_window=20):
    """
    """    
    sgn = [0]
    for i in range(1,len(x)):
        low = int(i-mean_window) if i-mean_window>0 else 0
        avg = np.mean(x[low:i])
        sgn.append( (x[i]-avg)/avg )
    sgn = np.array( sgn )
    
    return sgn

def covariance_matrix(m):
    """
    """
    return np.cov(m, rowvar=False)

def markowitz_optimization(sgn, cov, opt_returns, verbose=True):
    """
    """
    
    n = sgn.shape[1]

    P = opt.matrix(cov)
    q = opt.matrix(np.zeros((n, 1)))
    G = opt.matrix(np.concatenate((-np.array(pbar), -np.identity(n)), 0))
    A = opt.matrix(1.0, (1,n))
    b = opt.matrix(1.0)

    opt.solvers.options['show_progress'] = verbose

    optimal_weights = [solvers.qp(P, q, G, opt.matrix(np.concatenate((-np.ones((1, 1)) * mu, np.zeros((n, 1))), 0)), A, b)['x'] for ret in opt_returns]
        
    optimal_sigmas = [np.sqrt(np.matrix(w).T * cov.T.dot(np.matrix(w)))[0,0] for w in optimal_weights]

    return optimal_weights, optimal_sigmas
