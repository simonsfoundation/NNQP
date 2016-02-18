# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 13:55:03 2016

@author: agiovann
"""
import numpy as np
from scipy.optimize import nnls
import cvxpy as cvx
import numba
#%%
@numba.jit("f8[:](f8[:,:],f8[:])")
def nnqp(Q, f):
    tol=1e-3
    p = len(f)
    er = 1
    count = 1

    x = np.random.random(p)
    
    qdg=np.diag(Q)
    D = np.diag(qdg)
    M = Q-D
    Dinv = 1/qdg
    
    zero_count = 0;
    
    while er > tol:
        xprev = x.copy()

        for i in range(p):
            dum = Dinv[i]*(-M[i,:].dot(x)-f[i])
            x[i] = np.maximum(dum,0)

        ind = np.nonzero(xprev>0)[0]
        
        if len(ind)==0:
            if zero_count == 0:
                er = 1;
                zero_count = zero_count+1;
            else:
                er = 0            
        else:
#            er = np.max(np.abs(x[ind]-xprev[ind])/np.abs(xprev[ind]))
             er=np.max(compute_err(x[ind],xprev[ind]))
            
        if count > 10000:
            break
            print 'No Convergence'        
        
        count = count + 1;
    
    return x
#%%    
@numba.vectorize('float64(float64, float64)')
def compute_err(x, xprev):
    return np.abs(x-xprev)/np.abs(xprev)
    

  