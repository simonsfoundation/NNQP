# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 17:01:29 2016

@author: agiovann
"""
import numpy as  np
import nnqp
#%% EXAMPLE WITH RANDOM VARIABLES
p = 10

Q1 = np.random.randn(p,p)

Q = Q1.dot(Q1.T)
f = np.random.randn(p)

#%%
x = nnqp.nnqp(Q,f)
print x

#%% only if you need to compare with another solver
import cvxpy as cvx
cx = cvx.Variable(p) 
objective=cvx.Minimize(0.5*cvx.quad_form(cx, Q) + f.T*cx ) 
constraints= [cx >= 0]
prob = cvx.Problem(objective, constraints)
result = prob.solve(solver='SCS')
print(cx.value.T)

#%% compute the difference between outputs
print(np.linalg.norm(x-cx.value.T))/p
