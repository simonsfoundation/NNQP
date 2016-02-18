NNQP
====

Efficient implementation of nonnegative quadratic programming solver in numba 
The solver implements a coordinate descent algorithm to find a solution to the problem

```
1/2 X^T Q X + f' X

subject to

X>=0
```


**Authors:** 
Andrea Giovannucci and Cengiz Pehlevan

Installation
============

Requirements
------------
- numpy
- scipy
- numba (Anaconda distribution suggested) 
- cvxpy (only if you want to compare with another solver)

Example
========

```python

import numpy as  np
import nnqp
#%% EXAMPLE WITH RANDOM VARIABLES
p = 10
Q1 = np.random.randn(p,p)
Q = Q1.dot(Q1.T)
f = np.random.randn(p)

#%% solve with NNQP
x = nnqp.nnqp(Q,f)
print x

#%% run only if you want to compare with another solver (requires cvxpy)
import cvxpy as cvx
cx = cvx.Variable(p) 
objective=cvx.Minimize(0.5*cvx.quad_form(cx, Q) + f.T*cx ) 
constraints= [cx >= 0]
prob = cvx.Problem(objective, constraints)
result = prob.solve(solver='SCS')
print(cx.value.T)

#%% compute the difference between outputs
print(np.linalg.norm(x-cx.value.T))/p
```
