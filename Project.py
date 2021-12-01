#!/usr/bin/env python
# coding: utf-8

import numpy as np
import math as math
import matplotlib.pyplot as plt
from scipy import optimize
import numdifftools as nd

from mpl_toolkits import mplot3d

def line_search(f, grad, X0, pk):
    alpha = 100
    while f(X0+ alpha*pk) <= f(X0) + 0.1*alpha*grad@pk:
        alpha = alpha * 0.1
                                                     
    return alpha

def NewtonSystems(f, B, X0, max, tol):
    X1 = 0
    for i in range(0, max):
        grad = nd.Gradient(f)(X0)
        pk = np.linalg.inv(B(X0)) @ -grad
        alpha = line_search(f, grad, X0, pk)
        print(alpha)
        sk = alpha*pk
        X1 = X0 + sk
        for j in range(0, len(X0)):
            if abs(X0[j]-X1[j]) > tol: break
            elif j == len(X0)-1: return X0
        X0 = X1
    return X0

def BFGS(f, B, X0, max, tol):
    X1 = 0
    B0 = B(X0)
    for i in range(0, max):
        grad = nd.Gradient(f)(X0)
        pk = np.linalg.inv(B0) @ -grad
        alpha = line_search(f, grad, X0, pk)
        sk = alpha*pk
        X1 = X0 + sk
        yk = nd.Gradient(f)(X1) - grad
        for j in range(0, len(X0)):
            if abs(X0[j]-X1[j]) > tol: break
            elif j == len(X0)-1: return X0
        X0 = X1
        B0 = B0 - ((yk-B0@sk)@np.transpose(yk-B0@sk))/(np.transpose(sk)@(yk-B0@sk))
    return 'not enough iterations'

def SR1(f, B, X0, max, tol):
    X1 = 0
    B0 = B(X0)
    H0 = np.linalg.inv(B0)
    for i in range(0, max):
        grad = nd.Gradient(f)(X0)
        pk = -H0 @ grad
        alpha = line_search(f, grad, X0, pk)
        sk = alpha*pk
        X1 = X0 + sk
        yk = nd.Gradient(f)(X1) - grad
        for j in range(0, len(X0)):
            if abs(X0[j]-X1[j]) > tol: break
            elif j == len(X0)-1: return X0
        X0 = X1
        H0 = H0 +((sk-H0@yk)@np.transpose(sk-H0@yk))/(np.transpose(yk)@(sk-H0@yk))
    return 'not enough iterations'

# ax = plt.axes(projection = '3d')

# dim = np.linspace(0,3,1000)

# xline = lambda x, y, z: x - np.log(10-y*x)(dim,dim,dim)
# yline = lambda x, y, z: x - np.log(10-y*x)(dim,dim,dim)
# zline = lambda x, y, z: x - np.log(10-y*x)(dim,dim,dim)
# ax.scatter3D(answer[0][0],answer[1][0],answer[2][0])
# ax.plot3D(xline,yline,zline,'grey')

def func(x):
    return (1-x[0])**2+100*(x[1]-x[0]**2)**2

def H(x):
    return np.array([[-400*(x[1]-x[0]**2)+800*x[0]**2+2, -400*x[0]], [-400*x[0], 200]])

print(NewtonSystems(func, H, [10, 10], 100, 1e-3))
print(BFGS(func, H, [10, 10], 100, 1e-3))
print(SR1(func, H, [10, 10], 100, 1e-3))
