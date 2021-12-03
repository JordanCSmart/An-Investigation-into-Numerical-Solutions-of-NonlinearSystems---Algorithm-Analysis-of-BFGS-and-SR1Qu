#!/usr/bin/env python
# coding: utf-8

import numpy as np
import math as math
import matplotlib.pyplot as plt
from scipy import optimize
import numdifftools as nd
import warnings

from mpl_toolkits import mplot3d

warnings.filterwarnings('ignore')

def line_search(f, X0, pk, max):
    alpha = 10
    it = 0
    while f(X0+ alpha*pk) <= (f(X0) + alpha*np.transpose(nd.Gradient(f)(X0))@pk) :
        alpha = alpha * 0.1
        it = it +1
        if it == max or alpha == 0:
            break
                                           
    return alpha

def gradient(x):
    return np.array([-2*(1-x[0])**2-40*x[0]*(x[1]-x[0]**2), 20*(x[1]-x[0]**2)])

def NewtonSystems(f, B, X0, max, tol):
    X1 = 0
    for i in range(0, max):
        grad = nd.Gradient(f)(X0)
        pk = np.linalg.inv(B(X0)) @ -grad
        alpha = optimize.line_search(f, gradient, X0, pk)
        if type(alpha[0]) != type(None): alpha = alpha[0]
        else: alpha = alpha[4]
        sk = alpha*pk
        X1 = X0 + sk
        for j in range(0, len(X0)):
            if abs(X0[j]-X1[j]) > tol: break
            elif j == len(X0)-1: return X1
        X0 = X1
    return X0

def BFGS(f, B, X0, max, tol):
    X1 = 0
    B0 = np.linalg.inv(B(X0))
    for i in range(0, max):
        grad = nd.Gradient(f)(X0)
        pk = B0 @ -grad
        alpha = optimize.line_search(f, gradient, X0, pk)
        if type(alpha[0]) != type(None): alpha = alpha[0]
        else: alpha = alpha[4]
        sk = alpha*pk
        X1 = X0 + sk
        yk = nd.Gradient(f)(X1) - grad
        for j in range(0, len(X0)):
            if abs(X0[j]-X1[j]) > tol: break
            elif j == len(X0)-1: return X1
        X0 = X1
        pk = np.transpose(np.array([pk]))
        sk = np.transpose(np.array([sk]))
        yk = np.transpose(np.array([yk]))
        B0 = B0 + ((np.transpose(sk)@yk+np.transpose(yk)@B0@yk)*(sk@np.transpose(sk))/(np.transpose(sk)@yk)**2) - ((B0@yk@np.transpose(sk)+sk@np.transpose(yk)@B0))/(np.transpose(sk)@yk)
    return 'not enough iterations'

def SR1(f, B, X0, max, tol):
    X1 = 0
    B0 = B(X0)
    H0 = np.linalg.inv(B0)
    for i in range(0, max):
        grad = nd.Gradient(f)(X0)
        pk = -H0 @ grad
        alpha = line_search(f, X0, pk, max)
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

print(NewtonSystems(func, H, [-2, 3], 100000, 1e-4))
print(BFGS(func, H, [-2, 3], 100000, 1e-4))
print(SR1(func, H, [-2, 3], 100000, 1e-4))
