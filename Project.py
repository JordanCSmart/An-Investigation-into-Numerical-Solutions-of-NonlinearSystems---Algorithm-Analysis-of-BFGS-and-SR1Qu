#!/usr/bin/env python
# coding: utf-8

import numpy as np
import math as math
import matplotlib.pyplot as plt
from scipy import special

from mpl_toolkits import mplot3d

def line_search(f,grad, X0, alpha0, roe, pk, c):
    alpha = alpha0
    Grad = np.transpose(grad(f,X0))
    while f(X0- alpha*pk) <= f(X0) + c*alpha*Grad@pk:
        alpha = alpha * roe
                                                      
    return alpha

print(line_search(f,grad,X0,1,0.3,pk,0.3))

def NewtonSystem(f, g, h, J, x0, y0, z0, max, tol):
    X0 = np.array([[x0], [y0],[z0]])
    X1 = np.array([[0], [0],[0]])
    for i in range(0, max):
        X1 = X0 - np.linalg.inv(J(X0[0][0], X0[1][0],X0[2][0])) @ np.array([[f(X0[0][0], X0[1][0],X0[2][0])], [g(X0[0][0], X0[1][0],X0[2][0])],[h(X0[0][0], X0[1][0],X0[2][0])]])
        if abs(X0[0][0] - X1[0][0])/abs(X1[0][0]) < tol and abs(X0[1][0] - X1[1][0])/abs(X1[1][0]) < tol and abs(X0[2][0] - X1[2][0])/abs(X1[2][0]) < tol:
            print(i)
            return X1
        X0 = X1
    raise ValueError('not enought iterations')

def BFGS(f, g, h, J, x0, y0, z0, max, tol):
    X0 = np.array([x0, y0, z0])
    X1 = np.array([0, 0,0])
    grad = np.gradient([f(x0, y0, z0), g(x0, y0, z0), h(x0, y0, z0)])
    B0 = np.transpose(J(grad[0], grad[1], grad[2]))
    for i in range(0, max):
        grad = np.gradient([f(X0[0], X0[1], X0[2]), g(X0[0], X0[1], X0[2]), h(X0[0], X0[1], X0[2])])
        pk = np.linalg.inv(B0) @ -grad
        alpha = 10000
        for j in range(1, 101):
            if alpha > np.argmin(f(X0[0] + j*pk[0], X0[1] + j*pk[1], X0[2] + j*pk[2])):
                alpha = np.argmin(f(X0[0] + j*pk[0], X0[1] + j*pk[1], X0[2] + j*pk[2]))
        sk = alpha*pk
        X1 = X0 + sk
        yk = np.gradient([f(X1[0], X1[1], X1[2]), g(X1[0], X1[1], X1[2]), h(X1[0], X1[1], X0[2])]) - grad
        X0 = X1
        B0 = B0 + ((yk-B0@sk)@np.transpose((yk-B0@sk)))/(np.transpose(sk)@(yk-B0))
    return X0

print('newtons:')
answer = NewtonSystem(lambda x, y, z: x - np.log(10-y*x), lambda x, y, z: y - np.log(10-z*y), lambda x, y, z: z - np.log(10-z*x), lambda x, y, z: np.array([[1-(1/(1-y*x))*(y),0,-(1/(1-z*x))*z], [-(1/(1-y*x))*x,1-(1/(1-z*y))*x,0],[0,-(1/(1-z*y))*y,1-(1/(1-z*x))*x]]), 3,3,3, 100000, 1e-3)
print('x =', answer[0][0])
print('y =', answer[1][0])
print('z =', answer[2][0])

# ax = plt.axes(projection = '3d')

# dim = np.linspace(0,3,1000)

# xline = lambda x, y, z: x - np.log(10-y*x)(dim,dim,dim)
# yline = lambda x, y, z: x - np.log(10-y*x)(dim,dim,dim)
# zline = lambda x, y, z: x - np.log(10-y*x)(dim,dim,dim)
# ax.scatter3D(answer[0][0],answer[1][0],answer[2][0])
# ax.plot3D(xline,yline,zline,'grey')

print(BFGS(lambda x, y, z: x - np.log(10-y*x), lambda x, y, z: y - np.log(10-z*y), lambda x, y, z: z - np.log(10-z*x), lambda x, y, z: np.array([[1-(1/(1-y*x))*(y),0,-(1/(1-z*x))*z], [-(1/(1-y*x))*x,1-(1/(1-z*y))*x,0],[0,-(1/(1-z*y))*y,1-(1/(1-z*x))*x]]), 3, 3, 3, 100, 1e-3))
