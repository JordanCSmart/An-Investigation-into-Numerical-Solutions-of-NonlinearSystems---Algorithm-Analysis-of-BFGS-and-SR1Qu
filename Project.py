#!/usr/bin/env python
# coding: utf-8

import numpy as np
import math as math
import matplotlib.pyplot as plt
from numpy.core.shape_base import atleast_1d
from scipy import optimize
import numdifftools as nd
import warnings
import time

from mpl_toolkits import mplot3d

warnings.filterwarnings('ignore')

def NewtonSystems(f, B, gradient, X0, max, tol):
    X1 = 0
    for i in range(0, max):
        pk = np.linalg.inv(B(X0)) @ -gradient(X0)
        alpha = optimize.line_search(f, gradient, X0, pk)
        if type(alpha[0]) != type(None): alpha = alpha[0]*.3
        else: alpha = alpha[4]*.3
        sk = alpha*pk
        X1 = X0 + sk
        for j in range(0, len(X0)):
            if abs(X0[j]-X1[j]) > tol: break
            elif j == len(X0)-1: return X1, i+1
        X0 = X1
    return 'not enough iterations'

def BFGS(f, B, gradient, X0, max, tol):
    X1 = 0
    B0 = np.linalg.inv(B(X0))
    for i in range(0, max):
        pk = B0 @ -gradient(X0)
        alpha = optimize.line_search(f, gradient, X0, pk)
        if type(alpha[0]) != type(None): alpha = alpha[0]*.3
        else: alpha = 1*.3
        sk = alpha*pk
        X1 = X0 + sk
        yk = gradient(X1)-gradient(X0)
        for j in range(0, len(X0)):
            if abs(X0[j]-X1[j]) > tol: break
            elif j == len(X0)-1: return X1, i+1
        X0 = X1
        skt = np.array([sk])
        ykt = np.array([yk])
        sk = np.transpose(skt)
        yk = np.transpose(ykt)
        B0 = B0 + ((skt@yk+ykt@B0@yk)*(sk@skt)/(skt@yk)**2) - ((B0@yk@skt+sk@ykt@B0))/(skt@yk)
    return 'not enough iterations'

def SR1(f, B, gradient, X0, max, tol):
    X1 = 0
    B0 = np.linalg.inv(B(X0))
    for i in range(0, max):
        pk = B0 @ -gradient(X0)
        alpha = optimize.line_search(f, gradient, X0, pk)
        if type(alpha[0]) != type(None): alpha = alpha[0]*.3
        else: alpha = 1*.3
        sk = alpha*pk
        X1 = X0 + sk
        yk = gradient(X1)-gradient(X0)
        for j in range(0, len(X0)):
            if abs(X0[j]-X1[j]) > tol: break
            elif j == len(X0)-1: return X1, i+1
        X0 = X1
        skt = np.array([sk])
        ykt = np.array([yk])
        sk = np.transpose(skt)
        yk = np.transpose(ykt)
        B0 = B0 + ((sk-B0@yk)@np.transpose(sk-B0@yk))/(np.transpose(sk-B0@yk)@yk)
    return 'not enough iterations'

# ax = plt.axes(projection = '3d')

# dim = np.linspace(0,3,1000)

# xline = lambda x, y, z: x - np.log(10-y*x)(dim,dim,dim)
# yline = lambda x, y, z: x - np.log(10-y*x)(dim,dim,dim)
# zline = lambda x, y, z: x - np.log(10-y*x)(dim,dim,dim)
# ax.scatter3D(answer[0][0],answer[1][0],answer[2][0])
# ax.plot3D(xline,yline,zline,'grey')
it1 = 0
it2 = 0
it3 = 0
hr1 = 0
hr2 = 0
hr3 = 0

# def func(x):
#     return (1-x[0])**2+100*(x[1]-x[0]**2)**2

# def H(x):
#     return np.array([[-400*(x[1]-x[0]**2)+800*x[0]**2+2, -400*x[0]], [-400*x[0], 200]])

# def gradient1(x):
#     return np.array([-2*(1-x[0])**2-400*x[0]*(x[1]-x[0]**2), 200*(x[1]-x[0]**2)])

# s = time.time()
# a1 = NewtonSystems(func, H, gradient1, [-2, 20], 100000, 1e-4)
# e = time.time()
# hr1 = e-s
# s = time.time()
# a2 = BFGS(func, H, gradient1, [-2, 20], 100000, 1e-4)
# e = time.time()
# hr2 = e-s
# s = time.time()
# a3 = SR1(func, H, gradient1, [-2, 20], 100000, 1e-4)
# e = time.time()
# hr3 = e-s
# it1 = a1[1]
# it2 = a2[1]
# it3 = a3[1]
# print("Newtons Method", "number of iterations:", a1[1], ", time taken:", hr1, "secends")
# print("BFGS", "number of iterations:", a2[1], ", time taken:", hr2, "secends")
# print("SR1", "number of iterations:", a3[1], ", time taken:", hr3, "secends")
# print("Newton values", a1[0])
# print("BFGS values", a2[0])
# print("SR1 values", a3[0])

def func2(x):
    y = 0
    for i in range(0, len(x)):
        y = y + x[i]**2
    return y

def H2(x):
    return 2*np.identity(len(x))

def gradient2(x):
    y = []
    for i in range(0, len(x)):
        y.append(2*x[i])
    return np.array(y)

input = []
for j in range(0, 5):
   input.append(j+1)

iterationsnewtons = []
timearraynewtons = []
iterationbfgs = []
timearraybfgs = []
iterationssr1 = []
timearraysr1 = []

ts = time.time()
a = NewtonSystems(func2, H2, gradient2, input, 10000000000, 1e-4)
te = time.time()
iterationsnewtons.append(5)
timearraynewtons.append(te-ts)
ts = time.time()
a = BFGS(func2, H2, gradient2, input, 10000000000, 1e-4)
te = time.time()
iterationbfgs.append(5)
timearraybfgs.append(te-ts)
ts = time.time()
a = SR1(func2, H2, gradient2, input, 10000000000, 1e-4)
te = time.time()
iterationssr1.append(5)
timearraysr1.append(te-ts)

for j in range(0, 10):
   input.append(j+1)

ts = time.time()
a = NewtonSystems(func2, H2, gradient2, input, 10000000000, 1e-4)
te = time.time()
iterationsnewtons.append(10)
timearraynewtons.append(te-ts)
ts = time.time()
a = BFGS(func2, H2, gradient2, input, 10000000000, 1e-4)
te = time.time()
iterationbfgs.append(10)
timearraybfgs.append(te-ts)
ts = time.time()
a = SR1(func2, H2, gradient2, input, 10000000000, 1e-4)
te = time.time()
iterationssr1.append(10)
timearraysr1.append(te-ts)

for j in range(0, 50):
   input.append(j+1)

ts = time.time()
a = NewtonSystems(func2, H2, gradient2, input, 10000000000, 1e-4)
te = time.time()
iterationsnewtons.append(50)
timearraynewtons.append(te-ts)
ts = time.time()
a = BFGS(func2, H2, gradient2, input, 10000000000, 1e-4)
te = time.time()
iterationbfgs.append(50)
timearraybfgs.append(te-ts)
ts = time.time()
a = SR1(func2, H2, gradient2, input, 10000000000, 1e-4)
te = time.time()
iterationssr1.append(50)
timearraysr1.append(te-ts)

for j in range(0, 100):
   input.append(j+1)

ts = time.time()
a = NewtonSystems(func2, H2, gradient2, input, 10000000000, 1e-4)
te = time.time()
iterationsnewtons.append(100)
timearraynewtons.append(te-ts)
ts = time.time()
a = BFGS(func2, H2, gradient2, input, 10000000000, 1e-4)
te = time.time()
iterationbfgs.append(100)
timearraybfgs.append(te-ts)
ts = time.time()
a = SR1(func2, H2, gradient2, input, 10000000000, 1e-4)
te = time.time()
iterationssr1.append(100)
timearraysr1.append(te-ts)

for j in range(0, 500):
   input.append(j+1)

ts = time.time()
a = NewtonSystems(func2, H2, gradient2, input, 10000000000, 1e-4)
te = time.time()
iterationsnewtons.append(500)
timearraynewtons.append(te-ts)
ts = time.time()
a = BFGS(func2, H2, gradient2, input, 10000000000, 1e-4)
te = time.time()
iterationbfgs.append(500)
timearraybfgs.append(te-ts)
ts = time.time()
a = SR1(func2, H2, gradient2, input, 10000000000, 1e-4)
te = time.time()
iterationssr1.append(500)
timearraysr1.append(te-ts)

for j in range(0, 1000):
   input.append(j+1)

ts = time.time()
a = NewtonSystems(func2, H2, gradient2, input, 10000000000, 1e-4)
te = time.time()
iterationsnewtons.append(a[1])
timearraynewtons.append(te-ts)
ts = time.time()
a = BFGS(func2, H2, gradient2, input, 10000000000, 1e-4)
te = time.time()
iterationbfgs.append(a[1])
timearraybfgs.append(te-ts)
ts = time.time()
a = SR1(func2, H2, gradient2, input, 10000000000, 1e-4)
te = time.time()
iterationssr1.append(a[1])
timearraysr1.append(te-ts)

for j in range(0, 5000):
   input.append(j+1)

ts = time.time()
a = NewtonSystems(func2, H2, gradient2, input, 10000000000, 1e-4)
te = time.time()
iterationsnewtons.append(a[1])
timearraynewtons.append(te-ts)
ts = time.time()
a = BFGS(func2, H2, gradient2, input, 10000000000, 1e-4)
te = time.time()
iterationbfgs.append(a[1])
timearraybfgs.append(te-ts)
ts = time.time()
a = SR1(func2, H2, gradient2, input, 10000000000, 1e-4)
te = time.time()
iterationssr1.append(a[1])
timearraysr1.append(te-ts)

for j in range(0, 10000):
   input.append(j+1)

ts = time.time()
a = NewtonSystems(func2, H2, gradient2, input, 10000000000, 1e-4)
te = time.time()
iterationsnewtons.append(a[1])
timearraynewtons.append(te-ts)
ts = time.time()
a = BFGS(func2, H2, gradient2, input, 10000000000, 1e-4)
te = time.time()
iterationbfgs.append(a[1])
timearraybfgs.append(te-ts)
ts = time.time()
a = SR1(func2, H2, gradient2, input, 10000000000, 1e-4)
te = time.time()
iterationssr1.append(a[1])
timearraysr1.append(te-ts)

for j in range(0, 50000):
   input.append(j+1)

ts = time.time()
a = NewtonSystems(func2, H2, gradient2, input, 10000000000, 1e-4)
te = time.time()
iterationsnewtons.append(a[1])
timearraynewtons.append(te-ts)
ts = time.time()
a = BFGS(func2, H2, gradient2, input, 10000000000, 1e-4)
te = time.time()
iterationbfgs.append(a[1])
timearraybfgs.append(te-ts)
ts = time.time()
a = SR1(func2, H2, gradient2, input, 10000000000, 1e-4)
te = time.time()
iterationssr1.append(a[1])
timearraysr1.append(te-ts)

plt.loglog(iterationsnewtons, timearraynewtons, label='newtons')
plt.loglog(iterationbfgs, timearraybfgs, label='bfgs')
plt.loglog(iterationssr1, timearraysr1, label='sr1')
plt.xlabel('size of n')
plt.ylabel('time in seconds')
plt.legend()
plt.show()

# ts1 = time.time()
# a1 = NewtonSystems(func2, H2, gradient2, input, 10000000000, 1e-4)
# te1 = time.time()
# ts2 = time.time()
# a2 = BFGS(func2, H2, gradient2, input, 10000000000, 1e-4)
# te2 = time.time()
# ts3 = time.time()
# a3 = SR1(func2, H2, gradient2, input, 10000000000, 1e-4)
# te3 = time.time()
# it1 = it1 + a1[1]
# it2 = it2 + a2[1]
# it3 = it3 + a3[1]
# hr1 = hr1 + (te1 - ts1)
# print("Newtons Method", "number of iterations:", a1[1], ", time taken:", hr1, "secends")
# hr2 = hr2 + (te2 - ts2)
# print("BFGS", "number of iterations:", a2[1], ", time taken:", hr2, "secends")
# hr3 = hr3 + (te3 - ts3)
# print("SR1", "number of iterations:", a3[1], ", time taken:", hr3, "secends")

# def func3(x):
#     return x[0]**2 + (x[1]-5)**2 + x[2]**2 + np.sin(x[0])**2

# def H3(x):
#     return np.array([[2-np.sin(x[0]),0,0],[0, 2, 0],[0, 0, 2]])

# def gradient3(x):
#     x1 = x[0]
#     x2 = x[1]
#     x3 = x[2]
    
#     return np.array([2*x1 + 2*np.sin(x1)*np.cos(x1), 2*(x2-5), 2*x3])

# ts1 = time.time()
# a1 = NewtonSystems(func3, H3, gradient3, [34,32,23], 10000000000, 1e-4)
# te1 = time.time()
# ts2 = time.time()
# a2 = BFGS(func3, H3, gradient3, [34,32,23], 10000000000, 1e-4)
# te2 = time.time()
# ts3 = time.time()
# a3 = SR1(func3, H3, gradient3, [34,32,23], 10000000000, 1e-4)
# te3 = time.time()
# it1 = it1 + a1[1]
# it2 = it2 + a2[1]
# it3 = it3 + a3[1]
# hr1 = hr1 + (te1 - ts1)
# print("Newtons Method", "number of iterations:", a1[1], ", time taken:", hr1, "secends")
# hr2 = hr2 + (te2 - ts2)
# print("BFGS", "number of iterations:", a2[1], ", time taken:", hr2, "secends")
# hr3 = hr3 + (te3 - ts3)
# print("SR1", "number of iterations:", a3[1], ", time taken:", hr3, "secends")
# print("Newton values", a1[0])
# print("BFGS values", a2[0])
# print("SR1 values", a3[0])

# def func4(x):
#     x1 = x[0]
#     x2 = x[1]
    
#     return (4 - x1**2 - 2*x2**2)**2

# def H4(x):
#     return np.array([[8*x[0]**2-4*(-x[0]**2-2*x[1]**2+4), 16*x[0]*x[1]],[16*x[0]*x[1], 32*x[1]**2-8*(-x[0]**2-2*x[1]**2+4)]])

# def gradient4(x):
#     x1 = x[0]
#     x2 = x[1]
    
#     return np.array([-4*x1*(-x1**2 - 2*(x2**2) + 4), -8*x2*(-x1**2 -2*(x2**2) + 4)])

# ts1 = time.time()
# a1 = NewtonSystems(func4, H4, gradient4, [34,12], 10000000000, 1e-4)
# te1 = time.time()
# ts2 = time.time()
# a2 = BFGS(func4, H4, gradient4, [34,12], 10000000000, 1e-4)
# te2 = time.time()
# ts3 = time.time()
# a3 = SR1(func4, H4, gradient4, [34,12], 10000000000, 1e-4)
# te3 = time.time()
# it1 = it1 + a1[1]
# it2 = it2 + a2[1]
# it3 = it3 + a3[1]
# hr1 = hr1 + (te1 - ts1)
# print("Newtons Method", "number of iterations:", a1[1], ", time taken:", hr1, "secends")
# hr2 = hr2 + (te2 - ts2)
# print("BFGS", "number of iterations:", a2[1], ", time taken:", hr2, "secends")
# hr3 = hr3 + (te3 - ts3)
# print("SR1", "number of iterations:", a3[1], ", time taken:", hr3, "secends")
# print("Newton values", a1[0])
# print("BFGS values", a2[0])
# print("SR1 values", a3[0])

# print("\nNewtons Method", it1, "iterations,", hr1, "seconds",  hr1/it1, "seconds per iteration")
# print("BFGS", it2, "iterations,", hr2, "seconds", hr2/it2, "seconds per iteration")
# print("SR1", it3, "iterations,", hr3, "seconds", hr3/it3, "seconds per iteration")
# print("\nBFGS is", hr1/hr2, "times faster than newtons and", hr1/it1/(hr2/it2), "per iteration faster")
