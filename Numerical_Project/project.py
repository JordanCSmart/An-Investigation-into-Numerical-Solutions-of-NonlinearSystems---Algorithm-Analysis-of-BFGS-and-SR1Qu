import numpy as np
import matplotlib.pyplot as plt
import math

def NewtonSystem(f, g, J, x0, y0, max, tol):
    X0 = np.array([[x0], [y0]])
    X1 = np.array([[0], [0]])
    for i in range(0, max):
        X1 = X0 - np.linalg.inv(J(X0[0][0], X0[1][0])) @ np.array([[f(X0[0][0], X0[1][0])], [g(X0[0][0], X0[1][0])]])
        if abs(X0[0][0] - X1[0][0])/abs(X1[0][0]) < tol and abs(X0[1][0] - X1[1][0])/abs(X1[1][0]) < tol:
            return X1
        print(abs(X0[1][0] - X1[1][0])/abs(X1[1][0]))
        X0 = X1
    raise ValueError(X1)

def NewtonSystem3d(f, g, h, J, x0, y0, z0, max, tol):
    X0 = np.array([[x0], [y0], [z0]])
    X1 = np.array([[0], [0], [0]])
    for i in range(0, max):
        print('iteration', i)
        X1 = X0 - np.linalg.inv(J(X0[0][0], X0[1][0], X0[2][0])) @ np.array([[f(X0[0][0], X0[1][0], X0[2][0])], [g(X0[0][0], X0[1][0], X0[2][0])], [h(X0[0][0], X0[1][0], X0[2][0])]])
        if abs(X0[0][0] - X1[0][0])/abs(X1[0][0]) < tol and abs(X0[1][0] - X1[1][0])/abs(X1[1][0]) < tol and abs(X0[2][0] - X1[2][0])/abs(X1[2][0]) < tol:
            return X1
        X0 = X1
    raise ValueError(X1)

if __name__ == "__main__":
    print(NewtonSystem(lambda x, y: x**2 + y**2 - 4, lambda x, y: math.e**x + y - 1, lambda x, y: np.array([[2*x, 2*y], [math.e**x, 1]]), 1, -1, 1000, 10e-20))