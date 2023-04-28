import numpy as np
from scipy.optimize import rosen

"""
This function contains full information and implementation of 
functions in Table 1, Table 2 and Table 3 in the paper
"""

"""
lb: lower bound: lb = [lb_1, lb_2, ..., lb_d]
ub: lower bound: ub = [ub_1, ub_2, ..., ub_d]
dim: the number of variables (dimensions of the problems)
"""


def Ufun(x, a, k, m):
    return k * ((x - a) ** m) * (x > a) + k * ((-x - a) ** m) * (x < (-a))


# Function 1-7: Uni-modal test functions

def F1(x):
    return np.sum(np.square(x))


def F2(x):
    abs_x = np.abs(x)
    return np.sum(abs_x) + np.prod(abs_x)


def F3(x):
    dim = np.size(x, 0)
    result = 0
    y = x[::-1]
    for d in range(dim):
        result += (dim + 1) * y[d]

    return result


def F4(x):
    return np.max(np.abs(x))


def F5(x):
    # Rosenbrock
    return rosen(x)


def F6(x):
    return np.sum(np.square(x + 0.5))


def F7(x):
    dim = np.size(x, 0)
    return np.sum(np.arange(1, dim + 1) * (x ** 4)) + np.random.rand()


# Function 8-13: Multi-modal test functions
def F8(x):
    return np.sum(-x * np.sin(np.sqrt(np.abs(x))))


def F9(x):
    # Rastrigin
    return 10.0 * np.size(x, 0) + np.sum(np.square(x) - 10 * np.cos(2 * np.pi * x))


def F10(x):
    return -20 * np.exp(-0.2 * np.sqrt(np.mean(np.square(x)))) - np.exp(
        np.mean(np.cos(2 * np.pi * x))) + 20 + np.exp(1)


def F11(x):
    # Griewank
    return 1.0 + np.sum(np.square(x)) / 4000.0 - np.prod(np.cos(x / np.sqrt(np.arange(1, np.size(x, 0) + 1))))


def F12(x):
    return (np.pi / np.size(x, 0)) * 10 * np.sin(np.pi * (1 + (x[0] + 1) / 4)) + np.sum(
        (np.square((x[:-1] + 1) / 4)) * (1 + 10 * np.square(np.sin(np.pi * ((x[:-1] + 1) / 4 + 1) + 1)) + np.sum(Ufun(x, 10, 100, 4))))


def F13(x):
    return 0.1 * (np.square(np.sin(3 * np.pi * x[0])) + np.sum(
        np.square(x - 1) * (1 + np.square(np.sin(3 * np.pi * x + 1)))) + (np.square(x[-1] - 1)) + np.square(
        np.sin(2 * np.pi * x[-1]))) + np.sum(Ufun(x, 5, 100, 4))


def F14(x):
    aS = np.array(
        [[-32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32],
         [-32, -32, -32, -32, -32, -16, -16, -16, -16, -16, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32]])
    # aS.shape = (2, 25)
    bS = np.random.rand(25)
    for i in range(25):
        bS[i] = np.sum((x.transpose() - aS[:, i]) ** 6)
    return (1 / 500 + np.sum(1 / (np.arange(1, 26) + bS))) ** (-1)


def F15(x):
    aK = np.array([0.1957, 0.1947, 0.1735, 0.16, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246])
    bK = np.reciprocal(np.array([0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16]))
    return np.sum(np.square(aK - ((x[0] * (np.square(bK) + x[1] * bK)) / (np.square(bK) + x[2] * bK + x[3]))))


def F16(x):
    return 4 * (x[0] ** 2) - 2.1 * (x[0] ** 4) + (x[0] ** 6) / 3 + x[0] * x[1] - 4 * (x[1] ** 2) + 4 * (x[1] ** 4)


def F17(x):
    return (x[1] - (x[0] ** 2) * 5.1 / (4 * (np.pi ** 2)) + 5 / np.pi * x[0] - 6) ** 2 + 10 * (
            1 - 1 / (8 * np.pi)) * np.cos(x[0]) + 10


def F18(x):
    return (1 + ((x[0] + x[1] + 1) ** 2) * (
            19 - 14 * x[0] + 3 * (x[0] ** 2) - 14 * x[1] + 6 * x[0] * x[1] + 3 * (x[1] ** 2))) * (
                   30 + ((2 * x[0] - 3 * x[1]) ** 2) * (
                   18 - 32 * x[0] + 12 * (x[0] ** 2) + 48 * x[1] - 36 * x[0] * x[1] + 27 * (x[1] ** 2)))


def F19(x):
    aH = np.array([[3, 10, 30],
                   [0.1, 10, 35],
                   [3, 10, 30],
                   [0.1, 10, 35]])
    cH = np.array([[1], [1.2], [3], [3.2]])
    pH = np.array([[0.3689, 0.1170, 0.2673],
                   [0.4699, 0.4387, 0.7470],
                   [0.1091, 0.8732, 0.5547],
                   [0.0382, 0.5743, 0.8828]])
    o = 0
    for i in range(4):
        o = o - cH[i] * np.exp(-(np.sum(aH[i] * (np.square(x - pH[i])))))
    return o


def F20(x):
    aH = np.array([[10, 3, 17, 3.5, 1.7, 8],
                   [0.05, 10, 17, 0.1, 8, 14],
                   [3, 3.5, 1.7, 10, 17, 8],
                   [17, 8, 0.05, 10, 0.1, 14]])
    cH = np.array([[1], [1.2], [3], [3.2]])
    pH = np.array([[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                   [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                   [0.2348, 0.1415, 0.3522, 0.2883, 0.3047, 0.6650],
                   [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]])
    o = 0
    for i in range(4):
        o = o - cH[i] * np.exp(-(np.sum(aH[i] * ((x - pH[i]) ** 2))))
    return o


def F21(x):
    aSH = np.matrix([[4, 4, 4, 4],
                     [1, 1, 1, 1],
                     [8, 8, 8, 8],
                     [6, 6, 6, 6],
                     [3, 7, 3, 7],
                     [2, 9, 2, 9],
                     [5, 5, 3, 3],
                     [8, 1, 8, 1],
                     [6, 2, 6, 2],
                     [7, 3.6, 7, 3.6]])
    cSH = np.matrix([[0.1], [0.2], [0.2], [0.4], [0.4], [0.6], [0.3], [0.7], [0.5], [0.5]])
    o = 0
    for i in range(5):
        y = x - aSH[i]
        o = o - (y * y.transpose() + cSH[i]) ** (-1)
    return o


def F22(x):
    aSH = np.matrix([[4, 4, 4, 4],
                     [1, 1, 1, 1],
                     [8, 8, 8, 8],
                     [6, 6, 6, 6],
                     [3, 7, 3, 7],
                     [2, 9, 2, 9],
                     [5, 5, 3, 3],
                     [8, 1, 8, 1],
                     [6, 2, 6, 2],
                     [7, 3.6, 7, 3.6]])
    cSH = np.matrix([[0.1], [0.2], [0.2], [0.4], [0.4], [0.6], [0.3], [0.7], [0.5], [0.5]])
    o = 0
    for i in range(7):
        y = x - aSH[i]
        o = o - (y * y.transpose() + cSH[i]) ** (-1)
    return o


def F23(x):
    aSH = np.matrix([[4, 4, 4, 4],
                     [1, 1, 1, 1],
                     [8, 8, 8, 8],
                     [6, 6, 6, 6],
                     [3, 7, 3, 7],
                     [2, 9, 2, 9],
                     [5, 5, 3, 3],
                     [8, 1, 8, 1],
                     [6, 2, 6, 2],
                     [7, 3.6, 7, 3.6]])
    cSH = np.matrix([[0.1], [0.2], [0.2], [0.4], [0.4], [0.6], [0.3], [0.7], [0.5], [0.5]])
    o = 0
    for i in range(10):
        y = x - aSH[i]
        o = o - (y * y.transpose() + cSH[i]) ** (-1)
    return o


def Rosenbrock(x):
    # return np.sum(100 * np.square(x[1:] - np.square(x[:-1])) + np.square(x[:-1] - 1))
    return rosen(x)


def Rastrigin(x):
    return 10.0 * np.size(x, 0) + np.sum(np.square(x) - 10 * np.cos(2 * np.pi * x))


def Griewank(x):
    dim = np.size(x, 0)
    return 1.0 + np.sum(np.square(x)) / 4000.0 - np.prod(np.cos(x / np.sqrt(np.arange(1, dim + 1))))


def func(F):
    dim = 0
    obj = None
    lb = 0
    ub = 0

    if F == 'F1':
        obj = F1
        lb = -100
        ub = 100
        dim = 100
    elif F == 'F2':
        obj = F2
        lb = -1
        ub = 3
        dim = 100
    elif F == 'F3':
        obj = F3
        lb = -100
        ub = 100
        dim = 100
    elif F == 'F4':
        obj = F4
        lb = -100
        ub = 100
        dim = 100
    elif F == 'F5':
        obj = F5
        lb = -5
        ub = 10
        dim = 100
    elif F == 'F6':
        obj = F6
        lb = -100
        ub = 100
        dim = 100
    elif F == 'F7':
        obj = F7
        lb = -1.28
        ub = 1.28
        dim = 100
    elif F == 'F8':
        obj = F8
        lb = -500
        ub = 500
        dim = 100
    elif F == 'F9':
        obj = F9
        lb = 2.56
        ub = 5.12
        dim = 100
    elif F == 'F10':
        obj = F10
        lb = -32
        ub = 32
        dim = 100
    elif F == 'F11':
        obj = F11
        lb = 300
        ub = 600
        dim = 100
    elif F == 'F12':
        obj = F12
        lb = -50
        ub = 50
        dim = 100
    elif F == 'F13':
        obj = F13
        lb = -50
        ub = 50
        dim = 100
    elif F == 'F14':
        obj = F14
        lb = -65.536
        ub = 65.536
        dim = 2
    elif F == 'F15':
        obj = F15
        lb = -5
        ub = 5
        dim = 4
    elif F == 'F16':
        obj = F16
        lb = -5
        ub = 5
        dim = 2
    elif F == 'F17':
        obj = F17
        lb = np.array([-5, 0])
        ub = np.array([10, 15])
        dim = 2
    elif F == 'F18':
        obj = F18
        lb = -2
        ub = 2
        dim = 2
    elif F == 'F19':
        obj = F19
        lb = 0
        ub = 1
        dim = 3
    elif F == 'F20':
        obj = F20
        lb = 0
        ub = 1
        dim = 6
    elif F == 'F21':
        obj = F21
        lb = 0
        ub = 10
        dim = 4
    elif F == 'F22':
        obj = F22
        lb = 0
        ub = 10
        dim = 4
    elif F == 'F23':
        obj = F23
        lb = 0
        ub = 10
        dim = 4
    elif F == 'Rosenbrock':
        obj = Rosenbrock
        lb = -5
        ub = 10
        dim = 100
    elif F == 'Rastrigin':
        obj = Rastrigin
        lb = 2.56
        ub = 5.12
        dim = 50
    elif F == 'Griewank':
        obj = Griewank
        lb = 300
        ub = 600
        dim = 50
    else:
        pass

    return obj, lb, ub, dim
