# author: Jingbo Su
# Gannet Optimization Algorithm

import numpy as np
from scipy.special import gamma


class GOA:
    """
    population: population size
    dimension: problem dimension
    max_iter: maximum number of iterations
    lb, ub: lower bound, upper bound
    func: fitness function
    """

    def __init__(self, population, dimension, max_iter, lb, ub, func):
        self.population = population
        self.dimension = dimension
        self.max_iter = max_iter
        self.lb = lb
        self.ub = ub
        self.X = None
        self.MX = None
        self.Xb = None
        self.pop_fit = np.ones(self.population) * np.inf
        self.best = np.inf
        self.curve = np.zeros(self.max_iter + 1)
        self.fitness_func = func

    def exploration(self, iteration):
        t = 1 - iteration / self.max_iter
        a = 2 * np.cos(2 * np.pi * np.random.rand()) * t

        def V(x):
            return ((-1 / np.pi) * x + 1) * (0 < x < np.pi) + ((1 / np.pi) * x - 1) * (np.pi <= x < 2 * np.pi)

        b = 2 * V(2 * np.pi * np.random.rand()) * t
        A = (2 * np.random.rand() - 1) * a
        B = (2 * np.random.rand() - 1) * b
        
        for i in range(self.population):
            q = np.random.rand()
            Xi = self.X[i]
            if q >= 0.5:
                u1 = np.random.uniform(-a, a, self.dimension)
                rand = np.random.randint(self.population)
                Xr = self.X[rand]
                u2 = A * (Xi - Xr)
                self.MX[i] = Xi + u1 + u2
            else:
                v1 = np.random.uniform(-b, b, self.dimension)
                Xm = np.mean(self.X)
                v2 = B * (Xi - Xm)
                self.MX[i] = Xi + v1 + v2

            self.bound_check()
            self.update()

    def exploitation(self, iteration):
        t2 = 1 + iteration / self.max_iter
        M = 2.5
        vel = 1.5
        L = 0.2 + (2 - 0.2) * np.random.rand()
        R = (M * vel ** 2) / L
        CC = 1 / (R * t2)
        c = 0.2  # 0.15
        for i in range(self.population):
            Xi = self.X[i]
            if CC >= c:
                delta = CC * np.abs(Xi - self.Xb)
                self.MX[i] = t2 * delta * (Xi - self.Xb) + Xi
            else:
                P = self.Levy(self.dimension)
                self.MX[i] = self.Xb - (Xi - self.Xb) * P * t2

            self.bound_check()
            self.update()

    @staticmethod
    def Levy(dimension):
        beta = 1.5
        sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) / (
            gamma(((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)))) ** (1 / beta)
        mu = np.random.rand(dimension)
        v = np.random.rand(dimension)
        return 0.01 * mu * sigma / ((np.abs(v)) ** (1 / beta))

    def bound_check(self):
        for i in range(self.population):
            self.MX[i] = np.where(
                self.MX[i] < self.lb, self.lb, self.MX[i])
            self.MX[i] = np.where(
                self.MX[i] > self.ub, self.ub, self.MX[i])

    def update(self):
        for i in range(self.population):
            new_fit = self.fitness_func(self.MX[i])
            if new_fit < self.pop_fit[i]:
                self.pop_fit[i] = new_fit
                self.X[i] = self.MX[i]
            if new_fit < self.best:
                self.best = new_fit
                self.Xb = self.MX[i]

    def run(self):
        # X: D * N
        self.X = np.random.rand(self.population, self.dimension)
        for d in range(self.dimension):
            self.X[:, d] = np.random.rand(self.population) * (self.ub[d] - self.lb[d]) + self.lb[d]
        self.MX = self.X
        self.Xb = self.X[0]

        for i in range(self.population):
            self.pop_fit[i] = self.fitness_func(self.X[i])
            if self.pop_fit[i] < self.best:
                self.best = self.pop_fit[i]
                self.Xb = self.X[i]

        for iteration in range(1, self.max_iter + 1):
            rand = np.random.rand()
            if rand > 0.5:
                self.exploration(iteration)
            else:
                self.exploitation(iteration)
            self.curve[iteration] = self.best
