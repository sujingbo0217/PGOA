# author: Jingbo Su
# Parallel Gannet Optimization Algorithm

import numpy as np
from scipy.special import gamma


class PGOA:
    """
    population: population size
    dimension: problem dimension
    max_iter: maximum number of iterations
    lb, ub: lower bound, upper bound
    func: fitness function
    groups: number of parallel groups (2^m, m is a positive integer)
    strategy: parallel strategy choice (1, 2, 3)
    migration: the rate of the best particle position is migrated and mutated to substitute the particles of the receiving group (0.25, 0.5, 0.75, 1.0)
    copies: the number of worse particles substituted at each receiving group (1, 2)
    """

    def __init__(self, population, dimension, max_iter, lb, ub, func, group=8):
        self.population = population
        self.dimension = dimension
        self.max_iter = max_iter
        self.lb = lb
        self.ub = ub

        self.group = group
        self.Np = self.population // self.group

        self.X = None
        self.MX = None

        self.curve = np.zeros(self.max_iter + 1)
        self.fitness_func = func

        self.global_best = np.zeros(self.dimension, dtype=int)
        self.global_fmin = np.inf

        self.group_best = np.zeros(self.group, dtype=int)
        self.group_fmin = np.ones(self.group) * np.inf
        self.group_worst = np.zeros(self.group, dtype=int)
        self.group_fmax = np.ones(self.group) * (-np.inf)
        self.pop_fit = np.ones((self.group, self.Np)) * np.inf

        self.copies = self.copies_ = 2
        self.migration = 0.75

    def exploration(self, iteration, group):
        t = 1 - iteration / self.max_iter
        a = 2 * np.cos(2 * np.pi * np.random.rand()) * t

        def V(x):
            return ((-1 / np.pi) * x + 1) * (0 < x < np.pi) + ((1 / np.pi) * x - 1) * (np.pi <= x < 2 * np.pi)

        b = 2 * V(2 * np.pi * np.random.rand()) * t
        A = (2 * np.random.rand() - 1) * a
        B = (2 * np.random.rand() - 1) * b

        for i in range(self.Np):
            q = np.random.rand()
            Xi = self.X[group * self.Np + i]
            if q >= 0.5:
                u1 = np.random.uniform(-a, a, self.dimension)
                rand = np.random.randint(self.Np)
                Xr = self.X[group * self.Np + rand]
                u2 = A * (Xi - Xr)
                self.MX[group * self.Np + i] = Xi + u1 + u2
            else:
                v1 = np.random.uniform(-b, b, self.dimension)
                Xm = np.mean(self.X[group * self.Np: (group + 1) * self.Np])
                v2 = B * (Xi - Xm)
                self.MX[group * self.Np + i] = Xi + v1 + v2

            self.bound_check(group)
            self.update(group)

    def exploitation(self, iteration, group):
        t2 = 1 + iteration / self.max_iter
        M = 2.5
        vel = 1.5
        L = 0.2 + (2 - 0.2) * np.random.rand()
        R = (M * vel ** 2) / L
        CC = 1 / (R * t2)
        c = 0.2  # 0.15
        for i in range(self.Np):
            Xi = self.X[group * self.Np + i]
            if CC >= c:
                delta = CC * np.abs(Xi - self.group_best[group])
                self.MX[group * self.Np + i] = t2 * delta * (Xi - self.X[self.group_best[group]]) + Xi
            else:
                P = self.Levy(self.dimension)
                self.MX[group * self.Np + i] = self.X[self.group_best[group]] - (Xi - self.X[self.group_best[group]]) * P * t2

            self.bound_check(group)
            self.update(group)

    @staticmethod
    def Levy(dimension):
        beta = 1.5
        sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) / (
            gamma(((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)))) ** (1 / beta)
        mu = np.random.rand(dimension)
        v = np.random.rand(dimension)
        return 0.01 * mu * sigma / (v ** (1 / beta))

    def bound_check(self, group):
        for i in range(self.Np):
            self.MX[group * self.Np + i] = np.where(
                self.MX[group * self.Np + i] < self.lb, self.lb, self.MX[group * self.Np + i])
            self.MX[group * self.Np + i] = np.where(
                self.MX[group * self.Np + i] > self.ub, self.ub, self.MX[group * self.Np + i])

    def update(self, group):
        for i in range(self.Np):
            new_fit = self.fitness_func(self.MX[group * self.Np + i])
            if new_fit < self.pop_fit[group][i]:
                self.pop_fit[group][i] = new_fit
                self.X[group * self.Np + i] = self.MX[group * self.Np + i]
                if new_fit < self.global_fmin:
                    self.global_fmin = new_fit
                    self.global_best = group * self.Np + i

    def run(self):
        # X: D * N
        self.X = np.random.rand(self.population, self.dimension)
        for d in range(self.dimension):
            self.X[:, d] = np.random.rand(self.population) * (self.ub[d] - self.lb[d]) + self.lb[d]
        self.MX = self.X

        for g in range(self.group):
            for i in range(self.Np):
                self.pop_fit[g][i] = self.fitness_func(self.X[g * self.Np + i])
                # update group best value
                if self.pop_fit[g][i] < self.group_fmin[g]:
                    self.group_fmin[g] = self.pop_fit[g][i]
                    self.group_best[g] = g * self.Np + i

                # update global best value
                if self.pop_fit[g][i] < self.global_fmin:
                    self.global_fmin = self.pop_fit[g][i]
                    self.global_best = g * self.Np + i

        n = int(np.log2(self.group))
        m = 0

        for iter in range(1, self.max_iter + 1):
            for g in range(self.group):
                rand = np.random.rand()
                if rand > 0.5:
                    self.exploration(iter, g)
                else:
                    self.exploitation(iter, g)

                for i in range(self.Np):
                    fit = self.pop_fit[g][i]
                    xx = g * self.Np + i
                    # Update
                    if fit < self.group_fmin[g]:
                        self.group_fmin[g] = fit
                        self.group_best[g] = xx
                        if fit < self.global_fmin:
                            self.global_fmin = fit
                            self.global_best = xx
                    if fit > self.group_worst[g]:
                        self.group_fmax[g] = fit
                        self.group_worst[g] = xx

                # Strategy.1
                if iter % 5 == 0:
                    # Random selection
                    self.copies = self.copies_
                    while self.copies > 0:
                        q = g ^ (2 ** m)
                        m = 0 if m == n - 1 else m + 1
                        # q
                        sorted_pop_fit = np.sort(self.pop_fit[q])[::-1]
                        expected_pop_fit = sorted_pop_fit[int(np.size(sorted_pop_fit) * self.migration)]
                        for i in range(self.Np):
                            if self.fitness_func(self.X[q * self.Np + i]) >= expected_pop_fit:
                                self.X[q * self.Np + i] = self.X[self.group_best[g]]

                        self.copies -= 1

                # Strategy.2
                if iter % 3 == 0:
                    idx = self.group_worst[g]
                    wox = self.X[idx]
                    half = np.size(wox) // 2
                    wox[:half] = self.X[self.group_best[g]][:half]
                    wox[half:] = self.X[self.global_best][half:]
                    fit = self.fitness_func(wox)
                    if fit < self.group_fmax[g]:
                        self.group_fmax[g] = fit
                        self.group_worst[g] = idx
                        if fit < self.group_fmin[g]:
                            self.group_fmin[g] = fit
                            self.group_best[g] = idx
                            if fit < self.global_fmin:
                                self.global_fmin = fit
                                self.global_best = idx

            self.curve[iter] = self.global_fmin
