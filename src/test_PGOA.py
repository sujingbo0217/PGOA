# author: Jingbo Su
# Gannet Optimization Algorithm with Parallel

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
    communications: the number of iterations for communication
    """

    def __init__(self, population, dimension, max_iter, lb, ub, func, groups=4, strategy=1, migration=0.5, copies=1,
                 communications=20):
        self.population = population
        self.dimension = dimension
        self.max_iter = max_iter
        self.lb = lb
        self.ub = ub

        self.groups = groups
        self.Np = self.population // self.groups

        self.X = None
        self.MX = None

        self.curve = np.zeros(self.max_iter + 1)
        self.fitness_func = func

        self.global_best = np.zeros(self.dimension)
        self.global_min = np.inf

        self.group_best = np.zeros((self.groups, self.dimension))
        self.group_min = np.ones(self.groups) * np.inf
        self.pop_fit = np.ones((self.groups, self.Np))

        self.rate = self.max_iter // communications
        self.strategy = strategy
        self.copies = copies
        self.copies_ = copies
        self.migration = migration

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
            Xi = self.X[group, :, i]
            if q >= 0.5:
                u1 = np.random.uniform(-a, a, self.dimension)
                rand = np.random.randint(self.Np)
                while rand == i:
                    rand = np.random.randint(self.Np)
                Xr = self.X[group, :, rand]
                u2 = A * (Xi - Xr)
                self.MX[group, :, i] = Xi + u1 + u2
            else:
                v1 = np.random.uniform(-b, b, self.dimension)
                Xm = np.mean(self.X[group])
                v2 = B * (Xi - Xm)
                self.MX[group, :, i] = Xi + v1 + v2

            self.bound_check(group)
            self.update(group)

    def exploitation(self, iteration, group):
        t2 = 1 + iteration / self.max_iter
        M = 2.5
        vel = 1.5
        L = 0.2 + (2 - 0.2) * np.random.rand()
        R = (M * vel ** 2) / L
        Capturability = 1 / (R * t2)
        c = 0.2  # 0.15
        for i in range(self.Np):
            Xi = self.X[group, :, i]
            if Capturability >= c:
                delta = Capturability * np.abs(Xi - self.group_best[group])
                self.MX[group, :, i] = t2 * delta * \
                                       (Xi - self.group_best[group]) + Xi
            else:
                P = self.Levy(self.dimension)
                self.MX[group, :, i] = self.group_best[group] - \
                                       (Xi - self.group_best[group]) * P * t2

            self.bound_check(group)
            self.update(group)

    @staticmethod
    def Levy(dimension):
        beta = 1.5
        sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) / (
            gamma(((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)))) ** (1 / beta)
        mu = np.random.rand(dimension)
        v = np.random.rand(dimension)
        return 0.01 * mu * sigma / ((np.abs(v)) ** (1 / beta))

    def bound_check(self, group):
        for i in range(self.Np):
            self.MX[group, :, i] = np.where(
                self.MX[group, :, i] < self.lb, self.lb, self.MX[group, :, i])
            self.MX[group, :, i] = np.where(
                self.MX[group, :, i] > self.ub, self.ub, self.MX[group, :, i])

    def update(self, group):
        for i in range(self.Np):
            new_fit = self.fitness_func(self.MX[group, :, i])
            if new_fit < self.pop_fit[group][i]:
                self.pop_fit[group][i] = new_fit
                self.X[group, :, i] = self.MX[group, :, i]
            if new_fit < self.global_min:
                self.global_min = new_fit
                self.global_best = self.MX[group, :, i]

    def run(self):
        # X: G * D * Np
        self.X = np.zeros((self.groups, self.dimension, self.Np))
        for G in range(self.groups):
            for D in range(self.dimension):
                self.X[G, D, :] = np.random.rand(
                    self.Np) * (self.ub[D] - self.lb[D]) + self.lb[D]

        self.MX = self.X

        for G in range(self.groups):
            for i in range(self.Np):
                self.pop_fit[G][i] = self.fitness_func(self.X[G, :, i])
                # update group best value
                if self.pop_fit[G][i] < self.group_min[G]:
                    self.group_min[G] = self.pop_fit[G][i]
                    self.group_best[G] = self.X[G, :, i]

                # update global best value
                if self.pop_fit[G][i] < self.global_min:
                    self.global_min = self.pop_fit[G][i]
                    self.global_best = self.X[G, :, i]

        # sub_groups = self.groups // 2
        n = int(np.log2(self.groups))
        m = 0

        for iter in range(1, self.max_iter + 1):
            for G in range(self.groups):
                rand = np.random.rand()
                if rand > 0.5:
                    self.exploration(iter, G)
                else:
                    self.exploitation(iter, G)

                # Update
                for i in range(self.Np):
                    # update group best value
                    if self.pop_fit[G][i] < self.group_min[G]:
                        self.group_min[G] = self.pop_fit[G][i]
                        self.group_best[G] = self.X[G, :, i]

                    # update global best value
                    if self.pop_fit[G][i] < self.global_min:
                        self.global_min = self.pop_fit[G][i]
                        self.global_best = self.X[G, :, i]

                if self.strategy == 1:
                    if iter % self.rate == 0:
                        sorted_pop_fit = np.sort(self.pop_fit[G])[::-1]
                        expected_pop_fit = sorted_pop_fit[int(np.size(sorted_pop_fit) * self.migration)]

                        if iter % (self.rate * 2) == 0:
                            # global update
                            for i in range(self.Np):
                                if self.fitness_func(self.X[G, :, i]) >= expected_pop_fit:
                                    self.X[G, :, i] = self.global_best
                        else:
                            # local update
                            for i in range(self.Np):
                                if self.fitness_func(self.X[G, :, i]) >= expected_pop_fit:
                                    self.X[G, :, i] = self.group_best[G]

                elif self.strategy == 2:
                    if iter % self.rate == 0:
                        self.copies = self.copies_
                        while self.copies > 0:
                            q = G ^ (2 ** m)
                            if m == n - 1:
                                m = 0
                            else:
                                m += 1
                            for i in range(self.Np):
                                if self.group_min[G] < self.group_min[q]:
                                    self.group_min[q] = self.group_min[G]
                                    self.X[q, :, i] = self.group_best[G]
                            self.copies -= 1

                elif self.strategy == 3:
                    if iter % self.rate == 0:
                        rand = np.random.rand()

                        # Strategy 1
                        if rand <= 0.5:
                            if iter % self.rate == 0:
                                sorted_pop_fit = np.sort(self.pop_fit[G])[::-1]
                                expected_pop_fit = sorted_pop_fit[int(np.size(sorted_pop_fit) * self.migration)]

                                if iter % (self.rate * 2) == 0:
                                    # global update
                                    for i in range(self.Np):
                                        if self.fitness_func(self.X[G, :, i]) >= expected_pop_fit:
                                            self.X[G, :, i] = self.global_best
                                else:
                                    # local update
                                    for i in range(self.Np):
                                        if self.fitness_func(self.X[G, :, i]) >= expected_pop_fit:
                                            self.X[G, :, i] = self.group_best[G]

                        # Strategy 2
                        else:
                            self.copies = self.copies_
                            while self.copies > 0:
                                q = G ^ (2 ** m)
                                if m == n - 1:
                                    m = 0
                                else:
                                    m += 1
                                for i in range(self.Np):
                                    if self.group_min[G] < self.group_min[q]:
                                        self.group_min[q] = self.group_min[G]
                                        self.X[q, :, i] = self.group_best[G]
                                self.copies -= 1

            self.curve[iter] = self.global_min
