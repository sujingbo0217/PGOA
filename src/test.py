import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import GOA as GOA
import PGOA as PGOA
from sko.PSO import PSO
from sko.GA import GA

import get_functions_details as getf
import time
from datetime import datetime
import bless

now = datetime.now()
print("Starting Time: ", now.strftime("%Y-%m-%d %H:%M:%S"))

# bless.me()

"""
Test cases
D: dimension
N: population
Iter: max iteration
func: fitness function
lb: lower bound
ub: upper bound
"""

N = 16
Iter = 5

runs = 30
test_case = 4

for I in range(1, 14):
    running_time = np.zeros(test_case)
    curve = np.zeros((test_case, Iter))
    best = np.zeros(test_case)

    fs = 'F' + str(I)
    func, l_b, u_b, D = getf.func(fs)

    lb = np.ones(D) * l_b
    ub = np.ones(D) * u_b

    # Running
    print('Running ' + fs + '...\n')

    for iter in range(runs):
        print('Iteration:', iter + 1)

        # GA
        ga = GA(func=func, n_dim=D, size_pop=N, max_iter=Iter, prob_mut=0.001, lb=lb, ub=ub, precision=1e-7)
        zone_0 = time.time()
        best_x, best_y = ga.run()
        running_time[0] += time.time() - zone_0
        curve[0] += ga.best_y
        best[0] += best_y

        # PSO
        pso = PSO(func=func, n_dim=D, pop=N, max_iter=Iter, lb=lb, ub=ub, w=0.8, c1=0.5, c2=0.5)
        zone_1 = time.time()
        pso.run()
        running_time[1] += time.time() - zone_1
        for i in range(np.size(pso.gbest_y_hist)):
            curve[1][i] += pso.gbest_y_hist[i]
        best[1] += pso.gbest_y

        # GOA
        goa = GOA.GOA(N, D, Iter, lb, ub, func)
        zone_2 = time.time()
        goa.run()
        running_time[2] += time.time() - zone_2
        curve[2] += goa.curve[1:]
        best[2] += goa.best

        # PGOA
        pgoa = PGOA.PGOA(N, D, Iter, lb, ub, func)
        zone_3 = time.time()
        pgoa.run()
        running_time[3] += time.time() - zone_3
        curve[3] += pgoa.curve[1:]
        best[3] += pgoa.global_fmin

        """
        groups: number of parallel groups (2^m, m is a positive integer, N/groups is an integer)
        strategy: parallel strategy choice (1, 2, 3)
        migration: the rate of the best particle position is migrated and mutated to substitute the particles of the receiving group (0.25, 0.5, 0.75)
        copies: the number of worse particles substituted at each receiving group (1, 2, 4)
        communications: the number of iterations for communication (Iter/communications is an integer)
        """

    # Data processing
    for i in range(test_case):
        running_time[i] = running_time[i] / runs
        curve[i] = curve[i] / runs
        best[i] = best[i] / runs

        # outcomes
        print('Experiment', i, ' Avg Best (', best[i], ') Avg T (', running_time[i], ')')

    # plot
    plt.plot(np.arange(Iter), curve[0], 'c-', label='GA', linewidth=1)
    plt.plot(np.arange(Iter), curve[1], 'm--', label='PSO', linewidth=1)
    plt.plot(np.arange(Iter), curve[2], 'g--', label='GOA', linewidth=1)
    plt.plot(np.arange(Iter), curve[3], 'r--', label='PGOA', linewidth=1)

    plt.title(fs)
    mpl.rcParams.update({'font.size': 10})
    plt.xlabel('Iteration')
    plt.ylabel('Best Solution')
    plt.grid()
    plt.legend()
    mpl.rcParams.update({'font.size': 9})
    # plt.rcParams['figure.dpi'] = 1200
    # plt.rcParams['savefig.dpi'] = 1200
    # path = '/Users/sudo/Desktop/Research/src/figs/test/' + fs + '.png'
    # plt.savefig(path)
    plt.show()
