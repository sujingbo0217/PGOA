import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import GOA as GOA
import PGOA as PGOA
# from sko.PSO import PSO

import get_functions_details as getf
import time
from datetime import datetime
import bless

now = datetime.now()
print("Starting Time: ", now.strftime("%Y-%m-%d %H:%M:%S"))

bless.me()

"""
Test cases
D: dimension
N: population
Iter: max iteration
func: fitness function
lb: lower bound
ub: upper bound
"""

N = 80
Iter = 50

runs = 30
test_case = 2

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
        # ga = GA(func=func, n_dim=D, size_pop=N, max_iter=Iter, prob_mut=0.001, lb=lb, ub=ub, precision=1e-7)
        # zone_0 = time.time()
        # best_x, best_y = ga.run()
        # running_time[0] += time.time() - zone_0
        # curve[0] += ga.best_y
        # best[0] += best_y

        # PSO
        # pso = PSO(func=func, n_dim=D, pop=N, max_iter=Iter, lb=lb, ub=ub, w=0.3, c1=2, c2=2)
        # zone_0 = time.time()
        # pso.run()
        # running_time[0] += time.time() - zone_0
        # for i in range(np.size(pso.gbest_y_hist)):
        #     curve[0][i] += pso.gbest_y_hist[i]
        # best[0] += pso.gbest_y

        # GOA
        goa = GOA.GOA(N, D, Iter, lb, ub, func)
        zone_1 = time.time()
        goa.run()
        running_time[0] += time.time() - zone_1
        curve[0] += goa.curve[1:]
        best[0] += goa.best

        # PGOA
        pgoa = PGOA.PGOA(N, D, Iter, lb, ub, func, 8)
        zone_2 = time.time()
        pgoa.run()
        running_time[1] += time.time() - zone_2
        curve[1] += pgoa.curve[1:]
        best[1] += pgoa.global_fmin

    # Data processing
    for i in range(test_case):
        running_time[i] = running_time[i] / runs
        curve[i] = curve[i] / runs
        best[i] = best[i] / runs

        # outcomes
        print('Experiment', i, ' Avg Best (', best[i], ') Avg T (', running_time[i], ')')

    # plot
    # plt.plot(np.arange(Iter), curve[0], 'c-', label='PSO', linewidth=1)
    plt.plot(np.arange(Iter), curve[0], 'c-', label='GOA', linewidth=1)
    plt.plot(np.arange(Iter), curve[1], 'm-', label='PGOA', linewidth=1)

    plt.title(fs)
    mpl.rcParams.update({'font.size': 10})
    plt.xlabel('Iteration')
    plt.ylabel('Best Solution')
    plt.grid()
    plt.legend()
    mpl.rcParams.update({'font.size': 9})
    # plt.rcParams['figure.dpi'] = 1200
    plt.rcParams['savefig.dpi'] = 1200
    save_path = '/Users/sudo/Desktop/Research/src/figs/test/' + fs + '.png'
    plt.savefig(save_path)
    plt.show()
