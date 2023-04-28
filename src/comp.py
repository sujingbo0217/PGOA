import matplotlib.pyplot as plt
import numpy as np


abs_goa = np.array((559.26, 54.37, -687142.08, 5.41, 401622.14, 2635.95, 533.88,
                    -14601.28, 2161.49, 7.88, 3442.19, 916459459.72, 47749.57))

abs_pgoa = np.array((37.67, 40.81, -868354.29, 1.97, 51475.68, 364.50, 195.87,
                     -17828.41, 2129.12, 3.58, 3212.37, 51.93, 10.05))

art_goa = np.array((5.63, 7.27, 25.78, 5.32, 8.51, 5.69, 8.66,
                    7.23, 8.41, 13.68, 10.40, 24.19, 23.58))

art_pgoa = np.array((0.90, 1.10, 3.58, 0.84, 1.26, 0.89, 1.29,
                     1.12, 1.26, 1.95, 1.51, 3.30, 3.25))

# plot
myLabels = np.array(["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12", "F13"])
width = 0.75

# plt.pie((abs_goa - abs_pgoa) / abs(abs_goa), autopct='%.1f%%', labels=myLabels)

# plt.barh(myLabels, (abs_goa - abs_pgoa) / abs(abs_goa), color='c')
# plt.barh(myLabels, art_goa - art_pgoa, color='r')

# plt.bar(myLabels, (abs_goa - abs_pgoa) / abs(abs_goa), width=width, color='c')
plt.bar(myLabels, art_goa - art_pgoa, width=width, color='r')

# plt.title('Average Best Solution Diff. Rate', fontsize=16)
plt.title('Average Running Time Diff.', fontsize=16)
plt.rcParams['savefig.dpi'] = 1200
path = '/Users/sudo/Desktop/Research/PGOA/figs/art.png'
plt.savefig(path)
plt.show()