from pathlib import Path
import math
import matplotlib.pyplot as plt
import numpy as np
import pylab

one_access_500 = np.load(Path("../measurements/one_access_500.npy").open('rb'))
one_access_1000 = np.load(
    Path("../measurements/one_access_1000.npy").open('rb'))
pt = np.load(Path("../measurements/pytorch.npy").open('rb'))

fig, ax = plt.subplots()
fig.set_figheight(5)
fig.set_figwidth(7)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

line, = ax.plot(one_access_500, color='blue', lw=2)
line, = ax.plot(one_access_1000, color='lightblue', lw=2)
line, = ax.plot(pt, color='red', lw=2)
ax.set_yscale('log')
ax.legend([
    'One Access with Sample Size 500', 'One Access with Sample Size 1000',
    'PyTorch'
],
          loc=2,
          prop={'size': 14})
ax.set_xlabel('Iterations', fontsize=18)
ax.set_ylabel('Data Loading Time (s)', fontsize=18)

plt.savefig("cifar10_measurements.png")