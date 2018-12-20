from pathlib import Path
import math
import matplotlib.pyplot as plt
import numpy as np
import pylab

oa = np.load(Path("../measurements/one_access_400_full_epoch.npy").open('rb'))
pt = np.load(Path("../measurements/pytorch_full_epoch_w1.npy").open('rb'))

fig, ax = plt.subplots()
line, = ax.plot(oa, color='blue', lw=2)
line, = ax.plot(pt, color='red', lw=2)
ax.set_yscale('log')
ax.legend([
    'One Access with Sample Size 400',
    'PyTorch'
], loc=2, prop={'size': 14})
ax.set_xlabel('Iterations', fontsize=18)
ax.set_ylabel('Data Loading Time (s)', fontsize=18)
plt.savefig("coco_measurements.png")