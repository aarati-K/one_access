import matplotlib.pyplot as plt

ss1 = [0.349, 0.306, 0.431, 0.303, 0.18, 0.69, 0.557, 0.681, 0.424, 0.300]
ss10 = [0.742, 0.685, 0.679, 0.676, 0.673, 0.676, 0.551, 0.673, 0.669, 0.670]
ss100 = [0.713, 0.672, 0.668, 0.671, 0.668, 0.680, 0.682, 0.675, 0.673, 0.669]
ss1000 = [0.738, 0.689, 0.704, 0.693, 0.684, 0.683, 0.678, 0.677, 0.700, 0.687]
ss10000 = [
    0.765, 0.727, 0.717, 0.740, 0.723, 0.774, 0.720, 0.868, 0.724, 0.771
]

fig, ax = plt.subplots()
fig.set_figheight(5)
fig.set_figwidth(7)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

bplot = ax.boxplot(
    [ss1, ss10, ss100, ss1000, ss10000],
    vert=True,  # vertical box alignment
    patch_artist=True,  # fill with color
    labels=[1, 10, 100, 1000, 10000])  # will be used to label x-ticks
ax.set_title('Cifar-10 Sample Creation Times', fontsize=18)

# fill with colors
"""
colors = ['lightblue', 'lightblue', 'lightblue']

for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)
"""

# adding horizontal grid lines
ax.yaxis.grid(True)
ax.set_xlabel('Sample Size', fontsize=18)
ax.set_ylabel('Sample Creation Time (s)', fontsize=18)

plt.savefig("cifar10_sample_creation.png")