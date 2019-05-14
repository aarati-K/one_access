import matplotlib.pyplot as plt

ss1 = [1.12, 1.37, 2.32, 1.31, 2.42, 3.29, 2.92, 0.29, 1.53, 1.08]
ss10 = [7.6, 11.75, 7.53, 12.64, 10.5, 13.93, 8.56, 10.16, 7.23, 13.1]
ss100 = [24.1, 24.96, 24.9, 25.04, 25.07, 25.06, 25.25, 24.1, 22.52, 25.6]
ss1000 = [26.35, 26.56, 26.44, 26.27, 26.31, 26.39, 27.25, 27.18, 26.43, 26.34]
ss10000 = [28.3, 29.56, 29.55, 29.42, 29.09, 29.09, 29.36, 29.34, 29.8, 29.36]

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
ax.set_title('MS COCO Sample Creation Times', fontsize=18)

# fill with colors
colors = ['lightblue', 'lightblue', 'lightblue']

for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

# adding horizontal grid lines
ax.yaxis.grid(True)
ax.set_xlabel('Sample Size', fontsize=18)
ax.set_ylabel('Sample Creation Time (s)', fontsize=18)

plt.savefig("sample_creator_benchmark.png")