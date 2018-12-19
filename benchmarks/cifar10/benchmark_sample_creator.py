from store.cifar10 import Cifar10
import torchvision.transforms as transforms
import time
import matplotlib.pyplot as plt

batch_size = 1
rel_sample_size = 10000
ds = Cifar10(input_data_folder="/home/aarati/datasets/cifar-10-batches-py", \
    max_batches=2, batch_size=batch_size, rel_sample_size=rel_sample_size, \
    max_samples=1, transform=transforms.ToTensor())
ds.count_num_points()
ds.generate_IR()

all_times = []
for i in range(10):
    start = time.time()
    ds.initialize_samples()
    end = time.time()
    all_times.append(end-start)
    s = ds.samples[0].get()
print(all_times)

# Sample creation time for sample size:
# 1: [0.349, 0.306, 0.431, 0.303, 0.18, 0.69, 0.557, 0.681, 0.424, 0.300]
# 10: [0.742, 0.685, 0.679, 0.676, 0.673, 0.676, 0.551, 0.673, 0.669, 0.670]
# 100: [0.713, 0.672, 0.668, 0.671, 0.668, 0.680, 0.682, 0.675, 0.673, 0.669]
# 1000: [0.738, 0.689, 0.704, 0.693, 0.684, 0.683, 0.678, 0.677, 0.700, 0.687]
# 10000: [0.765, 0.727, 0.717, 0.740, 0.723, 0.774, 0.720, 0.868, 0.724, 0.771]

# Plotting code
# x = [1, 10, 50, 100, 1000, 10000]
# y = [0.45, 0.702, 0.703, 0.708, 0.715, 0.746]
# plt.plot(x, y, color='b', marker='o', markerfacecolor='k', markersize=10, fillstyle='full', linewidth=3, linestyle='solid')
# plt.xscale('log')
# plt.ylim(0.40, 0.78)
# plt.xlabel("Reservoir Sample Size", fontsize=20, fontweight='semibold', fontname='serif')
# plt.ylabel("Creation Time (s)", fontsize=20, fontweight='semibold', fontname='serif')
# plt.xticks(x, [1, 10, '', 100, 1000, 10000])
# _, ticks = plt.xticks()
# for tick in ticks:
#     tick.set_fontsize(16)
#     tick.set_fontweight('medium')
#     tick.set_fontname('serif')
# _, ticks = plt.yticks()
# for tick in ticks:
#     tick.set_fontsize(16)
#     tick.set_fontweight('medium')
#     tick.set_fontname('serif')
# plt.show()
