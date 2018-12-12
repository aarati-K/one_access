from store.cifar10 import Cifar10
import torchvision.transforms as transforms
import time
import matplotlib.pyplot as plt

batch_size = 1
ds = Cifar10(input_data_folder="/home/aarati/datasets/cifar-10-batches-py", \
    max_batches=2, batch_size=batch_size, rel_sample_size=10, \
    max_samples=1, transform=transforms.ToTensor())
start = time.time()
ds.initialize()
end = time.time()
print(end-start)

# Sample creation time for sample size:
# 1: 0.45
# 10: 0.702
# 50: 0.703
# 100: 0.708
# 1000: 0.715
# 10000: 0.746

# Plotting code
# x = [1, 10, 50, 100, 1000, 10000]
# y = [0.45, 0.702, 0.703, 0.708, 0.715, 0.746]
# plt.plot(x, y, color='b', marker='o', markerfacecolor='k', markersize=10, fillstyle='full', linewidth=3, linestyle='solid')
# plt.xscale('log')
# plt.ylim(0.40, 0.78)
# plt.xlabel("Reservoir Sample Size", fontsize=14, fontweight='semibold', fontname='serif')
# plt.ylabel("Creation Time (s)", fontsize=14, fontweight='semibold', fontname='serif')
# plt.xticks(x, [1, 10, '', 100, 1000, 10000])
# plt.show()
