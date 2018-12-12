from store.cifar10 import Cifar10
from load.load import DataLoader
import torchvision.transforms as transforms
import time
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


batch_size = 4
rel_sample_size = 500
ds = Cifar10(
    input_data_folder="/Users/srujith/bigdataproject/one_access/benchmarks/data/cifar-10-batches-py", \
    max_batches=2, batch_size=batch_size, rel_sample_size=rel_sample_size, max_samples=1, transform=transforms.ToTensor())
ds.initialize()
dl = DataLoader(ds)

total_time = 0
all_times = []
for i in range(2000):
    # if (i % 10 == 0):
    #     print(i)
    start = time.time()
    d, l = dl.get_next_batch()
    end = time.time()
    total_time += (end - start)
    all_times.append(end - start)

print(total_time)
# print(all_times)
# plt.plot(all_times)
# plt.show()

f = Path('/home/aarati/workspace/one_access/benchmarks/cifar10/measurements/one_access_{}.npy'.format(rel_sample_size)).open('wb')
np.save(f, np.array(all_times))
f.close()

dl.stop_batch_creation()
# 100: 14.42
# 500: 3.3
# 1000: 2.02

# Plotting code
# folder_name = "/home/aarati/workspace/one_access/benchmarks/cifar10/measurements/"
# fnames = ["one_access_1000.npy", "pytorch.npy"]
# data = []
# for fname in fnames:
#     data.append(np.load(folder_name+fname))

# plt.plot(range(1, 2001), data[1], color='orange', label="PyTorch (Total time 4.01s)")
# plt.plot(range(1, 2001), data[0], color='b', label="OneAccess (Total time 2.75s)")
# plt.yscale("log")
# plt.xlabel("Iterations $\longrightarrow$", fontsize=20, fontweight='semibold', fontname='serif')
# plt.ylabel("Time taken (s)", fontsize=20, fontweight='semibold', fontname='serif')
# plt.xticks([1, 1000, 2000])
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
# plt.legend(fontsize=16)
# plt.show()
# plt.legend(loc="upper right", fontsize=14, markerscale=1.5, edgecolor='k')
