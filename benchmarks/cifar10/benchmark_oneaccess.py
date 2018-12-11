from store.cifar10 import Cifar10
from load import DataLoader
import torchvision.transforms as transforms
import time
import matplotlib.pyplot as plt


batch_size = 4
ds = Cifar10(
    input_data_folder="/Users/srujith/bigdataproject/one_access/benchmarks/data/cifar-10-batches-py", \
    max_batches=2, batch_size=batch_size, max_samples=16, transform=transforms.ToTensor())
ds.initialize()
dl = DataLoader(ds)

total_time = 0
all_times = []
for i in range(100):
    # if (i % 10 == 0):
    #     print(i)
    start = time.time()
    d, l = dl.get_next_batch()
    end = time.time()
    total_time += (end - start)
    all_times.append(end - start)

print(total_time)
print(all_times)
plt.plot(all_times)
plt.show()
dl.stop_batch_creation()
