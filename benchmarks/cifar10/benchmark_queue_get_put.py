from store.cifar10 import Cifar10
import torchvision.transforms as transforms
import time
import matplotlib.pyplot as plt
from sampling.sample_creator import  SampleCreator

batch_size = 1
rel_sample_size = 2000
ds = Cifar10(
    input_data_folder="/home/aarati/datasets/cifar-10-batches-py", \
    max_batches=2, batch_size=batch_size, rel_sample_size=rel_sample_size, max_samples=1, transform=transforms.ToTensor())
ds.initialize()
start = time.time()
s = ds.samples[0].get()
end = time.time()
# print(type(s))
print(end-start)

# varying SAMPLE SIZES
# 1: 0.000275
# 10: 0.000269
# 100: 0.00088
# 300: 0.00177
# 500: 0.00267
# 700: 0.00374
# 900: 0,00489
# 1000: 0.00506
# 1200: 0.00608
# 1400: 0.00711
# 1600: 0.00673
# 1800: 0.00947
# 2000: 0.01038
# Plotting code - Get
# x = [1, 10, 100, 300, 500, 700, 900, 1000, 1200, 1400, 1600, 1800, 2000]
# y = [0.000275, 0.000269, 0.00088, 0.00177, 0.00267, 0.00374, 0.00489, 0.00506, 0.00608, 0.00711, 0.00673, 0.00947, 0.01038]
# plt.plot(x, [2*v for v in y])
# plt.show()

# batch_size = 10
# rel_sample_size = 1
# ds = Cifar10(
#     input_data_folder="/home/aarati/datasets/cifar-10-batches-py", \
#     max_batches=2, batch_size=batch_size, rel_sample_size=rel_sample_size, max_samples=1, transform=transforms.ToTensor())
# ds.initialize()
#  The code for printing is in sample_creator.py
#  The values turned out to be very similar to queue.get