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