from store.cifar10 import Cifar10
import time


start = time.time()
ds = Cifar10(input_data_folder="/home/aarati/datasets/cifar-10-batches-py")
ds.count_num_points()
ds.generate_IR()
end = time.time()
print("Total time: ", end-start)
# Total time: 1.5s
