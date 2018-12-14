import torch
import torchvision
import torchvision.transforms as transforms
import time
from cifar import CIFAR10
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

transform = transforms.Compose([transforms.ToTensor()])

trainset = CIFAR10(root='/home/aarati/datasets', train=True,
        download=False, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
        shuffle=True, num_workers=1)

time.sleep(4)

all_times = []
d = []
total_time = 0

start = time.time()
for i,data in enumerate(trainloader,0):
    if i==2000:
        break
    end = time.time()
    all_times.append(end-start)
    total_time += (end-start)

    inputs, labels = data
    start = time.time()

print(total_time)
plt.plot(all_times)
plt.show()
f = Path('/home/aarati/workspace/one_access/benchmarks/cifar10/measurements/pytorch.npy').open('wb')
np.save(f, np.array(all_times))
f.close()

# 4.015

# Workers
# 0: 3.906
# 1: 10.627
# 2: 7.514
# 3: 6.982
# 4: 7.017
