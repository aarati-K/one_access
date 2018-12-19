import torch
import torchvision
import torchvision.transforms as transforms
import time
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
])
num_workers = 4

trainset = torchvision.datasets.CocoDetection(root="/home/aarati/datasets/coco/train2017", \
                annFile="/home/aarati/datasets/coco/annotations/instances_train2017.json", \
                transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, num_workers=num_workers, shuffle=True)

total_time = 0
all_times = []
start = time.time()
dataiter = iter(trainloader)
end = time.time()
# Include the time needed to launch child processes
total_time += (end-start)

print(len(trainloader))

for i in range(len(trainloader)):
    start = time.time()
    _, data = dataiter.next()
    end = time.time()
    total_time += (end-start)
    all_times.append(end-start)
    if i%100 == 0:
        print(i)

print(total_time)
f = Path('/home/aarati/workspace/one_access/benchmarks/coco/measurements/pytorch_full_epoch_w{}.npy'.format(num_workers)).open('wb')
np.save(f, np.array(all_times))
f.close()
# 1 worker - Total time: 1378 ~ 23 min
# 2 workers - Total time: 715s ~ 12 min
# 4 workers - Total time: 413s ~ 6.8 min
