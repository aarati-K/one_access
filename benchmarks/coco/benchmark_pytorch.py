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

trainset = torchvision.datasets.CocoDetection(root="/home/aarati/datasets/coco/train2017", \
                annFile="/home/aarati/datasets/coco/annotations/instances_train2017.json", \
                transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, num_workers=1)

i = 0
total_time = 0
start = time.time()
dataiter = iter(trainloader)
while True:
    _, data = dataiter.next()
    end = time.time()
    total_time += (end-start)
    if i == 1000:
        break
    i += 1

print(total_time)
# Total time: 197107.41146445274
