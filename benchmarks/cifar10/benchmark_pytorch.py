import torch
import torchvision
import torchvision.transforms as transforms
import time
from cifar import CIFAR10
import matplotlib.pyplot as plt

transform = transforms.Compose([transforms.ToTensor()])

trainset = CIFAR10(root='./data', train=True,
                                        download=True, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)

time.sleep(4)

all_times = []
d = []
total_time = 0

start = time.time()
for i,data in enumerate(trainloader,0):
    if i==100:
        break
    end = time.time()
    all_times.append(end-start)
    total_time += (end-start)

    inputs, labels = data
    start = time.time()

print(total_time-all_times[0])
plt.plot(all_times[1:])
plt.show()

# Workers
# 0: 3.906
# 1: 10.627
# 2: 7.514
# 3: 6.982
# 4: 7.017
