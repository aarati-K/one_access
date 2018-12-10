import torch
import torchvision
import torchvision.transforms as transforms
import time
import matplotlib.pyplot as plt

transform = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)

all_times = []
d = []
total_time = 0

start = time.time()
for i, data in enumerate(trainloader, 0):
    end = time.time()
    all_times.append(end-start)
    total_time += (end-start)

    inputs, labels = data
    start = time.time()

print(total_time)
plt.plot(all_times)
plt.show()

# Workers
# 0: 3.906
# 1: 10.627
# 2: 7.514
# 3: 6.982
# 4: 7.017
