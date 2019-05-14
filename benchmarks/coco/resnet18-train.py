import csv
import time
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

# Data
transform = transforms.Compose(
    [transforms.RandomResizedCrop(224),
     transforms.ToTensor()])

train_dataset = datasets.CocoDetection(
    root='/home/aarati/datasets/coco/train2017',
    annFile='/home/aarati/datasets/coco/annotations/instances_train2017.json',
    transform=transform)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=2)

# Model
model = models.resnet18(pretrained=False)

# Train
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

num_epochs = 1
learning_rate = 0.001

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_step = len(train_loader)
curr_lr = learning_rate
for epoch in range(num_epochs):
    step = 0
    data_times = []
    compute_times = []
    start = time.time()
    for images, labels in train_loader:
        data_times.append(time.time() - start)
        start = time.time()
        step += 1
        if step == total_step:
            break

        images = images.to(device)
        labels = torch.ones([32, 1000])
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        compute_times.append(time.time() - start)
        if step % 100 == 0:
            print("Step = " + str(step) + " out of " + str(total_step))
        start = time.time()

    with open('/home/aarati/workspace/one_access/benchmarks/coco/measurements/coco-resnet18-pytorch', 'w', newline='') as logfile:
        logwriter = csv.writer(
            logfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in zip(data_times, compute_times):
            logwriter.writerow(row)
