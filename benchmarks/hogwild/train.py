# Source: https://github.com/pytorch/examples/tree/master/mnist_hogwild

import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets, transforms
from cifar import CIFAR10


def train_one_access(args, model, data_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    model.train()
    pid = os.getpid()
    i = 0
    while True:
        try:
            data, target = data_loader.get_next_batch()
        except:
            break
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        i += 1
        if i % args.log_interval == 0:
            print('{}\t{}\tLoss: {:.6f}'.format(
                pid, i, loss.item()))


# def backward_step(loss, optimizer):
#     loss.backward()
#     optimizer.step()


def train_pytorch(rank, args, model):
    torch.manual_seed(args.seed + rank)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_loader = torch.utils.data.DataLoader(
        CIFAR10('~/datasets', train=True, download=True, transform=transform),
        batch_size=args.batch_size, shuffle=True, num_workers=1)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    for epoch in range(1, args.epochs + 1):
        train_epoch_pytorch(epoch, args, model, train_loader, optimizer, criterion)


def train_epoch_pytorch(epoch, args, model, data_loader, optimizer, criterion):
    model.train()
    pid = os.getpid()
    i = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                pid, epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.item()))


def test(args, model):
    torch.manual_seed(args.seed)
    criterion = nn.CrossEntropyLoss()
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('~/datasets', train=False, transform=transform),
        batch_size=args.batch_size, shuffle=True, num_workers=1)

    test_epoch(model, test_loader, criterion)


def test_epoch(model, data_loader, criterion):
    # model.eval()
    # test_loss = 0
    # correct = 0
    # with torch.no_grad():
    #     for data, target in data_loader:
    #         output = model(data)
    #         test_loss += criterion(output, target, reduction='sum').item() # sum up batch loss
    #         pred = output.max(1)[1] # get the index of the max log-probability
    #         correct += pred.eq(target).sum().item()
    #
    # test_loss /= len(data_loader.dataset)
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(data_loader.dataset),
    #     100. * correct / len(data_loader.dataset)))
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))
