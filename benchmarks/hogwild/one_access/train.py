# Source: https://github.com/pytorch/examples/tree/master/mnist_hogwild

import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms


def train(args, model, data_loader):
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    model.train()
    # pid = os.getpid()
    while True:
        try:
            (data, target) = data_loader.get_next_batch()
        except:
            break
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        # if batch_idx % args.log_interval == 0:
        #     print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         pid, epoch, batch_idx * len(data), len(data_loader.dataset),
        #                     100. * batch_idx / len(data_loader), loss.item()))


def test(args, model):
    torch.manual_seed(args.seed)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('~/datasets/mnist', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.batch_size, shuffle=True, num_workers=1)

    test_epoch(model, test_loader)


def test_epoch(model, data_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target).sum().item()

    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))
