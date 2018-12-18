# Source: https://github.com/pytorch/examples/tree/master/mnist_hogwild

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torchvision import transforms

from load.load import DataLoader
from store.cifar10 import Cifar10
from train import train_pytorch, train_one_access, test
import time

ONE_ACCESS = 1
PYTORCH = 2

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--rel-sample-size', type=int, default=400, metavar='N',
                    help="relative sample size for sample creator (default: 500)")
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num-processes', type=int, default=2, metavar='N',
                    help='how many training processes to use (default: 2)')
parser.add_argument('--loader', type=int, default=2,
                    help='Which data loader to use')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    model = Net()
    model.share_memory() # gradients are allocated lazily, so they are not shared here

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    start = time.time()
    if args.loader == ONE_ACCESS:
        data_store = Cifar10(
            input_data_folder="/Users/srujith/datasets/cifar-10-batches-py", \
            max_batches=4, batch_size=args.batch_size, \
            rel_sample_size=args.rel_sample_size, max_samples=1, \
            transform=transform)
        data_store.initialize()
        data_loader = DataLoader(data_store, epochs=(args.epochs*args.num_processes))

    processes = []
    for rank in range(args.num_processes):
        if args.loader == ONE_ACCESS:
            p = mp.Process(target=train_one_access, args=(args, model, data_loader))
            # train_one_access(args, model, data_loader)
        else:
            p = mp.Process(target=train_pytorch, args=(rank, args, model))
            # train_pytorch(rank, args, model)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    end = time.time()
    print(end-start)
    if args.loader == ONE_ACCESS:
        data_loader.stop_batch_creation()

    # Once training is complete, we can test the model
    test(args, model)
