import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
from network import  CNN
from torch import optim
from torch.autograd import Variable
import numpy as np
from utils import Subset_noisy
import argparse
from losses import NormalizedCrossEntropy as NCE
from losses import MeanAbsoluteError as MAE
from losses import ReverseCrossEntropy as RCE
from losses import BetaCrossEnropyError as BCE


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
noise_rate=0.2
NUMClass=10

train_data = datasets.MNIST(
    root = 'data',
    train = True,
    transform = ToTensor(),
    download = True,
)


test_data = datasets.MNIST(
    root = 'data',
    train = False,
    transform = ToTensor()
)

if noise_rate:
    indices = np.random.permutation(np.arange(len(train_data )))
    train_size = len(train_data )
    train_data = Subset_noisy(train_data , indices[:int(train_size * noise_rate)], NUMClass)

loaders = {
    'train': torch.utils.data.DataLoader(train_data,
                                         batch_size=100,
                                         shuffle=True,
                                         num_workers=1),

    'test': torch.utils.data.DataLoader(test_data,
                                        batch_size=100,
                                        shuffle=True,
                                        num_workers=1),
}


parser = argparse.ArgumentParser()
#parser.add_argument("--noise_rate",  required=False, default=0.2)
parser.add_argument("--loss",required=False, default='BCE',choices=["CE,MAE,GCE,RCE,BCE"])
parser.add_argument(
    "--device", required=False, default="cuda" if torch.cuda.is_available() else "cpu"
)
args = parser.parse_args()

if args.loss == 'NCE':
    loss_func = NCE(NUMClass, args.device)
elif args.loss == 'MAE':
    loss_func = MAE(NUMClass, args.device)
elif args.loss == 'RCE':
    loss_func = RCE(NUMClass, args.device)
elif args.loss == 'BCE':
    loss_func = BCE(NUMClass, args.device, 0.00000001)
else:
    loss_func = nn.CrossEntropyLoss()

cnn = CNN()
optimizer = optim.Adam(cnn.parameters(), lr = 0.01)
num_epochs = 10



def train(num_epochs, cnn, loaders):
    cnn.train()

    # Train the model
    total_step = len(loaders['train'])

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders['train']):
            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images)  # batch x
            b_y = Variable(labels)  # batch y


            output = cnn(b_x)[0]
            loss = loss_func(output, b_y)

            # clear gradients for this training step
            optimizer.zero_grad()

            # backpropagation, compute gradients
            loss.backward()
            # apply gradients
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

        test()

def test():
    # Test the model
    accuracy=0
    cnn.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loaders['test']:
            test_output, last_layer = cnn(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy += (pred_y == labels).sum().item() / float(labels.size(0))
    print('Test Accuracy of the model on the 10000 test images: %.2f' % accuracy)


train(num_epochs, cnn, loaders)
