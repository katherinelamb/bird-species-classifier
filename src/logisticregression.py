import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torchvision.transforms as T
import torchvision.datasets as dset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import torchvision.datasets as dset
from torchvision import models

import os
dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_path = os.path.join(dir_path, "data/100-bird-species")

# Use GPU if applicable, taken from Assignment 2 notebook
USE_GPU = True

dtype = torch.float32

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('using device:', device)

# Set up a transform to preprocess the data by subtracting the mean RGB value and dividing by the
# standard deviation of each RGB value
transform = T.Compose(
    [T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

# Load data
trainset = dset.ImageFolder(root=os.path.join(data_path, "train"), transform=transform)
trainloader = DataLoader(trainset, batch_size=64, num_workers=0, shuffle=True)

valset = torchvision.datasets.ImageFolder(root=os.path.join(data_path, "valid"), transform=transform)
valloader = DataLoader(valset, batch_size=64, num_workers=0, shuffle=False)

testset = torchvision.datasets.ImageFolder(root=os.path.join(data_path, "test"), transform=transform)
testloader = DataLoader(testset, batch_size=64, num_workers=0, shuffle=False)

dataloaders = {
    "train": trainloader,
    "val": valloader,
    "test": testloader
}
datasizes = {
    "train": len(trainset),
    "val": len(valset),
    "test": len(testset)
}
CLASSES = list(trainset.class_to_idx.keys())

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

batch_size = 100
n_iters = 3000
# epochs = n_iters / (len(trainset) / batch_size)
epochs = 5
input_dim = 224 * 224 * 3
output_dim = 200
lr_rate = 0.001

model = LogisticRegression(input_dim, output_dim)

criterion = torch.nn.CrossEntropyLoss() # computes softmax and then the cross entropy

optimizer = torch.optim.SGD(model.parameters(), lr=lr_rate)

iter = 0
for epoch in range(int(epochs)):
    for i, (images, labels) in enumerate(trainloader):
        images = Variable(images.view(-1, 224 * 224 * 3))
        labels = Variable(labels)
        inputs = images.to(device=device, dtype=dtype)
        labels = labels.to(device=device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        iter+=1
        if iter%500==0:
            # calculate Accuracy
            correct = 0
            total = 0
            for images, labels in testloader:
                images = Variable(images.view(-1, 224 * 224 * 3))
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total+= labels.size(0)
                # for gpu, bring the predicted and labels back to cpu fro python operations to work
                correct+= (predicted == labels).sum()
            accuracy = 100 * correct/total
            print("Iteration: {}. Loss: {}. Accuracy: {}.".format(iter, loss.item(), accuracy))

# View a minibatch of data - from Vipin Chaudhary's Kaggle notebook
def imshow(img, size=(10, 10)):
    img = img / 2 + 0.5
    npimg = img.numpy()
    if size:
        plt.figure(figsize=size)

    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title("One mini batch")
    plt.axis("off")
    plt.pause(8)

def imshowaxis(ax, img, orig, pred):
    img = img / 2 + 0.5
    npimg = img.numpy()
    ax.imshow(np.transpose(npimg, (1, 2, 0)))
    if orig != pred:
        ax.set_title(orig + "\n" + pred, color="red")
    else:
        ax.set_title(orig + "\n" + pred)
    ax.axis("off")

def vis_model(model, num_images=25):
    was_training = model.training
    model.eval()
    images_so_far = 0
    figure, ax = plt.subplots(5, 5, figsize=(20, 20))

    with torch.no_grad():
        for i, (images, labels) in enumerate(testloader):
            # images = Variable(images.view(-1, 224 * 224 * 3))
            # labels = Variable(labels)
            inputs = images.to(device=device, dtype=dtype)
            labels = labels.to(device=device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for i in range(5):
                for j in range(5):
                    if images_so_far < num_images:
                        imshowaxis(ax[i][j], inputs.cpu().data[images_so_far], CLASSES[labels[images_so_far]], CLASSES[preds[images_so_far]])
                    else:
                        model.train(mode=was_training)
                        return
                    images_so_far += 1
        model.train(mode=was_training)

vis_model(model)
