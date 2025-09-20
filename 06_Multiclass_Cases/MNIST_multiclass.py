import torch
import torchvision
from torch import nn
from torchvision import datasets as ds
from torch.utils.data import DataLoader
from torch.utils.data import datapipes
from torchvision.transforms import ToTensor

torch.manual_seed(42)

train_mnist = ds.MNIST(train = True, download=True, root="./data_twice", transform=ToTensor())
test_mnist = ds.MNIST(train = False, download=True, root="./data_twice", transform=ToTensor())

train_dataloader = DataLoader(train_mnist, shuffle=True, batch_size=32)
test_dataloader = DataLoader(train_mnist, shuffle=False, batch_size=32)

for X, y in train_dataloader:
    print(X.shape)
    print(y.shape)
    break

for X, y in train_dataloader:
    print(X.shape)
    print(y.shape)
    X = X.reshape(-1, 784)
    y = y.reshape(-1, 1)
    print(X.shape)
    print(y.shape)
    break

for X, y in train_dataloader:
    X = X.reshape(-1, 784)
    y = y.type(torch.float32).reshape((-1, 1))

model = nn.Sequential(
    nn.Linear(784, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for iter in range(10):
    total_loss = 0
    optimizer.zero_grad()
    outputs = model(X)
    loss = loss_fn(outputs, y)
    loss.backward()
    optimizer.step()
    total_loss+=loss.item()
    print(total_loss)