# Softmax helps larger values stand out 
# helps that the value always stays positive
# cross entropyloss comes directly with softmax

import torch
import torchvision
from torch import nn
from torchvision import datasets as ds
from torch.utils.data import DataLoader
from torch.utils.data import datapipes
from torchvision.transforms import ToTensor

torch.manual_seed(42)

# we can not train the model with the y vector on the actual numbers, this doesnt work

train_mnist = ds.MNIST(train = True, download=True, root="./data_twice", transform=ToTensor())
test_mnist = ds.MNIST(train = False, download=True, root="./data_twice", transform=ToTensor())

train_dataloader = DataLoader(train_mnist, shuffle=True, batch_size=32)
test_dataloader = DataLoader(test_mnist, shuffle=False, batch_size=32)

for X, y in train_dataloader:
    print(X.shape)
    print(y.shape)
    break

for X, y in train_dataloader:
    print(X.shape)
    print(y.shape)
    X = X.reshape(-1, 784)
    y = nn.functional.one_hot(y, num_classes=10).type(dtype=torch.float32)
    print(X.shape)
    print(y.shape)
    break

model = nn.Sequential(
    nn.Linear(784, 100),
    nn.ReLU(),
    nn.Linear(100, 10)
)

loss_fn = nn.CrossEntropyLoss() # makes sense if a target may belong to several classes
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for iter in range(10):
    total_loss = 0
    for X, y in train_dataloader:
        X = X.reshape(-1, 784)
        y = nn.functional.one_hot(y, num_classes=10).type(dtype=torch.float32)
        optimizer.zero_grad() #das MUSS kurz vor dem model kommen und muss für jeden bacth zurückgesetzt werden un nicht nach der epoche, sonst sammelt sich der gradient einfach an
        outputs = model(X)
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    print(total_loss)

# with multiple classes we cant use a numerical scale to represent this
# we can there for not average class probabilities of 80% sure an image it is a cat /nr. 7) and 20 % sure its a dog (2)
# on average we land than by the number of a bird class number around 3
model.eval()
with torch.no_grad():
    accurate = 0
    total = 0
    for X, y in test_dataloader:
        X = X.reshape((-1, 784))
        # y = F.one_hot(y, num_classes=10).type(torch.float32)

        outputs = nn.functional.softmax(model(X), dim=1)
        print(outputs.sum(dim=1))
        #print(y)
        #print(outputs.max(dim=1).indices)
        #print(outputs)
        #break
        correct_pred = (y == outputs.max(dim=1).indices)
        total+=correct_pred.size(0)
        accurate+=correct_pred.type(torch.int).sum().item()
    print(accurate / total)

    #vs. CrossEntropyLoss: CrossEntropyLoss is designed for multiclass problems. 
    # It implicitly uses a Softmax function, which forces the output probabilities 
    # across all classes to sum to 1. This is fundamentally incorrect for multi-label problems, 
    # where probabilities should not sum to 1 because multiple classes can be active simultaneously. 
    # Using CrossEntropyLoss for a multi-label problem would incorrectly penalize the model for 
    # predicting multiple positive labels.

    #BCE might be good at the end if we want to tag an image or movie for examle with 3 max labels like Action, scifi and ahorror
    #or an image with cat, hat, and pillow to recogniz several stuff
    #there probas do not need add up to 100 %
    # original MNIST code was a multiclass problem (an image is one digit), 
    # so BCEWithLogitsLoss was not the ideal choice there (though it can be made to work by treating it as 10 independent binary problems, 
    # it's less direct than CrossEntropyLoss).' 

    # I should use BCEWithLogitsLoss (along with targets formatted as binary vectors) when my problem fits the definition 
    # of multi-label classification, meaning: An instance can belong to zero, one, or multiple classes at the same time.
    # The presence or absence of one label is independent of the presence or absence of other labels.
    # wichtig ist dass ich batches einen forecasten und bearbeiten muss aber die ergebnisse der batches kumulieren muss im loss
    # damit ich nicht die ergebnisse auf dem letzten batch bekomme 